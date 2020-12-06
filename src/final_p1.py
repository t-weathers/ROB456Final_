#!/usr/bin/env python
import rospy
import heapq
import tf2_geometry_msgs
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, PointStamped, Twist
import tf2_ros
import math
from visualization_msgs.msg import Marker
from plotting_test import *
import cv2
from tf.transformations import euler_from_quaternion

class RVizAStar(object):
    """
    Major steps(Callbacks)
    1: Read the map into the map_data attribute
    2: Read the odometry position in.

    All of the below will be run during both of the map and the odom callbacks

    Major steps(Planning and Plotting Code)
    1: See if map and odom are populated be populated
    2: Rearrange to the proper 2D array format
    3: Get robot pos
    4: Run A* with the given robot pos as a start and the hardcoded pos as a finish
    5: Convert all of the coordinates within the path to odom_pos
    6: Plot all of the locations using that Rviz plotting code


    Part 3 Todo: Reference(http://wiki.ros.org/gmapping)
    -How to not have map change while we are moving
    -Can we suppress publishing of gmapping
    -We only want gmapping to publish to /map in between steps

    Part 3 Steps:
    while not filled
        1: Read in SLAM map
        2: Use Thomas's code to find goal point
        3: plan a path to goal point using A*
        4: Move to goal point using P2 Code
    """
    def __init__(self):
        self.map_data = None
        self.odom_pos = None
        self.robot_map_pos = None
        self.map_metadata = None
        self.binary_occupancy_grid = None
        self.occ_path = None
        self.map_path = None
        self.problem_started = False
        self.driving = False
        self.pre_orientation_completed = False
        self.pre_orientation_thresh = 20#Degrees around center
        self.lin_x_feedfwd = None
        self.ang_z_feedfwd = None
        self.robot_map_pos_z_ori = None
        self.robot_move_start_time = None
        self.traj_generated = False
        self.at_goal = False
        self.prev_ang_err = 0
        self.prev_x_err = 0
        self.prev_time = 0

        #Movement parameters
        self.x_speed = 1
        self.x_back = .2
        self.x_accel_fwd = .2
        self.x_accel_bkwd = .1

        #Thresholding values for decoding the occupancy grid from ROS
        self.empty_thresh = 20

        #Coordinate Conversion Stuff
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        """Because in this example the only publisher to the '/map' topic is the
        original loader thing, the map subscriber will likely only be called once.
        Even if this runs after the map publishes, that is okay because it is 
        a latched topic which means that it will still get the latest broadcast
        """
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.marker_pub = rospy.Publisher('/marker', Marker, queue_size=2,latch = True)
        self.pub = rospy.Publisher('/recieved_map', Bool, queue_size=10)
        # publish twist message
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #To Try ***** Make a map publisher and call before transforming each time)

    def map_callback(self, msg):
        #First map callback for P2 comes from imported PGM file, for P3, this won't happen
        rospy.loginfo('In map callback')
        print "Map Recieved"
        self.map_data = msg.data #
        self.map_metadata = msg.info
        self.pub.publish(True)
        self.problem_one()

    def odom_callback(self, msg):
        rospy.loginfo('In odom callback' + str(self.odom_pos))
        self.odom_pos = msg.pose.pose
        try:
            h = Header()
            h.stamp = rospy.get_rostime()
            h.frame_id = 'base_footprint'
            pose = PoseStamped(h, Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)))

            # How do we get this transform to stay correct while moving
            # How do we keep this 'map' frame the same while we are moving
            # ***TO TRY**** Publishing our own saved version of the map which is the one that we used for A*
            new_pose = self.tf_buffer.transform(pose, 'map', rospy.Duration(1.0))
            rospy.logdebug(new_pose)
            self.robot_map_pos = new_pose.pose
            ori = self.robot_map_pos.orientation
            (r, p, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.robot_map_pos_z_ori = yaw

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return

        #Check if the robot is supposed to be driving
        if self.driving:
            #print "trying to drive"
            #Calculate the speed
            command = Twist()

            # Fill in the fields.  Field values are unspecified
            # until they are actually assigned. The Twist message
            # holds linear and angular velocities.
            command.linear.y = 0.0
            command.linear.z = 0.0
            command.angular.x = 0.0
            command.angular.y = 0.0

            #Turn the robot to be facing the correct direction before starting to move
            if not self.pre_orientation_completed and self.traj_generated:
                #rotate until within the angular velocity range
                ang_goal = math.atan2(self.map_path[1][1]-self.map_path[0][1],self.map_path[1][0]-self.map_path[0][0])
                if abs(self.robot_map_pos_z_ori - ang_goal) > 0.1: #within the goal thres
                    init_ang_dir = (self.robot_map_pos_z_ori - ang_goal - np.pi)/abs(self.robot_map_pos_z_ori - ang_goal - np.pi)
                    command.angular.z = init_ang_dir * 1.5;
                else:
                    command.angular.z = 0
                    self.pre_orientation_completed = True
                    self.robot_move_start_time = rospy.get_time()
                    self.prev_time = self.robot_move_start_time
                command.linear.x = 0
            elif self.traj_generated:
                #This is where we do the actual path following
                kp_x = 6
                kd_x = 0.5
                kp_ang = 4
                kd_ang = 0.5

                curr_time = rospy.get_time() - self.robot_move_start_time
                robot_pos_tuple = (self.robot_map_pos.position.x,self.robot_map_pos.position.y)

                feed_forward_x = self.lin_x_feedfwd(curr_time)
                e_x = self.lin_err(curr_time, robot_pos_tuple, self.robot_map_pos_z_ori)
                e_x = 0 if e_x < 0 else e_x
                de_x_dt = (e_x-self.prev_x_err)/(curr_time-self.prev_time)
                self.prev_x_err = e_x
                feedback_x = kp_x * e_x + kd_x*de_x_dt

                feed_forward_z = self.ang_z_feedfwd(curr_time)
                e_ang = self.ang_err(curr_time, robot_pos_tuple, self.robot_map_pos_z_ori)
                e_ang = (e_ang + np.pi) % (2*np.pi) - np.pi
                de_ang_dt = (e_ang - self.prev_ang_err) / (curr_time - self.prev_time)
                self.prev_ang_err = e_ang
                feedback_z = kp_ang * e_ang + kd_ang * de_ang_dt
                #print "robot_ori :" + str(self.robot_map_pos_z_ori)
                #print "e_ang: " + str(e_ang)
                #print "path_dir: " + str(e_ang + self.robot_map_pos_z_ori)

                command.linear.x = 0*feed_forward_x + feedback_x
                command.angular.z = 0*feed_forward_z + feedback_z

                if np.linalg.norm(np.array(robot_pos_tuple) - np.array(self.map_path[-1])) < 0.2:
                    self.at_goal = True
                    command.linear.x = 0
                    command.angular.z = 0

                self.prev_time = curr_time
            self.move_pub.publish(command)



        self.problem_one()

    def problem_one(self):
        """
        This is the method that will complete all of the code for problem one
        """

        #Step 1: Ensure that both the map and the odom_pos are populated
        if self.map_data is not None and self.robot_map_pos is not None and not self.problem_started:
            self.problem_started = True
            #rospy.sleep(2)
            #Step 2: Rearrange the row major data to the proper 2D array format
            self.reshape_thresh_occupancy_grid(8)
            #Step 3: Convert the robot_map_pos to map_pos and then to the image_coords
            end_map_pos = Pose(Point(-6,-1,0),Quaternion(0,0,0,1))

            start_occ_loc = self.map_pos_to_occ_grid(self.robot_map_pos)
            #print self.robot_map_pos.position.x
            #print self.robot_map_pos.position.y_grid(self.robot_map_pos)
            end_occ_loc = self.map_pos_to_occ_grid(end_map_pos)

            #Plot these points for reference
            #self.draw_points([(self.robot_map_pos.position.x, self.robot_map_pos.position.y), \
                              #(-2,1)],self.marker_pub,True)

            #print start_occ_loc
            #print end_occ_loc
            #print self.binary_occupancy_grid
            #print self.binary_occupancy_grid[int(start_occ_loc[0]),int(start_occ_loc[1])]
            #print self.binary_occupancy_grid[int(end_occ_loc[0]), int(end_occ_loc[1])]
            #Step 4: Run A* with the given occupancy grid and the start and end locations
            self.occ_path = self.A_star(self.binary_occupancy_grid,start_occ_loc,end_occ_loc)

            #print self.occ_path

            #Step 5 Convert all to map_pos
            self.map_path = [self.occ_grid_to_map_pos(loc) for loc in self.occ_path]

            #print self.map_path

            #Step 6 Plot to Rviz
            #self.draw_points(self.map_path,self.marker_pub,False)
            #rospy.sleep(1000)
            #Step 7 Trajectory generation stuff
            self.traj_generate(self.map_path)

            self.driving = True

    def traj_generate(self,map_path):
        #Define a function from the map_path that will be used to grab the the

        times = [0]
        ang_vel = [0]
        lin_vel = [0]
        x_vals = [map_path[0][0]]
        y_vals = [map_path[0][1]]
        #Create the times array and the angular velocity array
        for i in range(1,len(map_path)):
            segment_length = np.linalg.norm(np.array(map_path[i])-np.array(map_path[i-1]))
            if i > 1:
                segment_angle_change = math.atan2(map_path[i][1] - map_path[i - 1][1], \
                                                  map_path[i][0] - map_path[i - 1][0]) - \
                                       math.atan2(map_path[i - 1][1] - map_path[i - 2][1], \
                                                  map_path[i - 1][0] - map_path[i - 2][0])
            else:
                segment_angle_change = 0


            #Create the angular velocity array


            #This could be a little bit weird

            lin_vel.append(min(self.x_speed/(1 + abs(segment_angle_change)/(np.pi/8)),math.sqrt(lin_vel[-1]**2 + 2*self.x_accel_fwd*segment_length)) )
            #print "lin vel: " + str(lin_vel[-1])
            segment_time = segment_length / lin_vel[-1]  # This is the time that it will take the robot to drive this segment
            times.append(times[-1] + segment_time)
            ang_vel.append(segment_angle_change / segment_time)

            x_vals.append(map_path[i][0])
            y_vals.append(map_path[i][1])

        #Reverse accelleration limits
        #print "times:"
        #print times
        #rospy.sleep(5)
        lin_vel[-1] = 0
        for i in range(len(map_path)-1, 1, -1):
            segment_length = np.linalg.norm(np.array(map_path[i]) - np.array(map_path[i - 1]))
            lin_vel[i-1] = min(lin_vel[i-1],math.sqrt(lin_vel[i]**2 + 2*self.x_accel_bkwd*segment_length))
            segment_time = segment_length / lin_vel[i-1]
            diff_time = segment_time - (times[i] - times[i-1])
            #print "segment_times" + str(segment_time)
            #print "diff time " + str(diff_time)
            #print "segment_length " + str(segment_length)
            #print "lin vel: " + str(lin_vel[i-1])
            #Shift all of the times
            for j in range(i,len(map_path)):
                times[j] += diff_time
            #print times
            #rospy.sleep(1)


        #print "times:"
        #print times
        #rospy.sleep(5)

            #Create the feedforward functions for the linear and the angular velocity of the robot
        self.lin_x_feedfwd = lambda t: np.interp(t,times,lin_vel) if t < max(times) else 0
        self.ang_z_feedfwd = lambda t: np.interp(t,times,ang_vel) if t < max(times) else 0
        self.expected_point = lambda t: (np.interp(t,times,x_vals),np.interp(t,times,y_vals))
        lin_spacer = .04
        self.lin_err = lambda t, robot_pos,robot_dir: np.dot((np.cos(robot_dir),np.sin(robot_dir)),(np.array(self.expected_point(t))-np.array(robot_pos))) - self.x_back
        self.ang_err = lambda t, robot_pos,robot_dir: 0 - robot_dir + math.atan2(self.expected_point(t)[1] - robot_pos[1], \
                                                                             self.expected_point(t)[0] - robot_pos[0])
        #print "traj_generation done"
        self.traj_generated = True

    def map_pos_to_occ_grid(self,map_pos):
        """
        This function will take in a pose that is in the global map frame and will convert it to which "bin"
        of the occupancy grid it is in.
        :param map_pos: This is the position within the map frame that needs to be converted
        :return: occ_loc (x,y) or False if it is actually within the width of the map
        """
        dx = map_pos.position.x - self.map_metadata.origin.position.x
        dy = map_pos.position.y - self.map_metadata.origin.position.y

        #discretize the x and the y coords
        i_x = math.floor(dx/self.map_metadata.resolution)
        i_y = math.floor(dy/self.map_metadata.resolution)

        if i_x >= self.map_metadata.width or i_y >= self.map_metadata.height:
            return False
        else:
            occ_loc = (i_x, i_y)
            return occ_loc

    def occ_grid_to_map_pos(self,occ_loc):
        """
        This function will convert from the x and y indices in the occupancy grid to the location of the center of
        that cell within the global map frame
        :param occ_loc: This is a tuple of the x and y indices within the occupancy grid that needs to be converted
        :return: map_pos_tuple: This is a tuple of the x and y coordinates in the upper frame
        """
        i_x = occ_loc[0]
        i_y = occ_loc[1]

        #Adding the 0.5 puts it in the center of the cell
        x_loc = self.map_metadata.origin.position.x + (i_x + 0.5) * self.map_metadata.resolution
        y_loc = self.map_metadata.origin.position.y + (i_y + 0.5) * self.map_metadata.resolution

        return (x_loc, y_loc)


    def reshape_thresh_occupancy_grid(self,buffer = 0):
        """
        This function will reshape the map_data and threshold it
        Converts from probabilities to Booleans

        ***TODO***
        need to change to array of -1,0,1 to represent unknowns. Will need to integrate these changes downstream
        :param buffer: this will add a buffer of the specified size in a circle around each of the
        :return: binary_occupancy which is an array of True and False corresponding to occupancy
        """
        if self.map_data is not None:
            #First, threshold all of the of the values
            #print self.map_data
            #inline code
            binary_occupancy = []
            #binary_occupancy = [prob > self.empty_thresh for prob in self.map_data]
            #Testing Code
            for prob in self.map_data:
                binary_occupancy.append(prob > self.empty_thresh)
                if prob > self.empty_thresh:
                    #print "occupied"
                    #print len(binary_occupancy)
                    pass
            #Reshape the row major array
            self.binary_occupancy_grid = np.reshape(binary_occupancy,(self.map_metadata.height,self.map_metadata.width))
            self.binary_occupancy_grid = np.transpose(self.binary_occupancy_grid)

            grid_buffer = self.binary_occupancy_grid.copy()
            if buffer > 0:
                #print "blow er up"
                #print "*****************************************"
                #print "****************************************"
                #Add a buffer of occupied pixels around all of the obstacles
                grid_shape = np.shape(self.binary_occupancy_grid)
                grid_x = grid_shape[0]
                grid_y = grid_shape[1]
                for i in range(grid_x):
                    for j in range(grid_y):
                        #Check and see if the that one is filled
                        if self.binary_occupancy_grid[i,j]:
                            #Add to all of the cells that are within the specified distance
                            for dx in range(-buffer,buffer):
                                for dy in range(-buffer,buffer):
                                    #Check if that cell is already filled
                                    if not grid_buffer[i+dx,j+dy]:
                                        #Check and see if that cell is within the specified distance
                                        if dx**2+dy**2 <= buffer**2:
                                            grid_buffer[i + dx, j + dy] = True
                                            #print "changed a cell"
                self.binary_occupancy_grid = grid_buffer

            cv2.imwrite("/home/bamberjo/threshed.png",self.binary_occupancy_grid*255)
        else:
            raise ValueError("There was no map data")

    def A_star(self, map, start, goal):
        dimx = len(map[0])
        dimy = len(map)
        pq = []  # (weight, (x,y))
        heapq.heappush(pq, (0, start))
        visited = {}  # # node : (distance , (prev node))
        visited[start] = None  # (startx,starty): (prevx, prevy)
        distance_from_start = {}
        distance_from_start[start] = 0
        while pq != []:
            current = heapq.heappop(pq)
            #print "New pop step:"
            #print "X: " + str(current[1][0]-start[0])
            #print "Y: " + str(current[1][1] -start[1])
            #print "Weight: " + str(current[0])
            #rospy.sleep(1)
            #print current
            #print goal
            if current[1] == goal:
                print("goal node reached")
                break  # success
            for i in self.neighbors(map, current[1][0], current[1][1], dimx, dimy):
                #print distance_from_start
                temp_distance = distance_from_start[current[1]] + self.diag_distance(current[1],i)
                #print "new"
                #print "diag           : " +  str(self.diag_distance(current[1],i))
                #print "distance_start : " + str(distance_from_start[current[1]])
                #rospy.sleep(.1)
                if i not in distance_from_start or temp_distance < distance_from_start[i]:  # not yet seen node
                    if i in distance_from_start:
                        #print temp_distance
                        #print distance_from_start[i]
                        if temp_distance < distance_from_start[i]:
                            #print "overwriting with shorter"
                            pass
                    distance_from_start[i] = temp_distance

                    #A*
                    #heapq.heappush(pq, (temp_distance + self.diag_distance(i,goal),i))
                    heapq.heappush(pq, (temp_distance + self.diag_distance(current[1],goal), i))
                    # setup for backtrace
                    visited[i] = current[1]

        # backtracking
        if pq == []:
            #print("failed to find path")
            path = []
        else:
            #print "reconstructing"
            current = goal
            path = []
            #print current
            while current != start:
                #print goal
                path.insert(0, current)
                #print path
                current = visited[current]
            #Try to visualize
            blank = np.zeros(np.shape(map)) + 255

            for i in range(dimx):
                for j in range(dimy):
                    if (i,j) in distance_from_start:
                        blank[i,j] = distance_from_start[(i,j)] if distance_from_start[(i,j)] < 255 else 255

            cv2.imwrite("/home/bamberjo/flooded.png",blank)
        return path

    def neighbors(self, arr, x, y, dimx, dimy):
        """
        ***TODO***
        need to update with an is_occupied function to support -1,0,1 format
        :param arr:
        :param x:
        :param y:
        :param dimx:
        :param dimy:
        :return: all open neighboring pixels
        """
        if x >= dimx:
            return -1
        elif y >= dimy:
            return -1
        elif arr == []:
            return -1

        #print "x: " + str(x)
        #print "y: " + str(y)

        output = []  # above, above left, left, bottom left, below, below right, right, above right)
        if y > 0:
            if not arr[int(x), int(y - 1)]:
                output.append((x, y - 1))  # above
        if x > 0 and y > 0:
            if not arr[int(x - 1), int(y - 1)]:
                output.append((x - 1, y - 1))  # left-above
        if y > 0 and x < dimx - 1:
            if not arr[int(x + 1), int(y - 1)]:
                output.append((x + 1, y - 1))  # right-above
        if x > 0:
            if not arr[int(x - 1), int(y)]:
                output.append((x - 1, y))  # left
        if x > 0 and y < dimy - 1:
            if not arr[int(x - 1), int(y + 1)]:
                output.append((x - 1, y + 1))  # left below
        if x < dimx - 1 and y < dimy - 1:
            if not arr[int(x + 1), int(y + 1)]:
                output.append((x + 1, y + 1))  # right below
        if y < dimy - 1:
            if not arr[int(x), int(y + 1)]:
                output.append((x, y + 1))  # below
        if x < (dimx - 1):
            if not arr[int(x + 1), int(y)]:
                output.append((x + 1, y))  # right
        return output

    def diag_distance(self, node, end):
        dx = abs(node[0] - end[0])
        dy = abs(node[1] - end[1])
        D2 = math.sqrt(2)
        D = 1
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def draw_points(self, points, pub, to_delay):
        """
        Plots an array of points [(x, y)...] in rviz
        :param: points iterable of (x, y) pairs. If a numpy array, shape should be (n, 2)
        :return: None
        """
        #print "trying to plot those suckers"
        msg = Marker()
        # Marker header specifies what (and when) it is drawn relative to
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        # uint8 POINTS=8
        msg.type = 8
        # Disappear after 1sec. Comment this line out to make them persist indefinitely
        #msg.lifetime = rospy.rostime.Duration(1000, 0)
        # Set marker visual properties
        msg.color.b = 1.0
        msg.color.a = 1.0
        msg.scale.x = 0.03
        msg.scale.y = 0.03
        # Copy  (x, y) into message and publish
        for (x, y) in points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.1  # Places all points 10cm above the ground
            msg.points.append(p)
        # for i in range(500):
        #  pub.publish(msg)
        #  rospy.sleep(0.1)
        if to_delay:
            rospy.sleep(5)
        pub.publish(msg)


def locate_unexplored(Ogrid, start):
     queue = [];
     dimx = len(Ogrid)
     dimy = len(Ogrid[1])
     visited = {}    #(node: parent)
     queue.append(start)
     visited[start] = 0
     goal = start #set goal val to a temp of start, this is the return value
     while queue != []:
         current = queue.pop(0)
         if Ogrid[current[0]][current[1]] == -1: #unmapped area
             goal = current
             break
         for i in neighbors(Ogrid,current[0],current[1],dimx,dimy):
             if i not in visited and i not in queue:
                 queue.append(i)
                 visited[i] = current
     if goal != start:
         #print("new goal location found: " + goal)
         return goal #x,y location
     else:
         #print("no unexplored territory remains")
         return 0

if __name__ == '__main__':
    rospy.init_node('final_p1')
    rate = rospy.Rate(10)
    gp = RVizAStar()
    rospy.spin()
