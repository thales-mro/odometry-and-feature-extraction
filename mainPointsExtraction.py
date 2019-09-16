import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
from robot import Robot
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from time import time

braitenbergL=[-0.2,-0.4,-0.6,-0.8,-1.0,-1.2,-1.4,-1.6]
braitenbergR=[-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2]

detect = [0,0,0,0,0,0,0,0]
noDetectionDist = 1.0
maxDetectionDist = 0.4



def main():

	robot = Robot()
	stop = False
	all_x = []
	all_y = []
	robot_trajectory = []


	if(robot.get_connection_status() != -1):
		
		
		all_x, all_y, robot_trajectory = forward(robot, 3.15, all_x, all_y, robot_trajectory) # 3.15
		all_x, all_y = rotate(robot, degreesToRadians(82.0), 1, all_x, all_y) # 82
		all_x, all_y, robot_trajectory = forward(robot, 4.6, all_x, all_y, robot_trajectory) # 4.6

		
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90
		all_x, all_y, robot_trajectory = forward(robot, 2.2, all_x, all_y, robot_trajectory) # 2.2
		all_x, all_y = rotate(robot, degreesToRadians(88.0), 1, all_x, all_y) # 88.0
		all_x, all_y, robot_trajectory = forward(robot, 2.4, all_x, all_y, robot_trajectory) # 2.4
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 4.5, all_x, all_y, robot_trajectory) # 4.5
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 2.5, all_x, all_y, robot_trajectory) # 2.5
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 1.7, all_x, all_y, robot_trajectory) # 1.7
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 4.0, all_x, all_y, robot_trajectory) # 4.0

		## So far its tested!
		all_x, all_y = rotate(robot, degreesToRadians(95.0), 0, all_x, all_y) # 95.0
		all_x, all_y, robot_trajectory = forward(robot, 7.35, all_x, all_y, robot_trajectory) # 7.35
		
		all_x, all_y = rotate(robot, degreesToRadians(82.0), 0, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 4.0, all_x, all_y, robot_trajectory) # 4.0
		
		all_x, all_y = rotate(robot, degreesToRadians(86.0), 0, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 3.45, all_x, all_y, robot_trajectory)
		
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90.0
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90.0
		all_x, all_y, robot_trajectory = forward(robot, 3.45, all_x, all_y, robot_trajectory)

		
		'''
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 4.0, all_x, all_y)
		
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 4.0, all_x, all_y) # 1.9

		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 2.5, all_x, all_y) # 1.9

		'''
		'''
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		
		all_x, all_y = forward(robot, 1.10, all_x, all_y) 
		all_x, all_y = rotate(robot, degreesToRadians(35.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 0.85, all_x, all_y) 
		all_x, all_y = rotate(robot, degreesToRadians(35.0), 0, all_x, all_y) # 90.0
		
		all_x, all_y = forward(robot, 0.9, all_x, all_y) # 1.1
		all_x, all_y = rotate(robot, degreesToRadians(10.0), 0, all_x, all_y) # 90.0
		
		all_x, all_y = forward(robot, 1.15, all_x, all_y) # 0.8 + 0.25
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 1.5, all_x, all_y)

		all_x, all_y = rotate(robot, degreesToRadians(10.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 4.5, all_x, all_y) # 2.0
		all_x, all_y = rotate(robot, degreesToRadians(70.0), 1, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 4.0, all_x, all_y) # 0.3
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 90.0
		all_x, all_y = forward(robot, 1.0, all_x, all_y) # 0.3

		all_x, all_y = rotate(robot, degreesToRadians(2.5), 1, all_x, all_y) # 57.5
		all_x, all_y = forward(robot, 1.9, all_x, all_y) # 1.9
		all_x, all_y = rotate(robot, degreesToRadians(15.0), 0, all_x, all_y) # 57.5
		all_x, all_y = forward(robot, 1.5, all_x, all_y) # 1.9
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 57.5
		all_x, all_y = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y) # 57.5
		'''

	robot_trajectory = np.array(robot_trajectory)
	plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
	plt.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'go')
	plt.show()


def degreesToRadians(degree_angle):
	radian_angle = degree_angle*np.pi/180
	return radian_angle


def forward(robot, dist, all_x, all_y, robot_trajectory):

	stop = False
	ref0_pos = np.array(robot.get_current_position())
	while(robot.get_connection_status() != -1 and not stop):
		ir_distances = robot.read_laser()
		ir_distances = np.array(ir_distances).reshape(len(ir_distances)//3,3)[::150]

		robot_angles = robot.get_current_orientation()
		theta = robot_angles[2]

		plot_rot_x, plot_rot_y = rotation(theta, ir_distances[:,0], ir_distances[:,1])
		robot_pos = robot.get_current_position()
		plot_x, plot_y = translation(robot_pos[0], robot_pos[1], plot_rot_x, plot_rot_y)

		robot_trajectory.append([robot_pos[0], robot_pos[1]])
		all_x.extend(plot_x)
		all_y.extend(plot_y)	

		ref1_pos = np.array(robot.get_current_position())

		if norm(ref1_pos - ref0_pos) >= dist:
			robot.set_left_velocity(0.0)
			robot.set_right_velocity(0.0)
			stop = True

		else:
			robot.set_left_velocity(3.0)
			robot.set_right_velocity(3.0)


	return all_x, all_y, robot_trajectory



		


def rotate(robot, angle, orientation, all_x, all_y):

	stop = False
	robot_angles = robot.get_current_orientation()
	ref0_theta = robot_angles[2]

	while(robot.get_connection_status() != -1 and not stop):
		ir_distances = robot.read_laser()
		ir_distances = np.array(ir_distances).reshape(len(ir_distances)//3,3)[::150]

		robot_angles = robot.get_current_orientation()
		theta = robot_angles[2]

		plot_rot_x, plot_rot_y = rotation(theta, ir_distances[:,0], ir_distances[:,1])
		robot_pos = robot.get_current_position()
		plot_x, plot_y = translation(robot_pos[0], robot_pos[1], plot_rot_x, plot_rot_y)

		all_x.extend(plot_x)
		all_y.extend(plot_y)	

		robot_angles = robot.get_current_orientation()
		ref1_theta = robot_angles[2]

		# Robot rotates to right
		if orientation == 0:

			if ref0_theta <= 0.0 and ref1_theta >= 0.0:

				if (ref0_theta + np.pi) + (np.pi - ref1_theta) < angle:
					robot.set_left_velocity(0.5)
					robot.set_right_velocity(-0.5)
				else:
					robot.set_left_velocity(0.0)
					robot.set_right_velocity(0.0)
					stop = True

			elif abs(ref1_theta - ref0_theta) < angle:
				robot.set_left_velocity(0.5)
				robot.set_right_velocity(-0.5)

			else:
				robot.set_left_velocity(0.0)
				robot.set_right_velocity(0.0)
				stop = True

		#Robot rotates to left
		elif orientation == 1:

			if ref0_theta >= 0.0 and ref1_theta <= 0.0:
				#print(np.pi - ref0_theta, ref1_theta + np.pi, angle)
				if (ref1_theta + np.pi) + (np.pi - ref0_theta) < angle:
					robot.set_left_velocity(-0.5)
					robot.set_right_velocity(0.5)
				else:
					robot.set_left_velocity(0.0)
					robot.set_right_velocity(0.0)
					stop = True

			elif abs(ref1_theta - ref0_theta) < angle:
				robot.set_left_velocity(-0.5)
				robot.set_right_velocity(0.5)

			else:
				robot.set_left_velocity(0.0)
				robot.set_right_velocity(0.0)
				stop = True



	return all_x, all_y


	

def braitenberg(dist, vel):
    """
        Control the robot movement by the distances read with the ultrassonic sensors. More info: https://en.wikipedia.org/wiki/Braitenberg_vehicle
        Args:
            dist: Ultrassonic distances list
            vel:  Max wheel velocities
    """
    vLeft = vRight = vel
    for i in range(len(dist)):
        if(dist[i] < noDetectionDist):
            detect[i] = 1 - ((dist[i]-maxDetectionDist)/(noDetectionDist-maxDetectionDist))
        else:
            detect[i]=0
        for i in range(8):
            vLeft = vLeft + braitenbergL[i]*detect[i]
            vRight = vRight+ braitenbergR[i]*detect[i]

    return [vLeft, vRight]


def rotation(theta, x, y):
	rotated_x = np.cos(theta)*x - np.sin(theta)*y
	rotated_y = np.sin(theta)*x + np.cos(theta)*y
	return rotated_x, rotated_y

def translation(robot_x, robot_y, x, y):
	translated_x = robot_x + x
	translated_y = robot_y + y
	return translated_x, translated_y

def filterAngles(threshold, num_points, all_x, all_y):

	#print(len(vectors))
	if len(all_x) > num_points:

		vectors = np.array(list(zip(all_x, all_y)))
		j = 0

		while(j != len(vectors)):
			ref = vectors[j]
			N = len(vectors)
			remove_angles = []
			for i in range(j+1, N):
				cos = np.dot(ref, vectors[i])/(norm(ref)*norm(vectors[i]))
				if(cos > threshold):
					remove_angles.append(i)

			vectors = np.delete(vectors, remove_angles, 0)
			j += 1

		#print(len(vectors))
		all_x = vectors[:,0].tolist()
		all_y = vectors[:,1].tolist()

	return all_x, all_y

	





if __name__ == '__main__':
	main()


## Backup
'''
robot = Robot()
	stop = False
	all_x = []
	all_y = []

	t0 = time()
	ref0_pos = np.array(robot.get_current_position())

	robot_angles = robot.get_current_orientation()
	ref0_theta = robot_angles[2]
	ref1_theta = robot_angles[2]

	while(robot.get_connection_status() != -1 and not stop):
		ir_distances = robot.read_laser()
		#vel = braitenberg(us_distances[:8], 3) #Using only the 8 frontal sensors
		#robot.set_left_velocity(vel[0])
		#robot.set_right_velocity(vel[1])
		ir_distances = np.array(ir_distances).reshape(len(ir_distances)//3,3)

		#filterX_ir_distances = ir_distances[ir_distances[:,0] >= 0.0]
		#filterY_ir_distances = filterX_ir_distances[filterX_ir_distances[:,1] >= 0.0]

		robot_angles = robot.get_current_orientation()
		theta = robot_angles[2]

		
		#print(ir_distances[0])
		plot_rot_x, plot_rot_y = rotation(theta, ir_distances[:,0], ir_distances[:,1])
		robot_pos = robot.get_current_position()
		plot_x, plot_y = translation(robot_pos[0], robot_pos[1], plot_rot_x, plot_rot_y)

		
		#print(plot_x[0], plot_y[0])
		all_x.extend(plot_x)
		all_y.extend(plot_y)	

		
		if 0.0 < theta < 0.1:
			one_rotation = False
			robot.set_left_velocity(0)
			robot.set_right_velocity(0)
		


		if len(all_x) > 10.000:
			vectors = np.array(list(zip(all_x, all_y)))
			all_x, all_y = filterAngles(0.9999, vectors)

		ref1_pos = np.array(robot.get_current_position())
		robot_angles = robot.get_current_orientation()
		ref1_theta = robot_angles[2]

		if norm(ref1_pos - ref0_pos) >= 2:
			robot.set_left_velocity(-0.5)
			robot.set_right_velocity(0.5)

		if ref0_theta + np.pi//2 < ref1_theta:
			robot.set_left_velocity(3.0)
			robot.set_right_velocity(3.0)



	
		us_distances = robot.read_ultrassonic_sensors()
		
		foward_distances = np.array(us_distances)[2:8]
		min_dist = np.min(foward_distances)
		
		if min_dist < maxDetectionDist:
			min_index = np.where(foward_distances == min_dist)[0][0]

			if(min_index <= 2):
				robot.set_left_velocity(0.5)
				robot.set_right_velocity(-0.5)
			else:
				robot.set_left_velocity(-0.5)
				robot.set_right_velocity(0.5)

		elif min_dist < 2:
			robot.set_left_velocity(1.5)
			robot.set_right_velocity(1.5)

		else:
			robot.set_left_velocity(3.0)
			robot.set_right_velocity(3.0)

		

		t1 = time()
		dt = t1 - t0
		if dt > 500:
			stop = True
			robot.set_left_velocity(0)
			robot.set_right_velocity(0)

		elif 30 < dt and dt < 35:
			stop = True
			robot.set_left_velocity(-0.5)
			robot.set_right_velocity(0.5)

		

		#plt.plot(-1*np.array(all_x), np.array(all_y), 'o')

		#plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
		#plt.show()

		#plt.show()
	    #print(ir_distances)
		#robot_angles = robot.get_current_orientation()

		#theta = robot_angles[2]

		#exit()

'''
