import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'lib')
from robot import Robot
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import vrep



def main():

	robot = Robot()
	#Circuit03(robot)

	all_x = []
	all_y = []
	robot_trajectory = []
	

	x0,y0,z0 = robot.get_current_position()
	
	orientation_odometry = 0
	odometry_trajectory = [[x0,y0]]
	moviment_state = [x0, y0, orientation_odometry]



	
	# Hard-coded trajectory 
	if(robot.get_connection_status() != -1):
		
		

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 3.15

		
		#res, gyroX = vrep.simxGetFloatSignal(robot.clientID, "gyroX", vrep.simx_opmode_streaming)
		#res, gyroY = vrep.simxGetFloatSignal(robot.clientID, "gyroY", vrep.simx_opmode_streaming)
		#res, gyroZ = vrep.simxGetFloatSignal(robot.clientID, "gyroZ", vrep.simx_opmode_streaming)
	
		#print(gyroX, gyroY, gyroZ)
		
		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(85.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		
		
		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 4.6

		
		
		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(85.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 4.6

		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(85.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 4.6

 
		#robot.set_left_velocity(0.0)
		#robot.set_right_velocity(2.0)
		#all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(85.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82
		

		#delta_space = len(robot_trajectory)//1000
		odometry_trajectory = np.array(odometry_trajectory)#[::delta_space]
		robot_trajectory = np.array(robot_trajectory)#[::delta_space]
		#pointsToSave = np.array([all_x, all_y])

		#print(np.array(robot_trajectory).shape)
		
		#plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
		#plt.show()

		plt.plot(-1*odometry_trajectory[:,0], -1*odometry_trajectory[:,1], '.')
		plt.plot(-1*robot_trajectory[:,0], -1*robot_trajectory[:,1], 'g.')
		plt.show()
	


def Circuit03(robot):

	all_x = []
	all_y = []
	robot_trajectory = []
	

	x0,y0,z0 = robot.get_current_position()
	
	orientation_odometry = 0
	odometry_trajectory = [[x0,y0]]
	moviment_state = [x0, y0, orientation_odometry]

	robot.set_left_velocity(0.0)
	robot.set_right_velocity(0.0)

	# Hard-coded trajectory 
	if(robot.get_connection_status() != -1):
		
		
		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 3.15, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 3.15

		
		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(82.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		
		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 4.6, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 4.6

		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.5, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory) # 4.6

		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(180.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 0.3, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		

		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(95.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82
		

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 4.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		
		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(90.0), 0, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 7.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		
		

		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(75.0), 0, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 3.5, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		

		
		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(180.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		
		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 4.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		

		robot.set_left_velocity(0.0)
		robot.set_right_velocity(2.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(70.0), 1, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 4.0, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		

		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(80.0), 0, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.9, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		
		robot.set_left_velocity(2.0)
		robot.set_right_velocity(0.0)
		all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory = rotate(robot, degreesToRadians(80.0), 0, all_x, all_y, robot_trajectory,moviment_state, odometry_trajectory) # 82

		robot.set_left_velocity(3.0)
		robot.set_right_velocity(3.0)
		all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory = forward(robot, 1.2, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory)
		


	delta_space = len(robot_trajectory)//1000
	odometry_trajectory = np.array(odometry_trajectory)[::delta_space]
	robot_trajectory = np.array(robot_trajectory)[::delta_space]
	pointsToSave = np.array([all_x, all_y])

	print(np.array(robot_trajectory).shape)
	
	plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
	plt.show()

	plt.plot(-1*odometry_trajectory[:,0], -1*odometry_trajectory[:,1], '.')
	plt.plot(-1*robot_trajectory[:,0], -1*robot_trajectory[:,1], 'g.')
	plt.show()

	np.save("ExtractedPoints.npy", pointsToSave, fix_imports=False)
	np.save("Trajectory.npy", robot_trajectory, fix_imports=False)
	np.save("Odometry.npy", odometry_trajectory, fix_imports=False)


def degreesToRadians(degree_angle):
	radian_angle = degree_angle*np.pi/180
	return radian_angle


def forward(robot, dist, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory):

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
		## ---- Odometry ---- ##
		x, y, orientation_odometry = odometry(robot, moviment_state[0], moviment_state[1], moviment_state[2])
		#print(x,y, orientation_odometry, 'f')
		odometry_trajectory.append([x,y])

		moviment_state[0] = x
		moviment_state[1] = y
		moviment_state[2] = orientation_odometry

		## ------------------- ##

		all_x.extend(plot_x)
		all_y.extend(plot_y)	

		ref1_pos = np.array(robot.get_current_position())

		if norm(ref1_pos - ref0_pos) >= dist:
			robot.set_left_velocity(0.0)
			robot.set_right_velocity(0.0)
			stop = True


	return all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory



		


def rotate(robot, angle, orientation, all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory):

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

		robot_trajectory.append([robot_pos[0], robot_pos[1]])

		## ---- Odometry ---- ##
		x, y, orientation_odometry = odometry(robot, moviment_state[0], moviment_state[1], moviment_state[2])
		#print(x,y, 'r')
		odometry_trajectory.append([x,y])

		moviment_state[0] = x
		moviment_state[1] = y
		moviment_state[2] = orientation_odometry

		## ------------------- ##

		all_x.extend(plot_x)
		all_y.extend(plot_y)	

		robot_angles = robot.get_current_orientation()
		ref1_theta = robot_angles[2]

		# Robot rotates to right
		if orientation == 0:

			if ref0_theta <= 0.0 and ref1_theta >= 0.0:

				if (ref0_theta + np.pi) + (np.pi - ref1_theta) >= angle:
					robot.set_left_velocity(0.0)
					robot.set_right_velocity(0.0)
					stop = True

			elif abs(ref1_theta - ref0_theta) >= angle:
				robot.set_left_velocity(0.0)
				robot.set_right_velocity(0.0)
				stop = True
			
		#Robot rotates to left
		elif orientation == 1:

			if ref0_theta >= 0.0 and ref1_theta <= 0.0:
				#print(np.pi - ref0_theta, ref1_theta + np.pi, angle)
				if (ref1_theta + np.pi) + (np.pi - ref0_theta) >= angle:
					robot.set_left_velocity(0.0)
					robot.set_right_velocity(0.0)
					stop = True


			elif abs(ref1_theta - ref0_theta) >= angle:
				robot.set_left_velocity(0.0)
				robot.set_right_velocity(0.0)
				stop = True


	return all_x, all_y, robot_trajectory, moviment_state, odometry_trajectory



def odometry(robot, x, y, orientation):

	angle0_left = get_left_enconder(robot)
	angle0_right = get_right_enconder(robot)

	res, gyroZ = vrep.simxGetFloatSignal(robot.clientID, "gyroZ", vrep.simx_opmode_streaming)
	
	time.sleep(0.1)

	angle1_left = get_left_enconder(robot)
	angle1_right = get_right_enconder(robot)


	if angle0_left > 0.0 and angle1_left < 0.0:
		dtheta_left = (angle1_left + np.pi) + (np.pi - angle0_left)
	else:
		dtheta_left = abs(angle1_left - angle0_left)

	if angle0_right > 0.0 and angle1_right < 0.0:
		dtheta_right = (angle1_right + np.pi) + (np.pi - angle0_right)
	else:
		dtheta_right = abs(angle1_right - angle0_right) 

	dangle_encoder = ((robot.WHEEL_RADIUS*(dtheta_right - dtheta_left))/robot.ROBOT_WIDTH) #*1.15 #-- encoder odometry
	#dangle = gyroZ #*0.75
	dangle = (0.75*gyroZ + 1.0*dangle_encoder)/2 #*0.75


	if abs(dangle) < 0.01:
		#print(dangle, orientation, x, y)
		dangle = 0.0

	if dtheta_right < 0.01:
		dtheta_right = 0.0

	if dtheta_left < 0.01:
		dtheta_left = 0.0

	ds = (robot.WHEEL_RADIUS*(dtheta_right + dtheta_left))/2

	dx = ds*np.cos(orientation + dangle/2)
	dy = ds*np.sin(orientation + dangle/2)

	print(orientation, dangle)

	return x+dx, y+dy, orientation+dangle



def get_left_enconder(robot):
	values = vrep.simxGetJointPosition(robot.clientID, robot.motors_handle["left"], vrep.simx_opmode_streaming)
	return values[1]
	
def get_right_enconder(robot):
	values = vrep.simxGetJointPosition(robot.clientID, robot.motors_handle["right"], vrep.simx_opmode_streaming)
	return values[1]

def getAngle(angle):

	if angle < 0:
		angle = angle + 2*np.pi

	return angle

def rotation(theta, x, y):
	rotated_x = np.cos(theta)*x - np.sin(theta)*y
	rotated_y = np.sin(theta)*x + np.cos(theta)*y
	return rotated_x, rotated_y

def translation(robot_x, robot_y, x, y):
	translated_x = robot_x + x
	translated_y = robot_y + y
	return translated_x, translated_y 


if __name__ == '__main__':
	main()