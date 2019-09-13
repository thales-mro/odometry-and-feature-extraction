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

	#a = np.array([-1,2,3,4,5,6,7,8,9,-10,11,12])
	#b = a.reshape(4,3)

	#c = b[b[:,0] >= 0.0]
	#print(c)
	#print(b[:,0])

	#plt.plot([1,2,3],[4,5,6], 'o')

	#plt.show()

	
	robot = Robot()
	stop = False
	all_x = []
	all_y = []

	t0 = time()
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

		'''
		if 0.0 < theta < 0.1:
			one_rotation = False
			robot.set_left_velocity(0)
			robot.set_right_velocity(0)
		'''


		if len(all_x) > 10.000:
			vectors = np.array(list(zip(all_x, all_y)))
			all_x, all_y = filterAngles(0.9999, vectors)


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

		if t1 - t0 > 500:
			stop = True
			print(t1 - t0)
			robot.set_left_velocity(0)
			robot.set_right_velocity(0)



		#plt.plot(-1*np.array(all_x), np.array(all_y), 'o')

		#plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
		#plt.show()

		#plt.show()
	    #print(ir_distances)
		#robot_angles = robot.get_current_orientation()

		#theta = robot_angles[2]

		#exit()
	

	plt.plot(-1*np.array(all_x), -1*np.array(all_y), 'o')
	plt.show()

	#vectors = np.array(list(zip(all_x, all_y)))
	#filterAngles(0.9, vectors)
	
	

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

def filterAngles(threshold, vectors):

	#print(len(vectors))

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
	return vectors[:,0].tolist(), vectors[:,1].tolist()

	





if __name__ == '__main__':
	main()