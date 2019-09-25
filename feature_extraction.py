import cv2
import numpy as np

def main():
    #print(np.array([2, 1, 2]))
    data = np.load('ExtractedPoints.npy')
    #print(len(data[0]))

    points = np.zeros(shape=(len(data[0]), 2), dtype=np.float32)
    for i, _ in enumerate(points):
        points[i] = [data[0][i], data[1][i]]
        #print(i, points[i])
    #points = [np.array([x, y], dtype=np.float32) for x, y in zip(data[0], data[1])]
    
    print(data[0].min(), data[0].max())
    print(data[1].min(), data[1].max())

    plot_arr = np.full((1024, 1024), 255,dtype=np.uint8)
    result = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    #print(points[0][0], points[0][1])
    for point in points:
        x = int((1024/14)*((-1)*point[0] + 6))
        y = 1024 - int((1024/14)*((-1)*point[1] + 8))
        #print(x, y)
        plot_arr[y][x] = 0
        #print(x, y)

    cv2.imwrite("test_mono.png", plot_arr)
    points = points.reshape((-1, 1, 2))

    intersection_points = []
    #lines = cv2.HoughLinesP(plot_arr, 0.5, (np.pi/180.0), 100000)
    lines = cv2.HoughLinesPointSet(points, 15, 100000, 0, 10, 0.5, 0, (np.pi/2.0), (np.pi/180.0))
    for l in lines:
        for line in l:
            votes, rho, theta = line
            print(votes, "rho:", rho, "theta:", theta)

            if (theta == 0):
                x0 = rho
                x1 = rho
                y0 = -6
                y1 = 8
            else:
                x0 = -8
                x1 = 6
                y0 = (rho - x0*np.cos(theta))/np.sin(theta)
                y1 = (rho - x1*np.cos(theta))/np.sin(theta)
            print(x0, y0, x1, y1)

            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 =a*rho
            # y0 = b*rho
            #x0 = int((1024/14)*(((-1)*a*rho) + 6))#a*rho
            #y0 = 1024 - int((1024/14)*(((-1)*b*rho) + 8))#b*rho
            #cv2.line(result, (x0, y0), (x0 + 100, y0), (0, 0, 255), 3)
            #print(x0, y0)
            intersection_points.append((x0, y0))

    for i_p in intersection_points:
        x = int((1024/16)*(i_p[0] + 6))
        y = 1024 - int((1024/16)*(i_p[1] + 8))
        result[y][x-10:x+10][:] = 0
    #print(intersection_points)

    #cv2.imshow("Window", result)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()
