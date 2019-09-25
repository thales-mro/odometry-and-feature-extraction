import cv2
import numpy as np

def split(x_coord, y_coord, left_idx, right_idx, canvas):
    if right_idx <= left_idx:
        return

    distance = np.zeros(shape=(1, len(x_coord)), dtype=np.float32)
    norm_dist = np.zeros_like(distance)

    #lines = cv2.HoughLinesP(plot_arr, 0.5, (np.pi/180.0), 100000)
    A = np.vstack([x_coord, np.ones(len(x_coord))]).T
    m, c = np.linalg.lstsq(A, y_coord, rcond=None)[0]
    print(m, c)
    print("Min y:", (np.min(x_coord*m) + c), "max y:", (np.max(x_coord*m) + c))
    denom = np.sqrt((m**2) + 1)
    x_0 = int((1024/14)*(np.min(x_coord) + 6))
    y_0 = 1024 - int((1024/14)*((np.min(x_coord*m) + c) + 8))
    x_f = int((1024/14)*(np.max(x_coord) + 6))
    y_f = 1024 - int((1024/14)*((np.max(x_coord*m) + c) + 8))
    print("y_0:", y_0, "y_f:", y_f)
    cv2.line(canvas, (x_0, y_0), (x_f, y_f), (255, 0, 0), 3)

    for idx, element in enumerate(x_coord):
        distance[0][idx] = np.abs((m * element) + y_coord[idx] + c)/denom
    norm_dist = [distance/np.max(distance)]
    print(norm_dist)


def main():
    data = np.load('ExtractedPoints.npy')

    data = data*(-1)
 
    points = np.zeros(shape=(len(data[0]), 2), dtype=np.float32)
    for i, _ in enumerate(points):
        points[i] = [data[0][i], data[1][i]]

    print("Mix x:", np.min(data[0]), "Max x:", np.max(data[0]))

    plot_arr = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    result = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    #print(points[0][0], points[0][1])
    for point in points:
        x = int((1024/14)*(point[0] + 6))
        y = 1024 - int((1024/14)*(point[1] + 8))
        #print(x, y)
        plot_arr[y][x][0] = 0
        plot_arr[y][x][1] = 0
        plot_arr[y][x][2] = 0
        #print(x, y)

    points = points.reshape((-1, 1, 2))

    split(data[0], data[1], 0, len(data[0]), plot_arr)
    #print("intersection_points")

    cv2.imshow("Window", plot_arr)
    cv2.waitKey(0)
    cv2.imwrite("features.png", plot_arr)

if __name__ == '__main__':
    main()
