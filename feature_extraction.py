import cv2
import math
import numpy as np
import random

def ransac(points, canvas):
    good = 0
    while good <= 10:
        valid_guess = False
        guess1 = 0
        guess2 = 0

        while not valid_guess:
            guess1 = random.randint(0, points.shape[0] - 1)
            guess2 = random.randint(0, points.shape[0] - 1)
            
            if guess1 != guess2:
                valid_guess = True

        point1 = points[guess1]
        point2 = points[guess2]

        m = (point2[1] - point1[1])/(point2[0] - point1[0])
        b = point2[1] + point2[0]*((point1[1] - point2[1])/(point2[0] - point1[0]))

        denom = np.sqrt((m**2) + 1)

        distance = []
        for _, element in enumerate(points):
            distance.append(np.abs((m * element[0]) + element[1] + b)/denom)
        norm_dist = distance/np.max(distance)

        hit_rate = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist):
            if elem < 0.1:
                hit_rate += 1
                hit_idx.append(idx)
        print(hit_rate)
        
        if hit_rate > 20000:
            print("Vai pintar")
            min_x = 8
            max_x = -8
            min_y = 8
            max_y = -8

            points = points[points[:, 1].argsort()]
            points = points[points[:, 0].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            for idx in hit_idx:
                #print("@@@@@@@@2", points[idx][0]**2)
                dist_between_points = math.sqrt(((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                #print(dist_between_points)
                if dist_between_points > 0.1:
                    #print("n points", n_points)
                    if n_points > 1000:
                        print("Vai pintar meeeesmo", n_points)
                        cv2.line(canvas, (int((1024/14)*(p_0[0] + 6)), (1024 - int((1024/14)*(p_0[1] + 8)))), (int((1024/14)*(last_x + 6)), (1024 - int((1024/14)*(last_y + 8)))), (255, 0, 0), 3)
                    n_points = 0
                    p_0[0] = points[idx][0]
                    p_0[1] = points[idx][1] 
                else:
                    n_points += 1
                last_x = points[idx][0]
                last_y = points[idx][1]
                '''if points[idx][0] < min_x:
                    min_x = points[idx][0]
                    min_y = points[idx][1]
                if points[idx][0] > max_x:
                    max_x = points[idx][0]
                    max_y = points[idx][1]
                #if points[idx][1] < min_y:
                #if points[idx][1] > max_y:
            x_0 = int((1024/14)*(min_x + 6))
            y_0 = 1024 - int((1024/14)*(min_y + 8))
            x_f = int((1024/14)*(max_x + 6))
            y_f = 1024 - int((1024/14)*(max_y + 8))'''

            # cv2.line(canvas, (x_0, y_0), (x_f, y_f), (255, 0, 0), 3)

            cv2.imwrite("features.png", canvas)

        good += 1






    '''

    #print("points", points.shape)
    print("Left:", left_idx, "Right:", right_idx)
    if right_idx <= left_idx:
        #print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOW")
        return

    distance = []
    #norm_dist = np.zeros_like(distance)

    analyzed_x_coords = points[left_idx:right_idx, 0]
    #print("Analized coords:", analyzed_x_coords)
    analyzed_y_coords = points[left_idx:right_idx, 1]

    #lines = cv2.HoughLinesP(plot_arr, 0.5, (np.pi/180.0), 100000)
    A = np.vstack([analyzed_x_coords, np.ones(len(analyzed_x_coords))]).T
    m, c = np.linalg.lstsq(A, analyzed_y_coords, rcond=None)[0]
    #print(m, c)
    denom = np.sqrt((m**2) + 1)

    for idx, element in enumerate(analyzed_x_coords):
        distance.append(np.abs((m * element) + analyzed_y_coords[idx] + c)/denom)
    norm_dist = distance/np.max(distance)

    pivot_idx = 0
    pivot_val = 0
    for idx, element in enumerate(norm_dist):
        #print(element)
        if element > pivot_val and element != 1:
            pivot_val = element
            pivot_idx = idx

    pivot_idx += left_idx
    #print("Pivot idx and value:", pivot_idx, pivot_val, np.max(norm_dist))

    if pivot_val > 0.2:
        split(points, left_idx, pivot_idx - 1, canvas)
        split(points, pivot_idx + 1, right_idx, canvas)
    else:
        # draw line
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        x_0 = int((1024/14)*(np.min(analyzed_x_coords) + 6))
        y_0 = 1024 - int((1024/14)*((np.min(analyzed_x_coords)*m + c) + 8))
        x_f = int((1024/14)*(np.max(analyzed_x_coords) + 6))
        y_f = 1024 - int((1024/14)*((np.max(analyzed_x_coords)*m + c) + 8))
        cv2.line(canvas, (x_0, y_0), (x_f, y_f), (255, 0, 0), 3)
        cv2.imwrite("features.png", canvas)

    # print("Min y:", (np.min(x_coord*m) + c), "max y:", (np.max(x_coord*m) + c))
    # x_0 = int((1024/14)*(np.min(x_coord) + 6))
    # y_0 = 1024 - int((1024/14)*((np.min(x_coord*m) + c) + 8))
    # x_f = int((1024/14)*(np.max(x_coord) + 6))
    # y_f = 1024 - int((1024/14)*((np.max(x_coord*m) + c) + 8))
    # print("y_0:", y_0, "y_f:", y_f)
    # cv2.line(canvas, (x_0, y_0), (x_f, y_f), (255, 0, 0), 3)'''

def main():
    data = np.load('ExtractedPoints.npy')

    data = data*(-1)
 
    points = np.zeros(shape=(len(data[0]), 2), dtype=np.float32)
    for i, _ in enumerate(points):
        points[i] = [data[0][i], data[1][i]]

    points = points[points[:, 1].argsort()]
    points = points[points[:, 0].argsort(kind='mergesort')]

    print("Msix x:", np.min(data[0]), "Max x:", np.max(data[0]))
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

    #points = points.reshape((-1, 1, 2))

    ransac(points, plot_arr)
    #print("intersection_points")

    # cv2.imshow("Window", plot_arr)
    # cv2.waitKey(0)
    # cv2.imwrite("features.png", plot_arr)

if __name__ == '__main__':
    main()
