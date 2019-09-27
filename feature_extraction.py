import cv2
import math
import numpy as np
import random


def ransac_inverted_plane(points, canvas):
    good = 0
    while good <= 30:
        #print("Iter number #", good)
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

        m2 = (point2[0] - point1[0])/(point2[1] - point1[1])
        b2 = point2[0] + point2[1]*((point1[0] - point2[0])/(point2[1] - point1[1]))

        denom2 = np.sqrt((m2**2) + 1)

        distance2 = []
        for _, element in enumerate(points):
            distance2.append(np.abs((m2 * element[1]) + element[0] + b2)/denom2)
        norm_dist2 = distance2/np.max(distance2)

        hit_rate2 = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist2):
            if elem < 0.1:
                hit_rate2 += 1
                hit_idx.append(idx)
        #print(hit_rate2)
        
        if hit_rate2 > 10000:
            #print("Vai pintar")

            points = points[points[:, 0].argsort()]
            points = points[points[:, 1].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            to_be_deleted = []
            to_be_deleted_iter = []
            for idx in hit_idx:
                #print("@@@@@@@@2", points[idx][0]**2)
                dist_between_points = math.sqrt(((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                #print(dist_between_points)
                if dist_between_points > 0.1:
                    #print("n points", n_points)
                    if n_points > 1000:
                        #print("Vai pintar meeeesmo", n_points)
                        cv2.line(canvas, (int((1024/14)*(p_0[0] + 6)), (1024 - int((1024/14)*(p_0[1] + 8)))), (int((1024/14)*(last_x + 6)), (1024 - int((1024/14)*(last_y + 8)))), (255, 0, 0), 3)
                        cv2.imwrite("features.png", canvas)
                        to_be_deleted += to_be_deleted_iter
                    n_points = 0
                    p_0[0] = points[idx][0]
                    p_0[1] = points[idx][1]
                    to_be_deleted_iter = []
                else:
                    n_points += 1
                    to_be_deleted_iter.append(idx)
                last_x = points[idx][0]
                last_y = points[idx][1]

        
            points = np.delete(points, to_be_deleted, axis=0)
        good += 1

def ransac_regular_plane(points, canvas):

    lines = []
    good = 0
    while good <= 20:
        print("Iter number #", good)
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

        m1 = (point2[1] - point1[1])/(point2[0] - point1[0])
        b1 = point2[1] + point2[0]*((point1[1] - point2[1])/(point2[0] - point1[0]))

        m2 = (point2[0] - point1[0])/(point2[1] - point1[1])
        b2 = point2[0] + point2[1]*((point1[0] - point2[0])/(point2[1] - point1[1]))

        denom1 = np.sqrt((m1**2) + 1)
        denom2 = np.sqrt((m2**2) + 1)

        distance1 = []
        for _, element in enumerate(points):
            distance1.append(np.abs((m1 * element[0]) + element[1] + b1)/denom1)
        norm_dist1 = distance1/np.max(distance1)

        distance2 = []
        for _, element in enumerate(points):
            distance2.append(np.abs((m2 * element[1]) + element[0] + b2)/denom1)
        norm_dist2 = distance2/np.max(distance2)

        hit_rate1 = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist1):
            if elem < 0.1:
                hit_rate1 += 1
                hit_idx.append(idx)
        #print(hit_rate1)
        
        if hit_rate1 > 20000:
            #print("Vai pintar")

            points = points[points[:, 1].argsort()]
            points = points[points[:, 0].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            to_be_deleted = []
            to_be_deleted_iter = []
            for idx in hit_idx:
                #print("@@@@@@@@2", points[idx][0]**2)
                dist_between_points = math.sqrt(((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                #print(dist_between_points)
                if dist_between_points > 0.1:
                    #print("n points", n_points)
                    if n_points > 1000:
                        #print("Vai pintar meeeesmo", n_points)
                        #cv2.line(canvas, (int((1024/14)*(p_0[0] + 6)), (1024 - int((1024/14)*(p_0[1] + 8)))), (int((1024/14)*(last_x + 6)), (1024 - int((1024/14)*(last_y + 8)))), (255, 0, 0), 3)
                        #cv2.imwrite("features.png", canvas)
                        to_be_deleted += to_be_deleted_iter
                        lines.append([p_0[0], p_0[1], last_x, last_y])
                    n_points = 0
                    p_0[0] = points[idx][0]
                    p_0[1] = points[idx][1]
                    to_be_deleted_iter = []
                else:
                    n_points += 1
                    to_be_deleted_iter.append(idx)
                last_x = points[idx][0]
                last_y = points[idx][1]
            points = np.delete(points, to_be_deleted, axis=0)

        '''hit_rate2 = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist2):
            if elem < 0.1:
                hit_rate2 += 1
                hit_idx.append(idx)
        print(hit_rate2)
        
        if hit_rate2 > 20000:
            #print("Vai pintar")

            points = points[points[:, 0].argsort()]
            points = points[points[:, 1].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            to_be_deleted = []
            to_be_deleted_iter = []
            for idx in hit_idx:
                #print("@@@@@@@@2", points[idx][0]**2)
                dist_between_points = math.sqrt(((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                #print(dist_between_points)
                if dist_between_points > 0.1:
                    #print("n points", n_points)
                    if n_points > 1000:
                        #print("Vai pintar meeeesmo", n_points)
                        cv2.line(canvas, (int((1024/14)*(p_0[0] + 6)), (1024 - int((1024/14)*(p_0[1] + 8)))), (int((1024/14)*(last_x + 6)), (1024 - int((1024/14)*(last_y + 8)))), (255, 0, 0), 3)
                        cv2.imwrite("features.png", canvas)
                        to_be_deleted += to_be_deleted_iter
                    n_points = 0
                    p_0[0] = points[idx][0]
                    p_0[1] = points[idx][1]
                    to_be_deleted_iter = []
                else:
                    n_points += 1
                    to_be_deleted_iter.append(idx)
                last_x = points[idx][0]
                last_y = points[idx][1]

        
            points = np.delete(points, to_be_deleted, axis=0)'''
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


        good += 1

    return np.asarray(lines, dtype=np.float32)




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


    sorted_points = points[points[:, 0].argsort()]
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]

    #print("Msix x:", np.min(data[0]), "Max x:", np.max(data[0]))
    plot_arr = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    result = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    #print(points[0][0], points[0][1])
    for point in sorted_points:
        x = int((1024/14)*(point[0] + 6))
        y = 1024 - int((1024/14)*(point[1] + 8))
        #print(x, y)
        plot_arr[y][x][0] = 0
        plot_arr[y][x][1] = 0
        plot_arr[y][x][2] = 0
        #print(x, y)

    #points = points.reshape((-1, 1, 2))

    regular_lines = ransac_regular_plane(sorted_points, plot_arr)
    regular_lines = regular_lines[regular_lines[:, 0].argsort()]
    regular_lines = regular_lines[regular_lines[:, 1].argsort(kind='mergesort')]
    print("regular lines", regular_lines)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    last_line = []
    start_idx = 0
    last_incl = 0
    if regular_lines.any():
        last_line = regular_lines[0]
        last_incl = (last_line[3] - last_line[1])/(last_line[2] - last_line[0])
        incl = (last_line[3] - last_line[1])/(last_line[2] - last_line[0])

    for idx, line in enumerate(regular_lines):
        print("line:", line)
        if idx != start_idx:
            #print("09")
            incl = (line[3] - line[1])/(line[2] - line[0])
            diff_incl = abs(last_incl - incl)
            dist_y = abs(line[3] - last_line[1])
            dist_x = abs(line[0] - last_line[2])#math.sqrt((line[1] - last_line[3])**2 + (line[0] - last_line[2])**2)
            #print("incl", incl, "dist", dist)
            if diff_incl > 0.1 or (dist_y > 0.3 and dist_x > 0.2):
                #print("w0w", idx)
                cv2.line(plot_arr, (int((1024/14)*(regular_lines[start_idx][0] + 6)), (1024 - int((1024/14)*(regular_lines[start_idx][1] + 8)))), (int((1024/14)*(last_line[2] + 6)), (1024 - int((1024/14)*(last_line[3] + 8)))), (255, 0, 0), 3)
                start_idx = idx
                #cv2.putText(plot_arr, str(idx), (int((1024/14)*(regular_lines[start_idx][0] + 6)), (1024 - int((1024/14)*(regular_lines[start_idx][1] + 8)))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        last_line = line
        last_incl = incl
        cv2.line(plot_arr, (int((1024/14)*(regular_lines[start_idx][0] + 6)), (1024 - int((1024/14)*(regular_lines[start_idx][1] + 8)))), (int((1024/14)*(last_line[2] + 6)), (1024 - int((1024/14)*(last_line[3] + 8)))), (255, 0, 0), 3)

    cv2.imwrite("features.png", plot_arr)


    sorted_points = points[points[:, 0].argsort()]
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]
    #ransac_inverted_plane(sorted_points, plot_arr)


    #print("intersection_points")

    # cv2.imshow("Window", plot_arr)
    # cv2.waitKey(0)
    # cv2.imwrite("features.png", plot_arr)

if __name__ == '__main__':
    main()
