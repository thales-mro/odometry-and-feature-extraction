import cv2
import math
import numpy as np
import random

RANSAC_EXEC_TIMES = 99
MIN_HITS = 10000
MIN_CONSEC_POINTS = 1000

def ransac_inverted_plane(points):
    """
        Function to calculate potential lines from points in the plane (x and y coordinates),
        considering the plane as the inverted x and y references
        (to be able to map vertical lines in cartesian coordinates).
        Function uses RANSAC to do such task (guessing line from 2 different points,
        and testing if line is good by counting the number of points that fits the calculated
        one, considering specific threshold)
        It returns array of lines, in which each element has the following format:
            [x_0, y_0, x_f, y_f], where x_0, y_0 represent the coordinates of the first point in
            the line, and x_f, y_f represent the coordinates of the last point belonging to the line

        Keyword arguments:
        points -- array of points in the (x, y) format
    """
    lines = []
    loop_idx = 0
    while loop_idx <= RANSAC_EXEC_TIMES:
        print("Iter number #", loop_idx)
        valid_guess = False
        guess1 = 0
        guess2 = 0

        # calculates point guess
        while not valid_guess:
            guess1 = random.randint(0, points.shape[0] - 1)
            guess2 = random.randint(0, points.shape[0] - 1)

            if guess1 != guess2:
                valid_guess = True

        point1 = points[guess1]
        point2 = points[guess2]

        # calculates line parameters from 2 points in the inverted yx plane
        m = (point2[0] - point1[0])/(point2[1] - point1[1])
        b = point2[0] + point2[1]*((point1[0] - point2[0])/(point2[1] - point1[1]))

        denom = np.sqrt((m**2) + 1)

        # calculates distances from every point to line
        distance2 = []
        for _, element in enumerate(points):
            distance2.append(np.abs((m * element[1]) + element[0] + b)/denom)
        # normalizes calculate distances 
        norm_dist2 = distance2/np.max(distance2)

        # calculates the number of points that are close to the line according to threshold
        hit_rate = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist2):
            if elem < 0.1:
                hit_rate += 1
                hit_idx.append(idx)

        # it the number of close points to the line are greater than a threshold, we possibly have a line
        if hit_rate > MIN_HITS:
            points = points[points[:, 0].argsort()]
            points = points[points[:, 1].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            to_be_deleted = []
            to_be_deleted_iter = []
            # test if the points are consecutive (they can form multiple colinear lines)
            for idx in hit_idx:
                dist_between_points = math.sqrt(
                    ((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                if dist_between_points > 0.1:
                    if n_points > MIN_CONSEC_POINTS:
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
            #deletes points within found lines
            points = np.delete(points, to_be_deleted, axis=0)
        loop_idx += 1

    return np.asarray(lines, dtype=np.float32)

def ransac_regular_plane(points):
    """
    Function to calculate potential lines from points in the plane (x and y coordinates),
    considering the conventional plane (cannot map vertical lines).
    Function uses RANSAC to do such task (guessing line from 2 different points,
    and testing if line is good by counting the number of points that fits the calculated
    one, considering specific threshold)
    It returns array of lines, in which each element has the following format:
        [x_0, y_0, x_f, y_f], where x_0, y_0 represent the coordinates of the first point in
        the line, and x_f, y_f represent the coordinates of the last point belonging to the line

    Keyword arguments:
    points -- array of points in the (x, y) format
    """
    lines = []
    loop_idx = 0
    while loop_idx <= RANSAC_EXEC_TIMES:
        print("Iter number #", loop_idx)
        valid_guess = False
        guess1 = 0
        guess2 = 0

        # calculates point guess
        while not valid_guess:
            guess1 = random.randint(0, points.shape[0] - 1)
            guess2 = random.randint(0, points.shape[0] - 1)

            if guess1 != guess2:
                valid_guess = True

        point1 = points[guess1]
        point2 = points[guess2]

        # calculates line parameters from 2 points in the xy plane
        m = (point2[1] - point1[1])/(point2[0] - point1[0])
        b = point2[1] + point2[0]*((point1[1] - point2[1])/(point2[0] - point1[0]))
        denom = np.sqrt((m**2) + 1)

        # calculates distances from every point to line
        distance = []
        for _, element in enumerate(points):
            distance.append(np.abs((m * element[0]) + element[1] + b)/denom)
        # normalizes calculate distances 
        norm_dist = distance/np.max(distance)

        # calculates the number of points that are close to the line according to threshold
        hit_rate = 0
        hit_idx = []
        for idx, elem in enumerate(norm_dist):
            if elem < 0.1:
                hit_rate += 1
                hit_idx.append(idx)

        # it the number of close points to the line are greater than a threshold, we possibly have a line
        if hit_rate > MIN_HITS:

            points = points[points[:, 1].argsort()]
            points = points[points[:, 0].argsort(kind='mergesort')]

            last_x = points[0][0]
            last_y = points[0][1]
            p_0 = [last_x, last_y]
            n_points = 0
            to_be_deleted = []
            to_be_deleted_iter = []
            # test if the points are consecutive (they can form multiple colinear lines)
            for idx in hit_idx:
                dist_between_points = math.sqrt(
                    ((points[idx][0] - last_x)**2) + ((points[idx][1] - last_y)**2))
                if dist_between_points > 0.1:
                    if n_points > MIN_CONSEC_POINTS:
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
            #deletes points within found lines
            points = np.delete(points, to_be_deleted, axis=0)
        loop_idx += 1

    return np.asarray(lines, dtype=np.float32)

def main():
    # loads extracted points
    data = np.load('ExtractedPoints.npy')
    data = data*(-1)

    canvas = np.full((1024, 1024, 3), 255, dtype=np.uint8)

    points = np.zeros(shape=(len(data[0]), 2), dtype=np.float32)
    for i, _ in enumerate(points):
        points[i] = [data[0][i], data[1][i]]

    #sorts points
    sorted_points = points[points[:, 0].argsort()]
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]

    ''' #UNCOMMENT IF YOU DESIRE TO PRINT THE POINTS EXTRACTED FROM ROBOT
    for point in sorted_points:
        x = int((1024/14)*(point[0] + 6))
        y = 1024 - int((1024/14)*(point[1] + 8))
        canvas[y][x][0] = 0
        canvas[y][x][1] = 0
        canvas[y][x][2] = 0
    '''

    # calls ransac function for line calculation in the regular plane, orders obtained lines
    regular_lines = ransac_regular_plane(sorted_points)
    regular_lines = regular_lines[regular_lines[:, 0].argsort()]
    regular_lines = regular_lines[regular_lines[:, 1].argsort(kind='mergesort')]

    last_line = []
    start_idx = 0
    last_incl = 0
    #it tries to merge lines that are colinear and close enough, before painting into canvas
    if regular_lines.any():
        last_line = regular_lines[0]
        last_incl = (last_line[3] - last_line[1])/(last_line[2] - last_line[0])
        incl = (last_line[3] - last_line[1])/(last_line[2] - last_line[0])

    for idx, line in enumerate(regular_lines):
        if idx != start_idx:
            incl = (line[3] - line[1])/(line[2] - line[0])
            diff_incl = abs(last_incl - incl)
            dist_y = abs(line[3] - last_line[1])
            dist_x = abs(line[0] - last_line[2])

            if diff_incl > 0.1 or (dist_y > 0.1 or dist_x > 0.3):
                #if lines are not colinear or close enough, paint the first one in the canvas
                cv2.line(canvas,
                         (int((1024/14)*(regular_lines[start_idx][0] + 6)),
                          (1024 - int((1024/14)*(regular_lines[start_idx][1] + 8)))),
                         (int((1024/14)*(last_line[2] + 6)),
                          (1024 - int((1024/14)*(last_line[3] + 8)))),
                         (255, 0, 0), 3)
                start_idx = idx

        last_line = line
        last_incl = incl
        # paint last line into canvas
        cv2.line(canvas,
                 (int((1024/14)*(regular_lines[start_idx][0] + 6)),
                  (1024 - int((1024/14)*(regular_lines[start_idx][1] + 8)))),
                 (int((1024/14)*(last_line[2] + 6)),
                  (1024 - int((1024/14)*(last_line[3] + 8)))),
                 (255, 0, 0), 3)

    sorted_points = points[points[:, 0].argsort()]
    sorted_points = sorted_points[sorted_points[:, 1].argsort(kind='mergesort')]
    # calls ransac function for line calculation in the inverted plane, orders obtained lines
    inverted_lines = ransac_inverted_plane(sorted_points)
    inverted_lines = inverted_lines[inverted_lines[:, 1].argsort()]
    inverted_lines = inverted_lines[inverted_lines[:, 0].argsort(kind='mergesort')]

    last_line = []
    start_idx = 0
    last_incl = 0
    #it tries to merge lines that are colinear and close enough, before painting into canvas
    if inverted_lines.any():
        last_line = inverted_lines[0]
        last_incl = (last_line[2] - last_line[0])/(last_line[3] - last_line[1])
        incl = (last_line[2] - last_line[0])/(last_line[3] - last_line[1])

    for idx, line in enumerate(inverted_lines):
        if idx != start_idx:
            incl = (line[2] - line[0])/(line[3] - line[1])
            diff_incl = abs(last_incl - incl)
            dist_y = abs(line[3] - last_line[1])
            dist_x = abs(line[0] - last_line[2])
            #if lines are not colinear or close enough, paint the first one in the canvas
            if diff_incl > 0.1 or (dist_y > 0.2 and dist_x > 0.1):
                cv2.line(canvas,
                         (int((1024/14)*(inverted_lines[start_idx][0] + 6)),
                          (1024 - int((1024/14)*(inverted_lines[start_idx][1] + 8)))),
                         (int((1024/14)*(last_line[2] + 6)),
                          (1024 - int((1024/14)*(last_line[3] + 8)))),
                         (255, 0, 0), 3)
                start_idx = idx

        last_line = line
        last_incl = incl
        # paint last line into canvas
        cv2.line(canvas,
                 (int((1024/14)*(inverted_lines[start_idx][0] + 6)),
                  (1024 - int((1024/14)*(inverted_lines[start_idx][1] + 8)))),
                 (int((1024/14)*(last_line[2] + 6)),
                  (1024 - int((1024/14)*(last_line[3] + 8)))),
                 (255, 0, 0), 3)

    cv2.imwrite("all_features.png", canvas)

if __name__ == '__main__':
    main()
