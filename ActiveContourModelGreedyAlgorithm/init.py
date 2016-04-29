import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math

flag_drawing = False


def contour_selection(image, winname=-1, distance=10):
    if winname == -1:
        winname = "Test"

    overlay_image = image.copy()
    if len(overlay_image.shape) > 2:
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)
    point_list = []
    cv2.putText(overlay_image, "Click on image to draw initial snake", (200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0))
    cv2.imshow(winname, overlay_image)

    cv2.setMouseCallback(winname, on_mouse, param=[point_list, image, winname, distance])
    cv2.waitKey()
    cv2.imshow(winname, image)
    cv2.setMouseCallback(winname, lambda a, b, c, d, e: None)
    return point_list


def on_mouse(event, x, y, flag, param):
    global flag_drawing
    contour = param[0]
    overlay_image = cv2.cvtColor(param[1].copy(), cv2.COLOR_GRAY2RGB)
    winname = param[2]
    distance = param[3]

    if event == cv2.EVENT_LBUTTONDOWN:
        flag_drawing = not flag_drawing

    if event == cv2.EVENT_MOUSEMOVE and flag_drawing:
        xy = np.array([x, y])

        if len(contour) < 1:  # the first pixel clicked is always ok
            contour.append(xy)

        elif np.linalg.norm(xy - np.array(contour)[-1]) > distance:
            contour.append(xy)

        for i in range(len(contour)):
            cv2.circle(overlay_image, (contour[i][0], contour[i][1]), 2, 255, 2)
        cv2.polylines(overlay_image, np.array([contour]), 0, (0, 0, 255), 1)
        cv2.imshow(winname, overlay_image)
    return


def ImgEnrg(img, sigma):
    blur = cv2.GaussianBlur(img,(int(math.ceil(3*sigma)), int(math.ceil(3*sigma))), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    #cv2.imshow("blurred", np.sqrt(np.add(sobelx**2, sobely**2)))
    #cv2.waitKey()
    return np.sqrt(np.add(sobelx**2, sobely**2))


def getAvgDist(points, n):
    tot = 0.
    for i in xrange(n):
        tot += ((((points[i + 1:] - points[i]) ** 2).sum(1)) ** .5).sum()

    avg = tot / ((points.shape[0] - 1) * (points.shape[0]) / 2.)
    return avg


def GreedyAlgorithm(points, img, alpha, beta, gamma, s, sigma, maxIt):
    cThreshold = 0.3  # Set the curvature threshold
    imgEnrgT = 120  # Set the image energy threshold
    cnt = 0  # Define counter for the number of iterations

    # Initialize the alpha, beta and gamma values for each snake point
    points = np.asarray(points)
    # Adding Columns
    lengthofrows = points.shape[0]
    z = np.zeros((lengthofrows, 3))
    points = np.concatenate((points, z), axis=1)
    points[:, 2] = alpha
    points[:, 3] = beta
    points[:, 4] = gamma

    # Round indices of snake points
    n = lengthofrows  # number of points in snake

    enrgImg = ImgEnrg(img, sigma)
    avgDist = getAvgDist(points, n)  # average distance between points

    a = s ** 2
    dist = np.floor(s / 2)

    tmp = np.tile(((np.arange(1, s)) - s + dist), (s, 1))

    ##offsets = reshape(tmp,a,1),reshape(tmp.T,a,1))) ################
    #  (np.reshape(tmp,(a,1)), np.reshape(tmp.T,(a,1))), axis=1

    offsets = np.concatenate((np.reshape(tmp,(a,1)), np.reshape(tmp.T,(a,1))), axis=1)

    print offsets
    #Econt = np.zeros(1, a)
    #Ecurv = np.zeros(1, a)
    #Eimg = np.zeros(1, a)
    #c = np.zeros(1, n)

    flag = True

    # while flag == True:
    #    pointsMoved=0
    #    p = randperm(n) ###################


if __name__ == "__main__":
    img = cv2.imread('ct.jpg', 0)

    #points = contour_selection(img, "Selection of points")
    points = [[2,2],[4,4],[6,6],[8,8]]
    alpha = 0.05  # controls continuity
    beta = 1  # controls curvature
    gamma = 1.2  # controls strength of image energy
    s = 5  # controls the size of the neighborhood
    sigma = 15  # controls amount of Gaussian blurring
    maxIt = 200  # Defines the maximum number of snake iterations
    rs = "on"  # Controls whether to have resmapling on or off

    GreedyAlgorithm(points, img, alpha, beta, gamma, s, sigma, maxIt)
