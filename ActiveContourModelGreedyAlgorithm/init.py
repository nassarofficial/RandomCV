import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    blur = cv2.GaussianBlur(img, (int(math.ceil(3 * sigma)), int(math.ceil(3 * sigma))), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    # cv2.imshow("blurred", np.sqrt(np.add(sobelx**2, sobely**2)))
    # cv2.waitKey()
    return np.sqrt(np.add(sobelx ** 2, sobely ** 2))


def getAvgDist(points, n):
    tot = 0.
    for i in xrange(n):
        tot += ((((points[i + 1:] - points[i]) ** 2).sum(1)) ** .5).sum()

    avg = tot / ((points.shape[0] - 1) * (points.shape[0]) / 2.)
    return avg


def getModulo(i, n):
    modI = np.remainder(i, n)

    if modI == 0:
        modI = n

    modIminus = modI - 1
    modIplus = modI + 1

    if modIminus == 0:
        modIminus = n

    if modIplus > n:
        modIplus = 1
    return modI, modIminus, modIplus


def GreedyAlgorithm(points, img, alpha, beta, gamma, s, sigma, maxIt):
    counter = 0
    cThreshold = 0.3  # Set the curvature threshold
    imgEnrgT = 120  # Set the image energy threshold
    cnt = 0  # Define counter for the number of iterations

    # Initialize the alpha, beta and gamma values for each snake point
    # Adding Columns
    lengthofrows = points.shape[0]
    z = np.zeros((lengthofrows, 3))
    points = np.concatenate((points, z), axis=1)

    points[0] = np.array(points[0])
    points[1] = np.array(points[1])

    points[:, 2] = alpha
    points[:, 3] = beta
    points[:, 4] = gamma
    # Round indices of snake points


    n = lengthofrows  # number of points in snake

    enrgImg = ImgEnrg(img, sigma)

    avgDist = getAvgDist(points, n)  # average distance between points

    a = s ** 2
    dist = np.floor(s / 2)

    tmp = np.tile(((np.arange(1, 6)) - s + dist), (s, 1))
    sz = tmp.shape[0] * tmp.shape[1]
    x1 = np.reshape(tmp, (a, 1))
    x2 = np.reshape(tmp, (a, 1), order='F')

    offsets = np.hstack((x2, x1))

    Econt = []
    Ecurv = []
    Eimg = []
    c = []
    cellArray = np.array([])

    I = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    J = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    flag = True

    while flag == True:
        pointsMoved = 0
        p = np.random.permutation(n)
        #
        #     # Iterate through all snake points randomly
        for k in range(p.shape[0]):
            for i in range(p[k]):
                Econt = []
                Ecurv = []
                Eimg = []

                modI, modIminus, modIplus = getModulo(i, n)

                y0 = np.arange(points[modI - 1, 0] - dist, points[modI - 1, 0] + dist)

                y0 = np.append(y0, points[modI - 1, 0] + dist)
                y1 = np.arange(points[modI - 1, 1] - dist, points[modI - 1, 1] + dist)
                y1 = np.append(y1, points[modI - 1, 1] + dist)

                neighborhood = np.zeros((5, 5))
                for l in range(y0.shape[0]):
                    for m in range(y1.shape[0]):
                        neighborhood[l][m] = enrgImg[y0[l], y1[m]]

                enrgMin = np.amin(neighborhood)
                enrgMax = np.amax(neighborhood)

                if (enrgMax - enrgMin) < 5:
                    enrgMin = enrgMax - 5

                normNeigh = (enrgMin - neighborhood) / (enrgMax - enrgMin)
                pos = np.array([0, 0])
                # print offsets
                for j in range(a):
                    print j
                    pos = points[i, [0, 2]] + offsets[j]
                    Econt.append(abs(avgDist - np.linalg.norm(np.subtract(pos, points[modIminus, [0, 2]]))))
                    Ecurv.append(np.linalg.norm(
                        np.subtract(points[modIminus, [0, 2]], 2 * pos + points[modIminus, [0, 2]])) ** 2)
                    Eimg.append(normNeigh[I[j] - 1, J[j] - 1])

                Econt = Econt / max(Econt)
                Ecurv = Ecurv / max(Ecurv)

                Esnake = points[i, 3] * Econt + points[i, 4] * Ecurv + points[i, 5] * Eimg
                dummy, indexMin = min(Esnake)

                if math.ceil(a / 2) != indexMin:
                    points[modI, [0, 2]] = np.add(points[modI, [0, 2]], offsets[indexMin])
                    pointsMoved = pointsMoved + 1

                points[6, modI] = neighborhood[I(indexMin) - 1, J(indexMin) - 1]

            for j in range(n):
                modI, modIminus, modIplus = getModulo(i, n)
                if (c[modI] > c[modIminus] and c[modI] > c[modIplus] and c[modI] > cThreshold and points[
                    6, modI] > imgEnrgT and points[4, modI] != 0):
                    points[4, modI] = 0
                    print 'Relaxed beta for point nr. ' +  i

            counter += 1
            cellArray[counter] = points

            if (counter == maxIt or pointsMoved < 3):
                flag = False
                cellArray = cellArray[1:counter]

            avgDist = getAvgDist(points, n)

            return points

    def displaypoints(img,points):
        img = plt.imread(img)
        implot = plt.imshow(img)
        plt.scatter(points)
        plt.show()

    if __name__ == "__main__":
        img = cv2.imread('shark1.png', 0)
        pointselection = "wd"
        if pointselection == "user":
            points = contour_selection(img, "Selection of points")
        else:
            points = np.array(
                [219, 218, 215, 211, 207, 201, 195, 188, 180, 172, 163, 154, 146, 137, 128, 120, 112, 105, 99, 93, 89,
                 119,
                 127, 136, 144, 151, 158, 164, 169, 173, 177, 179, 180, 180, 179, 177, 173, 169, 164, 158, 151, 144, 85,
                 82,
                 81, 80, 81, 82, 85, 89, 93, 99, 105, 112, 120, 128, 137, 146, 154, 163, 172, 180, 188, 136, 127, 119,
                 110,
                 101, 93, 84, 76, 69, 62, 56, 51, 47, 43, 41, 40, 40, 41, 43, 47, 51, 195, 201, 207, 211, 215, 218, 219,
                 220, 56, 62, 69, 76, 84, 93, 101, 110])
            points = np.reshape(points, (-1, 2))
        # for i in range(50):
        #     points[i][j] = c[0] + math.floor(r * math.cos((i) * 2 * math.pi / i) + 0.5)
        #     points[i][j] = c[1] + math.floor(r * math.sin((i) * 2 * math.pi / i) + 0.5)

        alpha = 0.05  # controls continuity
        beta = 1  # controls curvature
        gamma = 1.2  # controls strength of image energy
        s = 5  # controls the size of the neighborhood
        sigma = 15  # controls amount of Gaussian blurring
        maxIt = 200  # Defines the maximum number of snake iterations
        rs = "on"  # Controls whether to have resmapling on or off

        C = GreedyAlgorithm(points, img, alpha, beta, gamma, s, sigma, maxIt)
        #print C
        displaypoints(img,C)

