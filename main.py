import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import copy
import pywt
import pywt.data


def crosshairs(img, point, colour=(0, 0, 255)):
    cv2.line(img, (point[0] - 2, point[1] - 2), (point[0] + 2, point[1] + 2), colour)
    cv2.line(img, (point[0] + 2, point[1] - 2), (point[0] - 2, point[1] + 2), colour)
    return img


def main():
    # Load image
    original = pywt.data.camera()
    images = ["Image001.jpg", "kaoe9m4shrt11.jpg", "6f1imtvb8pt11.jpg", "d1ibesz7but11.jpg"]
    for file in images:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # Wavelet transform of image, and plot approximation and details
        titles = ["Original",'Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']

        coeffs2 = pywt.dwt2(img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([img, LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 5, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()


def maaaain():
    speedup = 1

    image1RGB = cv2.imread("Image001.jpg")

    image2RGB = cv2.imread("Image002.jpg")
    blocksX = 8
    blocksY = 8
    image1 = cv2.cvtColor(image1RGB, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2RGB, cv2.COLOR_BGR2GRAY)
    # movedBrute = copy.deepcopy(image2)
    # motionsExtimated = BruteSearch(blocksX, blocksY, image1, image2, image2RGB, movedBrute, speedup)
    # print motionsExtimated
    # cv2.imshow('BrutalMovement', movedBrute)
    # cv2.waitKey(-1)
    #
    # movedDianond = copy.deepcopy(image2)
    #
    # searchRadius = 5
    # motionsEstimated = DiamondSearch(blocksX, blocksY, image1, image2, image2RGB, movedDianond, searchRadius)
    # print motionsEstimated
    # cv2.imshow('diamond', movedDianond)
    # cv2.waitKey(-1)
    #
    # cap = cv2.VideoCapture("Marvel Studios Avengers Infinity War Official Trailer.avi")
    # working = True
    # plt.ion()
    # figure = plt.figure()
    # ret, _ = cap.read()
    # previousFrameRGB = cv2.resize(_, (0, 0), fx=.5, fy=.5)
    # prevFrame = cv2.cvtColor(previousFrameRGB, cv2.COLOR_BGR2GRAY)
    # stds = [0]
    # while working:
    #     ret, _ = cap.read()
    #     frame = cv2.resize(_, (0, 0), fx=.5, fy=.5)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     moved = copy.deepcopy(gray)
    #
    #     movement = DiamondSearch(blocksX, blocksY, prevFrame, gray, previousFrameRGB, moved, searchRadius)
    #
    #     std = np.matrix([[math.sqrt((item[0] - item[2]) ** 2 + (item[1] - item[3]) ** 2) for item in line] for line in
    #                      movement]).std()
    #     print std
    #     stds.append(std)
    #     plt.clf()
    #     plt.plot(stds)
    #     figure.canvas.draw()
    #     plt.pause(1e-7)
    #     if len(stds)>80:
    #         stds.pop(0)
    #     cv2.imshow('frame', moved)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     previousFrameRGB = frame
    #     prevFrame = gray
    #
    # cap.release()

    # sift = cv2.KAZE_create()
    # mask = np.zeros(image1.shape, np.uint8)
    # cv2.circle(mask, (mask.shape[1]/2,mask.shape[0]/2),mask.shape[0]/3,(255,255,255),thickness=-1)
    # cv2.imshow("mask",mask)
    for sift in [cv2.KAZE_create(), cv2.AKAZE_create(), cv2.ORB_create(), cv2.BRISK_create()]:
        print sift
        kps, descs = sift.detectAndCompute(image1, None)
        kps2, descs2 = sift.detectAndCompute(image2, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matchezz = bf.match(descs, descs2)
        matches = sorted(matchezz, key=lambda x: x.distance)

        wKeypoints = cv2.drawMatches(image1, kps, image2, kps2, matches[:600], None, flags=2)
        wKeypoints1 = cv2.drawKeypoints(image1RGB, kps, np.array([]), (255, 0, 255))
        wKeypoints2 = cv2.drawKeypoints(image2RGB, kps, np.array([]), (255, 0, 255))
        allKeypionts = np.zeros((wKeypoints1.shape[0], wKeypoints1.shape[1] * 2, 3), np.uint8)
        allKeypionts[:, :wKeypoints1.shape[1], :] = wKeypoints1
        allKeypionts[:, wKeypoints1.shape[1]:, :] = wKeypoints2
        cv2.imshow("keypoints " + str(sift), allKeypionts)
        cv2.imshow("matches " + str(sift), wKeypoints)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def DiamondSearch(blocksX, blocksY, image1, image2, image2RGB, movedDianond, searchRadius):
    motionsEstimated = np.zeros((blocksX, blocksY, 4))
    lDSP = [(-searchRadius, 0),
            (-searchRadius / 2, -searchRadius / 2),
            (0, -searchRadius),
            (searchRadius / 2, -searchRadius / 2),
            (searchRadius, 0),
            (searchRadius / 2, searchRadius / 2),
            (0, searchRadius),
            (-searchRadius / 2, searchRadius / 2),
            (0, 0)]
    sDSP = [(0, searchRadius / 2),
            (-searchRadius / 2, 0),
            (0, -searchRadius / 2),
            (searchRadius / 2, 0),
            (0, 0)]
    for i in xrange(blocksX):
        for j in xrange(blocksY):
            currentB = image1[image1.shape[0] * i / blocksX: image1.shape[0] * (i + 1) / blocksX,
                       image1.shape[1] * j / blocksY: image1.shape[1] * (j + 1) / blocksY]
            minLoc = [image1.shape[0] * i / blocksX, image1.shape[1] * j / blocksY]
            minimalVisualDifference = 256 * currentB.shape[0] * currentB.shape[1]
            isStillAtLarge = True
            while isStillAtLarge:
                tempLoc = copy.deepcopy(minLoc)
                for point in lDSP:
                    ii = minLoc[0] + point[0]
                    jj = minLoc[1] + point[1]
                    if ii < 0 or ii >= image2.shape[0] - currentB.shape[0] \
                            or jj < 0 or jj >= image2.shape[1] - currentB.shape[1]:
                        continue
                    underlyingBlock = image2[ii:ii + currentB.shape[0], jj:jj + currentB.shape[1]]
                    differenceBlock = np.abs(np.subtract(underlyingBlock.astype(int), currentB.astype(int)))
                    visualDifference = differenceBlock.sum()
                    if minimalVisualDifference > visualDifference:
                        minimalVisualDifference = visualDifference
                        tempLoc = [ii + currentB.shape[0] / 2, jj + currentB.shape[1] / 2]
                    showThingsDiamond(currentB, ii, image2RGB, jj, lDSP, minLoc)
                if tempLoc == minLoc:
                    isStillAtLarge = False
                minLoc = tempLoc
            tempLoc = copy.deepcopy(minLoc)
            for point in sDSP:
                ii = minLoc[0] + point[0]
                jj = minLoc[1] + point[1]
                if ii < 0 or ii >= image2.shape[0] - currentB.shape[0] \
                        or jj < 0 or jj >= image2.shape[1] - currentB.shape[1]:
                    continue
                underlyingBlock = image2[ii:ii + currentB.shape[0], jj:jj + currentB.shape[1]]
                differenceBlock = np.abs(np.subtract(underlyingBlock.astype(int), currentB.astype(int)))
                visualDifference = differenceBlock.sum()
                if minimalVisualDifference > visualDifference:
                    minimalVisualDifference = visualDifference
                    tempLoc = [ii + currentB.shape[0] / 2, jj + currentB.shape[1] / 2]
                showThingsDiamond(currentB, ii, image2RGB, jj, sDSP, minLoc)
            minLoc = tempLoc
            motionsEstimated[i, j, :] = [image1.shape[0] * i / blocksX, image1.shape[1] * j / blocksY, minLoc[0],
                                         minLoc[1]]
            movedDianond[minLoc[0] - currentB.shape[0] / 2:minLoc[0] + currentB.shape[0] / 2 + currentB.shape[0] % 2,
            minLoc[1] - currentB.shape[1] / 2:minLoc[1] + currentB.shape[1] / 2 + currentB.shape[1] % 2] = currentB
            cv2.line(movedDianond, (image1.shape[0] * i / blocksX + currentB.shape[0] / 2,
                                    image1.shape[1] * j / blocksY + currentB.shape[1] / 2),
                     (minLoc[0], minLoc[1]), (0, 0, 0))
    return motionsEstimated


def BruteSearch(blocksX, blocksY, image1, image2, image2RGB, movedBrute, speedup):
    motionsEstimated = np.zeros((blocksX, blocksY, 4))
    for i in xrange(blocksX):
        for j in xrange(blocksY):
            currentB = image1[image1.shape[0] * i / blocksX: image1.shape[0] * (i + 1) / blocksX,
                       image1.shape[1] * j / blocksY: image1.shape[1] * (j + 1) / blocksY]
            minLoc = [image1.shape[0] * i / blocksX, image1.shape[1] * j / blocksY]
            minimalVisualDifference = 256 * currentB.shape[0] * currentB.shape[1]
            for ii in xrange(0, image2.shape[0] - currentB.shape[0], speedup):
                for jj in xrange(0, image2.shape[1] - currentB.shape[1], speedup):
                    underlyingBlock = image2[ii:ii + currentB.shape[0], jj:jj + currentB.shape[1]]
                    differenceBlock = np.abs(np.subtract(underlyingBlock.astype(int), currentB.astype(int)))
                    visualDifference = differenceBlock.sum()
                    if minimalVisualDifference > visualDifference:
                        minimalVisualDifference = visualDifference
                        minLoc = [ii + currentB.shape[0] / 2, jj + currentB.shape[1] / 2]
                    # temp = copy.deepcopy(image2RGB)
                    # temp[ii:ii + currentB.shape[0], jj:jj + currentB.shape[1]] = cv2.cvtColor(currentB,
                    #                                                                           cv2.COLOR_GRAY2BGR)
                    # cv2.imshow("temp", temp)
                    # cv2.waitKey(1)
            motionsEstimated[i, j, :] = [image1.shape[0] * i / blocksX, image1.shape[1] * j / blocksY, minLoc[0],
                                         minLoc[1]]
            movedBrute[minLoc[0] - currentB.shape[0] / 2:minLoc[0] + currentB.shape[0] / 2 + currentB.shape[0] % 2,
            minLoc[1] - currentB.shape[1] / 2:minLoc[1] + currentB.shape[1] / 2 + currentB.shape[1] % 2] = currentB
            cv2.line(movedBrute, (image1.shape[0] * i / blocksX + currentB.shape[0] / 2,
                                  image1.shape[1] * j / blocksY + currentB.shape[1] / 2),
                     (minLoc[0], minLoc[1]), (0, 0, 0))
    return motionsEstimated


def showThingsDiamond(currentB, ii, image2RGB, jj, lDSP, minLoc):
    pass
    # temp = copy.deepcopy(image2RGB)
    # temp[ii:ii + currentB.shape[0], jj:jj + currentB.shape[1]] = cv2.cvtColor(currentB, cv2.COLOR_GRAY2BGR)
    # for derp in lDSP:
    #     iii = derp[0] + minLoc[0] + currentB.shape[0] / 2
    #     jjj = derp[1] + minLoc[1] + currentB.shape[1] / 2
    #     temp = crosshairs(temp, (jjj, iii))
    # temp = crosshairs(temp, (jj + currentB.shape[1] / 2, ii + currentB.shape[0] / 2), (0, 255, 0))
    # # temp = cv2.resize(temp, (0, 0), fx=2., fy=2.)
    # cv2.imshow("temp", temp)
    # cv2.waitKey(1)


main()
