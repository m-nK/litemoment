import numpy as np
import streamlit as st
import imutils
import argparse
import cv2
import sys
from ssocr import detect_digit
def generate_code():
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(4):
        code = np.zeros((1000, 1000, 1), dtype="uint8")
        cv2.aruco.generateImageMarker(arucoDict, i, 1000, code, 1)
        cv2.imwrite("aruco_codes/" + str(i) + ".jpg", code)
    return

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    # rect = order_points(pts)
    (tl, tr, br, bl) = pts
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def detect_code(image):
    print(image)
    if image.ndim != 3:
        st.write("wrong image format")
    if image.shape[2] == 3:
        r,g,b = cv2.split(image)
        image = cv2.merge((b,g,r))
    elif image.shape[2] == 4:
        r,g,b,a = cv2.split(image)
        image = cv2.merge((b,g,r))
    else:
        st.write("wrong image format")

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # cv2.aruco.drawDetectedMarkers(image, corners, ids)
    # cv2.imshow("orig", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if len(corners) > 0:
        st.write("code detected")
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            if markerID == 0:
                corners = markerCorner.reshape((4, 2))
                # print(corners)
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                # new top right/ new top left
                # topRight_new = (topRight[0] + (topRight[0] - topLeft[0]) * 7, topRight[1] + (topRight[1] - topLeft[1]) * 7)
                # bottomLeft_new = (bottomLeft[0] + (bottomLeft[0] - topLeft[0]) * 4, bottomLeft[1] + (bottomLeft[1] - topLeft[1]) * 4)
                # bottomRight_new = (topRight[0] + (topRight[0] - topLeft[0]) * 7, bottomLeft[1] + (bottomLeft[1] - topLeft[1]) * 4)

                # corners_new = np.array([list(topLeft), list(topRight_new), list(bottomRight_new), list(bottomLeft_new)])
                corners[1][0] = topLeft[0] + (topRight[0] - topLeft[0]) * 8
                corners[1][1] = topLeft[1] + (topRight[1] - topLeft[1]) * 8
                # corners[2][0] = topLeft[0] + (bottomRight[0] - topLeft[0]) * 8
                # corners[2][1] = topLeft[1] + (bottomRight[1] - topLeft[1]) * 8
                corners[2][0] = topLeft[0] + (topRight[0] - topLeft[0]) * 8 + (bottomLeft[0] - topLeft[0]) * 4
                corners[2][1] = topLeft[1] + (bottomLeft[1] - topLeft[1]) * 4 + (topRight[1] - topLeft[1]) * 8
                corners[3][0] = topLeft[0] + (bottomLeft[0] - topLeft[0]) * 4
                corners[3][1] = topLeft[1] + (bottomLeft[1] - topLeft[1]) * 4
                print(corners)



                # print(corners_new)

                





                # new = image[topLeft[1] : bottomLeft[1] - (topLeft[1] - bottomLeft[1]) * 4, topLeft[0] : topRight[0] + (topRight[0] - topLeft[0]) * 7, :]
            # print(topLeft[1] , bottomLeft[1] - (topLeft[1] - bottomLeft[1]) * 4, topLeft[0] , topRight[0] + (topRight[0] - topLeft[0]) * 7)
                new = four_point_transform(image,corners)
                st.image(new)
                return new
        st.write("no valid code detected")
    else:
        st.write("no code detected")
    return image
def read(img):
    newimg = detect_code(img)
    return detect_digit(newimg, True)
# image = cv2.imread("testa.jpeg")
# new = detect_code(image)
# print(detect_digit(new, True))

