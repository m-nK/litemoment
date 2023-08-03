# -*- coding:utf-8 -*-
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import color
import streamlit as st
import sys
# from trystack import Segments
DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    # (0, 0, 0, 0, 0, 0, 1): '-'
}
# H_W_Ratio = 1.9
H_W_Ratio = 1.6
THRESHOLD = 50
arc_tan_theta = 8.0  # 数码管倾斜角度
# crop_y0 = 215
# crop_y1 = 470
# crop_x0 = 260
# crop_x1 = 890
sys.setrecursionlimit(1000)
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='path to image')
parser.add_argument('-s', '--show_image', action='store_const', const=True, help='whether to show image')
parser.add_argument('-d', '--is_debug', action='store_const', const=True, help='True or False')
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
def load_image(image, show=False):
    # todo: crop image and clear dc and ac signal
    # img = cv2.imread(path)
    # cv2.imshow("orig", lab);
    img = image.copy()
    # st.image(img)
    # cv2.imshow("orig", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)
    # closing operation
    kernel = np.ones((5,5), np.uint8)

    # threshold params
    low = 165
    high = 300
    iters = 3
    # cv2.imshow("l", l)
    # cv2.imshow("a", a)
    # cv2.imshow("b", b)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # make copy
    copy = b.copy()
    # st.image(b) ____________
    # threshold
    thresh = cv2.inRange(copy, low, high)
    # st.image(thresh) ____________
    # dilate
    for a in range(iters):
        thresh = cv2.dilate(thresh, kernel)

    # erode
    for a in range(iters):
        thresh = cv2.erode(thresh, kernel)
    thresh = np.invert(thresh)
    # st.image(thresh) ____________



    # cv2.imshow("thresh", thresh)



    # cv2.imshow("try", lab)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # return False
    # gray_img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    gray_img = thresh
    h, w = gray_img.shape
    # crop_y0 = 0 if h <= crop_y0_init else crop_y0_init
    # crop_y1 = h if h <= crop_y1_init else crop_y1_init
    # crop_x0 = 0 if w <= crop_x0_init else crop_x0_init
    # crop_x1 = w if w <= crop_x1_init else crop_x1_init
    # gray_img = gray_img[crop_y0:crop_y1, crop_x0:crop_x1]
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    # if show:
        # cv2.imshow('gray_img', gray_img)
        # cv2.imshow('blurred_img', blurred)
    return blurred, gray_img


def preprocess(img, threshold, show=False, kernel_size=(5, 5)):

    # 直方图局部均衡化
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)
    # 自适应阈值二值化
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    # 闭运算开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    # if show:
        # cv2.imshow('equlizeHist', img)
        # cv2.imshow('threshold', dst)
    return dst


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, reserved_threshold=20):
    digits_positions = []
    img_array = np.sum(img, axis=0)
    horizon_position = helper_extract(img_array, threshold=reserved_threshold)
    img_array = np.sum(img, axis=1)
    vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4)
    # make vertical_position has only one element
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    if len(digits_positions) <= 0:
        st.write("Failed to find digits's positions")
    # print(digits_positions)
    return digits_positions


def recognize_digits_area_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))
        # 对1的情况单独识别

        if w > h:
            digits.append("-")
            continue
        if w < suppose_W / 2:
            x0 = x0 + w - suppose_W
            w = suppose_W
            roi = input_img[y0:y1, x0:x1]
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        dhc = int(width * 0.8)
        # print('width :', width)
        # print('dhc :', dhc)

        small_delta = int(h / arc_tan_theta) // 4
        # print('small_delta : ', small_delta)
        segments = [
            # # version 1
            # ((w - width, width // 2), (w, (h - dhc) // 2)),
            # ((w - width - small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
            # ((width // 2, h - width), (w - width // 2, h)),
            # ((0, (h + dhc) // 2), (width, h - width // 2)),
            # ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
            # ((small_delta, 0), (w, width)),
            # ((width, (h - dhc) // 2), (w - width, (h + dhc) // 2))
            # # version 2
            ((w - width - small_delta, width // 2), (w, (h - dhc) // 2)),
            ((w - width - 2 * small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
            ((width - small_delta, h - width), (w - width - small_delta, h)),
            ((0, (h + dhc) // 2), (width, h - width // 2)),
            ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
            ((small_delta, 0), (w + small_delta, width)),
            ((width - small_delta, (h - dhc) // 2), (w - width - small_delta, (h + dhc) // 2))
        ]
        # cv2.rectangle(roi, segments[0][0], segments[0][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[1][0], segments[1][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[2][0], segments[2][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[3][0], segments[3][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[4][0], segments[4][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[5][0], segments[5][1], (128, 0, 0), 2)
        # cv2.rectangle(roi, segments[6][0], segments[6][1], (128, 0, 0), 2)
        # cv2.imshow('i', roi)
        # cv2.waitKey()
        # cv2.destroyWindow('i')
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            # plt.imshow(seg_roi)
            # plt.show()
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            # print(total / float(area))
            if total / float(area) > 0.45:
                on[i] = 1
        # st.write(on)
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'
        digits.append(digit)
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(digit), (x0 - 10, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

    return digits


def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))

        # 消除无关符号干扰
        if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue
        if w > h:
            continue
        # st.write(w)
        # st.write(h)
        # st.write(suppose_W)
        # 对1的情况单独识别
        if w < suppose_W / 1.6:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]

        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            # plt.imshow(seg_roi, 'gray')
            # plt.show()
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            # print('prob: ', total / float(area))
            if total / float(area) > 0.10:#0.25 or 0.10
                on[i] = 1
        # print('encode: ', on)
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
        # else:
        #     digit = '*'
        # digits.append(digit)
        # 小数点的识别
        # print('dot signal: ',cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9 / 16 * width * width))
        # if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
        #     digits.append('.')
        #     cv2.rectangle(output_img,
        #                   (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
        #                   (x1, y1), (0, 128, 0), 2)
        #     cv2.putText(output_img, 'dot',
        #                 (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
            # st.write("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
            # st.write(digit)
            # cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
    return digits
def find_copy(img):
    # lab
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB);
    # l,a,b = cv2.split(lab);

    # # show
    # cv2.imshow("orig", img);

    # # closing operation
    kernel = np.ones((5,5), np.uint8);
    # low = 165;
    # high = 250;
    iters = 2;


    # # make copy
    # copy = b.copy();

    # # threshold
    # thresh = cv2.inRange(copy, low, high);

    # dilate_______________________________________
    for a in range(iters):
        img = cv2.dilate(img, kernel);
    for a in range(iters):
        img = cv2.erode(img, kernel);
    # img = np.invert(img)
    # erode
    for a in range(iters):
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    # draw
    for contour in contours:
        cv2.drawContours(img, [contour], 0, 0, 1);
    #)(*)(*)(*
    # for b in range(iters):
    #     contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    # st.image(img)
    bounds = [];
    h, w = img.shape[:2];
    # contours = list(contours)
    for contour in contours:
        left = w;
        right = 0;
        top = h;
        bottom = 0;
        for point in contour:
            point = point[0];
            x, y = point;
            if x < left:
                left = x;
            if x > right:
                right = x;
            if y < top:
                top = y;
            if y > bottom:
                bottom = y;
        tl = [left, top];
        br = [right, bottom];
        bounds.append([tl, br, contour]);
    bounds.sort(key = lambda digit : digit[0][0])
    new_contour = []
    # st.write(bounds)
    big_holes = []
    for i in range(0, len(bounds)):
        # st.write(cv2.contourArea(bounds[i][2]))
        # temp = img.copy()
        # cv2.drawContours(temp, [bounds[i][2]], 0, 100, 20)
        # st.image(temp)
        if i > 0 and bounds[i][0][0] > new_contour[-1][0][0] and bounds[i][1][0] < new_contour[-1][1][0]:
            if cv2.contourArea(bounds[i][2]) * 10 > cv2.contourArea(new_contour[-1][2]):
                big_holes.append(bounds[i][2])
            continue
        else:
            new_contour.append(bounds[i])
    # st.write(big_holes)
    # cv2.drawContours(img, big_holes, 0, 180, 20)
    # st.image(img)
    cs = [a[2] for a in new_contour]
    bs = [[a[0], a[1]] for a in new_contour]
    cv2.fillPoly(img, pts =cs, color=255)
    cv2.fillPoly(img, pts =big_holes, color=0)
    # st.image(img)
    return (bs, img)
    # for b in range(1, len(bounds)):
    #     if bounds[b][0][0] 
    # return bounds
# def recursion(image, new_image, row, col, h, w):
#     if row < 0 or row >= h or col < 0 or col > w:
#         return
#     elif not image[row][col] and not new_image[row][col]:
#         new_image[row][col] = 1
#         recursion(image, new_image, row + 1, col, h, w)
#         recursion(image, new_image, row, col + 1, h, w)
def detect_digit(image, show_image):
    blurred, gray_img = load_image(image, show=show_image)
    output = blurred
    dst = preprocess(blurred, THRESHOLD, show=show_image)
    # st.image(dst)
    # gray_image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # st.image(gray_image)
    # st.write(dst.shape)
    # h, w = dst.shape
    # new_image = np.zeros([h,w], dtype = np.uint8)
    # st.write(type(dst[0][0]))
    # recursion(dst, new_image, 0, 0, h, w)
    # for row in range(h):
    #     for col in range(w):
    #         if not dst[row][col]:#and not new_image[row][col]:
    #             new_image[row][col] = 255
    #         else:
    #             break
    # for row in range(h):
    #     for col in range(w):
    #         if not dst[row][col]:#and not new_image[row][col]:
    #             new_image[row][col] = 255
    #         else:
    #             break

    # st.image(new_image)
    # for row in range(len(dst)):
    #     for col in range(len(dst[0])):
    #         # for val in range(len(dst[0][0])):
    #             # dst[row][col] = np.zeros(3, dtype = int)
    #         dst[row][col] = 1
            # print(image[row][col])
            # continue
    # dst = np.invert(dst)
    # erode1
    se = np.ones((7,7), dtype='uint8')
    for b in range(3):
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, se)
    for a in range(3):
        contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    # draw
    for contour in contours:
        cv2.drawContours(dst, [contour], 0, 255, 3);

    # st.write("after")
    # st.image(dst)
    digits_positions, dst = find_copy(dst)
    # st.write("after")
    # st.image(dst)
    # digits_positions.sort(key = lambda digit : digit[0][0])
    # st.image(dst)
    # newdigits = []
    # st.write(digits_positions)
    # for i in range(1, len(digits_positions)):
    #     if i > 1 and digits_positions[i][0][0] > newdigits[-1][0][0] and digits_positions[i][1][0] < newdigits[-1][1][0]:
    #         continue
    #     else:
    #         newdigits.append(digits_positions[i])
    digits = recognize_digits_line_method(digits_positions, output, dst)#______________________________________________________________________________________________________________________________
    



    # leftmost = digits_positions[0]
    # rightmost = digits_positions[-1]
    # fourpoints = np.array([digits_positions[0][0], [digits_positions[-1][1][0], digits_positions[-1][0][1]], digits_positions[-1][1], [digits_positions[0][0][0], digits_positions[0][1][1]]])
    # fourpoints = fourpoints.reshape((4, 2))
    # st.write(fourpoints)
    # st.image(dst)
    # dst = four_point_transform(dst, fourpoints)
    
    st.image(dst)
    # tessdata_dir_config = '--tessdata-dir "./"'
    result1 = pytesseract.image_to_string(dst, lang="7seg", config= ' --tessdata-dir "./modeldata" --psm 7 --oem 3 -c tessedit_char_whitelist=0123456789-')
    st.code("Results from Tesseract:" + str(result1))
    # digits = ""
    # for contour in digits_positions:
    #     # # Get the bounding rectangle of the contour
    #     # x, y, w, h = cv2.boundingRect(contour)
        
    #     # # Extract the region of interest (ROI)
    #     # roi = binary[y:y+h, x:x+w]
        
    #     # Set the configuration for Tesseract OCR
    #     # config = "--psm 7"  # Treat ROI as a single line of text a
    #     # x, y, w, h = cv2.boundingRect(contour)
        
    #     # Extract the region of interest (ROI)
    #     # st.write(contour)
    #     x0, y0 = contour[0]
    #     x1, y1 = contour[1]
    #     roi = dst[y0:y1, x0:x1]
    #     h, w = roi.shape
    #     suppose_W = max(1, int(h / H_W_Ratio))

    #     # 消除无关符号干扰
    #     if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
    #         continue
    #     if w > h:
    #         continue
    #     # st.write(w)
    #     # st.write(h)
    #     # st.write(suppose_W)
    #     # 对1的情况单独识别
    #     if w < suppose_W / 1.6:
    #         x0 = max(x0 + w - suppose_W, 0)
    #         roi = dst[y0:y1, x0:x1]
    #         w = roi.shape[1]
    #     # roi = dst[contour[0][1]:contour[1][1], contour[0][0]:contour[1][0]]
    #     st.write(contour)
    #     st.write(contour[1][1] - contour[0][1])
    #     st.write(contour[1][0] - contour[0][0])
    #     if contour[1][1] - contour[0][1] < contour[1][0] - contour[0][0]:
    #         st.write("ASDIKJHJKSHADKJHASKDJHDHAKSJDHA")
    #         continue
    #     # roi = dst[y:y+h, x:x+w]
    #     roi= np.invert(roi)
    #     st.image(roi)
    #     config = "--psm 10"  # Treat ROI as a single line of text
        
    #     # Perform OCR on the ROI
    #     # result = pytesseract.image_to_string(roi, config=config, lang="./7seg.traineddata")
    #     result = pytesseract.image_to_string(roi, lang="ssd", config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    #     st.write(result)
    #     digits += result.strip()







    
    # st.write(digits)


    # st.write(digits)______________________________________________________________________________________________________________________________
    # st.image(output)#______________________________________________________________________________________________________________________________
    time_out = ""
    count = 0
    for c in digits:
        if c != "*" and c != ".":
            # if count == 0:
            #     if c == 8:
            #         count += 1
            #         time_out += "1"
            #         continue
            if count == 2:
                time_out += "-"
                count = 0
            time_out += "1" if count == 0 and c == 8 else str(c)
            count += 1
    # print(time_out)
    # st.write(time_out)____________________________________________________________________________________
    return time_out


# def detect_digit(image, show_image):
#     # st.image(image)
#     blurred, gray_img = load_image(image, show=show_image)
#     # st.image(blurred)
#     # st.image(gray_img)
#     output = blurred
#     dst = preprocess(blurred, THRESHOLD, show=show_image)
#     # st.image(dst)
#     # digits_positions = find_digits_positions(dst)
#     # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);

#     digits_positions = find_copy(dst)
#     digits_positions.sort(key = lambda digits_positions : digits_positions[0][0])
#     #preprocess digits positions
#     # st.write(digits_positions)

#     # get res of each number
#     # bounds = [];
#     # h, w = image.shape[:2];
#     # for contour in contours:
#     #     left = w;
#     #     right = 0;
#     #     top = h;
#     #     bottom = 0;
#     #     for point in contour:
#     #         point = point[0];
#     #         x, y = point;
#     #         if x < left:
#     #             left = x;
#     #         if x > right:
#     #             right = x;
#     #         if y < top:
#     #             top = y;
#     #         if y > bottom:
#     #             bottom = y;
#     #     tl = [left, top];
#     #     br = [right, bottom];
#     #     bounds.append([tl, br]);

#     # crop out each number
#     cuts = [];
#     number = 0;
#     for bound in digits_positions:
#         tl, br = bound;
#         cut_img = dst[tl[1]:br[1], tl[0]:br[0]];
#         cuts.append(cut_img);
#         number += 1;
#         # cv2.imshow(str(number), cut_img);

#     # font 
#     font = cv2.FONT_HERSHEY_SIMPLEX;

#     # create a segment model
#     model = Segments();
#     index = 0;
#     for cut in cuts:
#         # save image
#         # cv2.imwrite(str(index) + "_" + str(number) + ".jpg", cut);

#         # process
#         model.digest(cut);
#         number = model.getNum();
#         st.write(number)
#         # cv2.imshow(str(index), cut);

#         # draw and save again
#         h, w = cut.shape[:2];
#         drawn = np.zeros((h, w, 3), np.uint8);
#         drawn[:, :, 0] = cut;
#         drawn = cv2.putText(drawn, str(number), (10,30), font, 1, (0,0,255), 2, cv2.LINE_AA);
#         # cv2.imwrite("drawn" + str(index) + "_" + str(number) + ".jpg", drawn);
        
#         index += 1;
#         # cv2.waitKey(0);



    # digits = recognize_digits_line_method(digits_positions, output, dst)
    # # print(digits)
    # # print(digits)
    # st.write(digits)
    # # st.write("output____")
    # st.image(output)
    # time_out = ""
    # count = 0
    # for c in digits:
    #     if c != "*":
    #         if count == 2:
    #             time_out += ":"
    #             count = 0
    #         time_out += str(c)
    #         count += 1
    # st.write("asjdhakd")
    # st.write(time_out)
    # return time_out

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # print(digits)
    # st.write(digits)
    # return digits