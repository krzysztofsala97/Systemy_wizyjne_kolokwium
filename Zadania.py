from __future__ import print_function
import cv2
import numpy as np
#from cv2 import xfeatures2d
import argparse

def ex1():
    img = cv2.imread("bliss.jpg", 0)
    mask = np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])
    img2 = img.copy()
    cv2.filter2D(img, -1, mask, img2)
    img2 = abs(img2)
    img2 = img2 / img2.max()
    img2 = img2 * 255

    cv2.imshow('Oryginał', img)
    cv2.imshow("Po filtracji", img2)

    cv2.waitKey(0)


def ex2():
    img_in = cv2.imread("zad_bit.png", 0)
    img_out = np.zeros(img_in.shape)

    for i in range(0, img_in.shape[0]-1):
        for j in range(0, img_in.shape[1]-1):
            if img_in[i, j] % 2 == 1:
                img_out[i, j] = 225

    print(img_out.max())
    cv2.imshow('Oryginał', img_in)
    cv2.imshow("Ukryta wiadomość", img_out)

    cv2.waitKey(0)


def ex3():
    def rotate(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            LU = img_out[0:y, 0:x].copy()
            LD = img_out[y:img_out.shape[0], 0:x].copy()
            RU = img_out[0:y, x:img_out.shape[1]].copy()
            RD = img_out[y:img_out.shape[0], x:img_out.shape[1]].copy()

            img_out[0:y, 0:x] = cv2.resize(LD, (x, y))
            img_out[0:y, x:img_out.shape[1]] = cv2.resize(LU, (img_out.shape[1]-x, y))
            img_out[y:img_out.shape[0], x:img_out.shape[1]] = cv2.resize(RU, (img_out.shape[1]-x, img_out.shape[0]-y))
            img_out[y:img_out.shape[0], 0:x] = cv2.resize(RD, (x, img_out.shape[0]-y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            img_out[0:y, 0:x] = img_in[0:y, 0:x]
            img_out[0:y, x:img_out.shape[1]] = img_in[0:y, x:img_in.shape[1]]
            img_out[y:img_out.shape[0], x:img_out.shape[1]] = img_in[y:img_in.shape[0], x:img_in.shape[1]]
            img_out[y:img_out.shape[0], 0:x] = img_in[y:img_in.shape[0], 0:x]

    img_in = cv2.imread("bliss.jpg")
    img_out = img_in.copy()
    cv2.namedWindow("Okno")
    cv2.setMouseCallback("Okno", rotate)
    while True:
        cv2.imshow("Okno", img_out)
        k = cv2.waitKey(1)
        if k == 32:
            break


def ex4():
    def callback(value):
        pass
    cv2.namedWindow("Okno")
    cv2.createTrackbar("bar", "Okno", 0, 100, callback)
    img = cv2.imread("bliss.jpg")
    copy = img.copy()

    while True:
        percent = cv2.getTrackbarPos("bar", "Okno")/100
        if percent == 0:
            copy = img.copy()
        elif percent == 1:
            copy = 255 - img.copy()
        else:
            copy[:, 0:int(copy.shape[1]*percent), :] = 255 - img[:, 0:int(copy.shape[1]*percent), :].copy()
            copy[:, int(copy.shape[1] * percent):copy.shape[1]-1, :] = img[:, int(copy.shape[1] * percent):copy.shape[1] - 1, :].copy()

        cv2.imshow("Okno", copy)
        key_code = cv2.waitKey(1)
        if key_code == 32:
            break


def ex5():
    pts = []
    img = cv2.imread("bliss.jpg")
    img_grey = cv2.imread("bliss.jpg", 0)

    def pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    cv2.namedWindow("Okno")
    cv2.setMouseCallback("Okno", pick_points)
    cv2.imshow("Okno", img)
    #cv2.imshow("Okno", img_grey)
    while len(pts) < 2:
        key_code = cv2.waitKey(10)
        if key_code > 0:
            break

    if pts[0][0] < pts[1][0]:
        first_x = pts[0][0]
        second_x = pts[1][0]
    else:
        first_x = pts[1][0]
        second_x = pts[0][0]

    if pts[0][1] < pts[1][1]:
        first_y = pts[0][1]
        second_y = pts[1][1]
    else:
        first_y = pts[1][1]
        second_y = pts[0][1]

    canny_square = cv2.Canny(img_grey[first_y:second_y, first_x:second_x], 50, 150)
    img[first_y:second_y, first_x:second_x,0] = canny_square
    img[first_y:second_y, first_x:second_x, 1] = canny_square
    img[first_y:second_y, first_x:second_x, 2] = canny_square
    #img_grey[first_y:second_y, first_x:second_x] = canny_square
    cv2.imshow("Okno", img)
    #cv2.imshow("Okno",img_grey)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def ex6():
    pts = []
    img = cv2.imread("bliss.jpg")
    copy = img.copy()
    negative = img.copy()

    def pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    cv2.namedWindow("Okno")
    cv2.setMouseCallback("Okno", pick_points)
    cv2.imshow("Okno", copy)

    while True:
        if len(pts) == 2:
            if pts[0][0] < pts[1][0]:
                first_x = pts[0][0]
                second_x = pts[1][0]
            else:
                first_x = pts[1][0]
                second_x = pts[0][0]

            if pts[0][1] < pts[1][1]:
                first_y = pts[0][1]
                second_y = pts[1][1]
            else:
                first_y = pts[1][1]
                second_y = pts[0][1]
            negative[:, :, 0] = 255 - copy[:, :, 0]
            negative[:, :, 1] = 255 - copy[:, :, 1]
            negative[:, :, 2] = 255 - copy[:, :, 2]
            copy[first_y:second_y, first_x:second_x] = negative[first_y:second_y, first_x:second_x]
            pts.clear()
            cv2.imshow("Okno", copy)
        if cv2.waitKey(20) & 0xFF == 27:
            break


def ex7():
    img = cv2.imread("zad_tygrys.png")
    img_grey = cv2.imread("zad_tygrys.png", 0)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if 150 < img_grey[i, j] < 200:
                img[i, j, 0] = 0
                img[i, j, 1] = 165
                img[i, j, 2] = 255
    cv2.imshow("original", img_grey)
    cv2.imshow("painted", img)

    cv2.waitKey(0)


def ex8():
    img = cv2.imread('zad_cukierki.jpg')
    image = img.copy()
    cts = []
    kernel = np.ones((5, 5), np.uint8)
    lower_hsv = np.array([87, 27, 37])
    higher_hsv = np.array([159, 151, 131])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 4000:
            cts.append(contour)
    cv2.drawContours(image, cts, -1, (0, 255, 0), 3)

    cv2.imshow("Lizaki", image)
    cv2.waitKey(0)


def ex9():
    img1 = cv2.imread("zad_nakaz_2.jpg")
    img2 = cv2.imread("zad_nakaz_3.jpg")
    img1_blur = cv2.GaussianBlur(img1, (7, 7), 0)
    img2_blur = cv2.GaussianBlur(img2, (7, 7), 0)

    hsv1 = cv2.cvtColor(img1_blur, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_blur, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)

    blue1 = cv2.inRange(hsv1, blue_lower, blue_upper)
    blue2 = cv2.inRange(hsv2, blue_lower, blue_upper)

    cnt1, h1 = cv2.findContours(blue1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt2, h2 = cv2.findContours(blue2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts1Sorted = sorted(cnt1, key=lambda x: cv2.contourArea(x))
    cnts2Sorted = sorted(cnt2, key=lambda x: cv2.contourArea(x))

    cv2.drawContours(img1, [cnts1Sorted[len(cnts1Sorted)-1]], -1, (255, 0, 255), 2)
    cv2.drawContours(img2, [cnts2Sorted[len(cnts2Sorted)-1]], -1, (255, 0, 255), 2)

    cv2.imshow("Result1", img1)
    cv2.imshow("Result2", img2)

    cv2.waitKey(0)


def ex10():
    img1 = cv2.imread("zad_kolka_1.png")
    print(img1.shape)
    blur = cv2.GaussianBlur(img1, (3, 3), 0)

    kernel = np.ones((3, 3), dtype=np.uint8)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([20, 255, 255])

    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    red = cv2.inRange(hsv, red_lower, red_upper)

    blue = cv2.dilate(blue, kernel, iterations=5)
    red = cv2.dilate(red, kernel, iterations=5)

    cnts_blue, h_b = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_red, h_r = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img1, cnts_blue, -1, (0, 0, 255), 2)
    cv2.drawContours(img1, cnts_red, -1, (255, 0, 0), 2)
    cv2.imshow('image', img1)

    cv2.waitKey(0)


def ex11():
    img = cv2.imread("zad_korona.jpg")
    copy = img.copy()
    kernel = np.ones((5, 5), dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    emerald_lower = np.array([36, 25, 25], np.uint8)
    emerald_upper = np.array([70, 255, 255], np.uint8)

    amethyst_lower = np.array([150, 25, 25], np.uint8)
    amethyst_upper = np.array([175, 255, 255], np.uint8)

    #ruby_lower = np.array([10, 25, 25], np.uint8)
    #ruby_upper = np.array([17, 255, 255], np.uint8)

    emerald = cv2.inRange(hsv, emerald_lower, emerald_upper)
    amethyst = cv2.inRange(hsv, amethyst_lower, amethyst_upper)
    #ruby = cv2.inRange(hsv, ruby_lower, ruby_upper)

    emerald = cv2.morphologyEx(emerald, cv2.MORPH_CLOSE, kernel, iterations=3)
    amethyst = cv2.morphologyEx(amethyst, cv2.MORPH_CLOSE, kernel, iterations=3)
    #ruby = cv2.morphologyEx(ruby, cv2.MORPH_CLOSE, kernel, iterations=3)

    cnts_e, h_e = cv2.findContours(emerald, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_a, h_a = cv2.findContours(amethyst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts_r, h_r = cv2.findContours(ruby, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts_e_sorted = sorted(cnts_e, key=lambda x: cv2.contourArea(x), reverse=True)
    cnts_a_sorted = sorted(cnts_a, key=lambda x: cv2.contourArea(x), reverse=True)
    #cnts_r_sorted = sorted(cnts_r, key=lambda x: cv2.contourArea(x), reverse=True)

    cv2.drawContours(copy, cnts_e_sorted, -1, (255, 0, 0), 5)
    cv2.imshow("Emeralds", copy)
    copy = img.copy()

    cv2.drawContours(copy, cnts_a_sorted, -1, (255, 0, 0), 5)
    cv2.imshow("Amethysts", copy)
    copy = img.copy()

    #cv2.drawContours(copy, cnts_r_sorted, -1, (255, 0, 0), 5)
    #cv2.imshow("Rubies", copy)
    #copy = img.copy()

    cv2.waitKey(0)


def ex12():
    img = cv2.imread("zad_monety_1.png")
    grey = cv2.imread("zad_monety_1.png", 0)

    grey = 255 - grey

    blur = cv2.blur(grey, (7, 7), 0)

    ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 50, param1=1, param2=10, minRadius=5, maxRadius=100)

    zl = 0
    gr = 0

    for i in circles[0, :]:

        if i[2] > 45:
            zl = zl + 1
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        else:
            gr = gr + 1
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    print(f'Jest {zl} pięciozłotówek i {gr} grosików.')

    cv2.imshow('result', img)
    cv2.waitKey(0)


def ex13():
    points = []
    MIN_MATCH_COUNT = 5

    def getPoints(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            points.append((x, y))

    def match(image_src, image_to_find):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', getPoints)
        while True:
            cv2.imshow('Image', image_src)
            if cv2.waitKey(20) & 0xFF == 27:
                break
            if len(points) >= 2:
                top_left = points[0]
                bottom_right = points[1]
                roi = image_src[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                cv2.imshow('roi', roi)
                cv2.waitKey(10)
                sift = cv2.xfeatures2d.SIFT_create()
                kp1, des1 = sift.detectAndCompute(roi, None)
                kp2, des2 = sift.detectAndCompute(image_to_find, None)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask.ravel().tolist()
                    h, w, d = roi.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    image_to_find = cv2.polylines(image_to_find, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

                else:
                    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                    matchesMask = None
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                # img3 = cv.drawMatches(roi, kp1, image_to_find, kp2, good, None, **draw_params)
                cv2.imshow('new', image_to_find)
                cv2.waitKey(10000)

    def main():
        image_src = cv2.imread('zad_dopasowanie_obiekty_wzorcowe.jpg')
        image_src = cv2.resize(image_src, (0, 0), fx=0.25, fy=0.25)
        image_to_find = cv2.imread('zad_dopasowanie_obiekty_do_znalezienia.jpg')
        image_to_find = cv2.resize(image_to_find, (0, 0), fx=0.25, fy=0.25)
        match(image_src, image_to_find)

    if __name__ == '__main__':
        main()


def ex14():
    pts = []

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            pts.append([x, y])

    image = cv2.imread('zad_tv_image_3.png')
    video = cv2.VideoCapture('zad_tv_movie_to_insert_1.wmv')

    cv2.namedWindow("Okno")
    cv2.setMouseCallback("Okno", callback)

    play = True
    quit = False
    while True:
        cv2.imshow("Okno", image)
        cv2.waitKey(10)
        if len(pts) >= 4:
            insert_pts = np.float32([pts[0], pts[1], pts[2], pts[3]])
            while play:
                cv2.imshow("Okno", image)
                ret, frame = video.read()
                if ret:
                    frame_pts = np.float32([[0, 0], [np.shape(frame)[1] - 1, 0], [np.shape(frame)[1] - 1, np.shape(frame)[0] - 1], [0, np.shape(frame)[0] - 1]])
                    M = cv2.getPerspectiveTransform(frame_pts, insert_pts)
                    dst = cv2.warpPerspective(frame, M, (np.shape(image)[1], np.shape(image)[0]))
                    ret, mask = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)
                    image = cv2.bitwise_and(image, mask_inv)
                    image = cv2.add(image, dst)
                    k = cv2.waitKey(30)
                    if k == 32:
                        play = False
                else:
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            k = cv2.waitKey(0)
            if k == 32:
                play = True


def ex15():
    pts = []

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            pts.append([x, y])

    img = cv2.imread("zad_memo.jpg")
    cv2.namedWindow("okno")
    cv2.setMouseCallback("okno", callback)
    while True:
        cv2.imshow("okno", img)
        if len(pts) >= 2:
            if pts[0][0] < pts[1][0]:
                first_x = pts[0][0]
                second_x = pts[1][0]
            else:
                first_x = pts[1][0]
                second_x = pts[0][0]

            if pts[0][1] < pts[1][1]:
                first_y = pts[0][1]
                second_y = pts[1][1]
            else:
                first_y = pts[1][1]
                second_y = pts[0][1]
            src = img[first_y:second_y, first_x:second_x]
            w, h = src.shape[:2]
            copy = img.copy()
            copy[first_y:second_y, first_x:second_x] = 0

            result = cv2.matchTemplate(copy, src, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + h, top_left[1] + w)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
            cv2.rectangle(img, (first_x, first_y), (second_x, second_y), (0, 0, 255), 2)
            pts.clear()
        if cv2.waitKey(20) & 0xFF == 32:
            break
    cv2.destroyAllWindows()


def ex16():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')  # CO
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                        default='zad_ruch.mpg')  # CO
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')  # CO
    args = parser.parse_args()
    # stworzenie narzędzia ktore tworzy maske tla z domyslnymi argumentami
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    # wczytanie sekwencji video
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # obliczanie maski dla dla danej klatki
        fgMask = backSub.apply(frame)
        # troche odfiltrowanie stworzonego obrazu, duzo szumu przez zmieniajaca sie jasnosc i ujednolicenie obiektow
        kernel = np.ones((3, 3), np.uint8)
        ret2, fgMask = cv2.threshold(fgMask, 0, 255, cv2.THRESH_OTSU)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        # wykrycie konturów
        height, width = fgMask.shape
        cnts = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # CO TEN IF ROBI
        # rysowanie ramek wokól obiektow
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 200:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        # cv.imshow('FG Mask', fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

def ad_ex1_2pts():
    pts = []

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            pts.append([x, y])

    cv2.namedWindow("okno")
    background = cv2.imread("zad_tv_image_to_insert_3.jpg")
    background = cv2.resize(background, None, fx=0.25, fy=0.25)
    insert_img = cv2.imread("zad_panda_2.jpg")
    cv2.setMouseCallback("okno", callback)
    while True:
        cv2.imshow("okno", background)
        if len(pts) >= 2:
            if pts[0][0] < pts[1][0]:
                first_x = pts[0][0]
                second_x = pts[1][0]
            else:
                first_x = pts[1][0]
                second_x = pts[0][0]

            if pts[0][1] < pts[1][1]:
                first_y = pts[0][1]
                second_y = pts[1][1]
            else:
                first_y = pts[1][1]
                second_y = pts[0][1]
            pts_clicked = np.float32([[first_x, first_y], [second_x, first_y], [second_x, second_y], [first_x, second_y]])
            pts_insert = np.float32(
                [[0, 0], [np.shape(insert_img)[1] - 1, 0], [np.shape(insert_img)[1] - 1, np.shape(insert_img)[0] - 1],
                 [0, np.shape(insert_img)[0] - 1]])
            M = cv2.getPerspectiveTransform(pts_insert, pts_clicked)
            dst = cv2.warpPerspective(insert_img, M, (np.shape(background)[1], np.shape(background)[0]))
            ret, mask = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(background, mask_inv)
            background = cv2.add(background, dst)
            pts.clear()
        k = cv2.waitKey(1)
        if k == 27 or k == 32:
            break

    cv2.destroyAllWindows()


def ad_ex1_4_pts():
    pts = []

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            pts.append([x, y])

    cv2.namedWindow("okno")
    background = cv2.imread("zad_tv_image_to_insert_3.jpg")
    background = cv2.resize(background, None, fx=0.25, fy=0.25)
    insert_img = cv2.imread("zad_panda_2.jpg")
    cv2.setMouseCallback("okno", callback)
    while True:
        cv2.imshow("okno", background)
        if len(pts) >= 4:
            pts_clicked = np.float32([pts[0], pts[1], pts[2], pts[3]])
            pts_insert = np.float32([[0, 0], [np.shape(insert_img)[1] - 1, 0], [np.shape(insert_img)[1] - 1, np.shape(insert_img)[0] - 1], [0, np.shape(insert_img)[0] - 1]])
            M = cv2.getPerspectiveTransform(pts_insert, pts_clicked)
            dst = cv2.warpPerspective(insert_img, M, (np.shape(background)[1], np.shape(background)[0]))
            ret, mask = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(background, mask_inv)
            background = cv2.add(background, dst)
            pts.clear()
        k = cv2.waitKey(1)
        if k == 27 or k == 32:
            break

    cv2.destroyAllWindows()


def ad_ex2():
    img1 = cv2.imread("zad_znak_stop_1.jpg")
    img2 = cv2.imread("zad_znak_stop_2.jpg")
    img1_blur = cv2.GaussianBlur(img1, (7, 7), 0)
    img2_blur = cv2.GaussianBlur(img2, (7, 7), 0)

    hsv1 = cv2.cvtColor(img1_blur, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_blur, cv2.COLOR_BGR2HSV)

    red_lower = np.array([175, 100, 100])
    red_upper = np.array([179, 255, 255])

    red1 = cv2.inRange(hsv1, red_lower, red_upper)
    red2 = cv2.inRange(hsv2, red_lower, red_upper)

    cnt1, h1 = cv2.findContours(red1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt2, h2 = cv2.findContours(red2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts1Sorted = sorted(cnt1, key=lambda x: cv2.contourArea(x))
    cnts2Sorted = sorted(cnt2, key=lambda x: cv2.contourArea(x))

    cv2.drawContours(img1, [cnts1Sorted[len(cnts1Sorted) - 1]], -1, (255, 0, 255), 2)
    cv2.drawContours(img2, [cnts2Sorted[len(cnts2Sorted) - 1]], -1, (255, 0, 255), 2)

    cv2.imshow("Result1", img1)
    cv2.imshow("Result2", img2)

    cv2.waitKey(0)


#ex1()
#ex2()
#ex3()
#ex4()
#ex5()
#ex6()
#ex7()
#ex8()
#ex9()
#ex10()
#ex11()
#ex12()
#ex13()
#ex14()
#ex15()
#ex16()
#ad_ex1_2pts()
#ad_ex1_4_pts()
ad_ex2()