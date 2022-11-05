# import os
# os.chdir("fisheye-master")
# curDirtory = os.getcwd()
# print(curDirtory)

from cgi import print_form
from concurrent.futures import thread
from sys import prefix
from time import sleep
import cv2
from cv2 import imshow
from cv2 import WINDOW_NORMAL
from cv2 import namedWindow
from cv2 import waitKey
from cv2 import imwrite
from cv2 import VideoCapture
from cv2 import INTER_LINEAR
from torch import Size
import numpy as np

def recordVideo():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G')) 
    writer = cv2.VideoWriter("cam3.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 15, (2592, 1944))
    for i in range(10):
        ret, img = cap.read()
    for i in range(500):
        ret, img = cap.read()
        writer.write(img)
    writer.release()
    cap.release()

def testCamera():
    cap = cv2.VideoCapture(0)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # print("width: {} , height: {}".format(width, height))

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))       # 非常重要，否则该摄像头以yuv格式读取视频，非常卡顿
    # cap.set(cv2.cv2.CAP_PROP_BACKLIGHT, 50)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # print(cap.get(cv2.CAP_PROP_BACKLIGHT))
    # return

    ret,img = cap.read()
    for i in range(10):
        ret, img = cap.read()
    cv2.imwrite("test210.jpg", img)
    # writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

    # namedWindow("test", WINDOW_NORMAL)
    while(ret):
        cv2.imshow("test", img)
        char = cv2.waitKey(1)
        if(char == 27):         # press esc
            cv2.destroyAllWindows()
            break
        # writer.write(img)
        ret, img = cap.read()
        # 降噪
        # img = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)

    cap.release()
    # writer.release()

def drawCircle():
    img  = cv2.imread("test.jpg")
    if not img.shape[1]:
        print("failed to read img!")
    cc = (int(img.shape[1]/2)+5, int(img.shape[0]/2) - 50)
    cv2.circle(img, cc, 710, (255,255,255), 3)
    cv2.namedWindow("circle", WINDOW_NORMAL)
    imshow("circle", img)
    char = cv2.waitKey(0)
    if char == 27:
        return
    # cv2.imwrite("mask_pre.jpg", img)

def genMask():
    img = cv2.imread("mask_pre.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("thresh", WINDOW_NORMAL)
    # thresh = cv2.GaussianBlur(thresh, (5,5), 15)
    # (_, thresh) = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY)
    cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.imwrite("fishEyeMask.jpg", thresh)

def DeNoiseTest():
    img  = cv2.imread("test.jpg")
    img = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
    namedWindow("denoised", WINDOW_NORMAL)
    imshow("denoised", img)
    imwrite("denoised.jpg", img)
    char = waitKey()
    if char == 27:
        return

def findCircle():
    namedWindow("output", WINDOW_NORMAL)
    image = cv2.imread("test.jpg")
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, 0, 100, 100, 600)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # break
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)

def extractFrame():
    prefix = "/home/cyx/programes/Videos/gear360/lab_data/test4/"
    videoname = "360_0104"
    cap = cv2.VideoCapture(prefix+videoname+".MP4")
    ret,img = cap.read()
    for i in range(30*8):
        ret, img = cap.read()
    cv2.imwrite(prefix+videoname+".jpg", img)

def cutImg():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/"
    imgname = "360_0103_right_rotate_midpoint_left"
    frame = cv2.imread(prefix+imgname+".jpg")
    # left = frame[:, :frame.shape[1] // 2]      # 获取双鱼眼左半部分
    # right = frame[:, frame.shape[1] // 2:]      # 获取双鱼眼右半部分
    # cv2.imwrite(prefix+imgname+"_left.jpg", left)
    # cv2.imwrite(prefix+imgname+"_right.jpg", right)
    down = frame[frame.shape[0]//2:, :]
    up = frame[:frame.shape[0]//2, :]
    cv2.imwrite(prefix+imgname+"_down.jpg", down)
    cv2.imwrite(prefix+imgname+"_up.jpg", up)

def mremap():
    cvfs = cv2.FileStorage('rebuild.yml', cv2.FileStorage_READ)
    xMapArr = cvfs.getNode("xMapArr").mat()
    yMapArr = cvfs.getNode("yMapArr").mat()
    print(xMapArr)
    print(yMapArr)
    
    img_right = cv2.imread("/home/cyx/programes/C++/fisheye/AlgorithmTest/build-UnWarpper-Desktop_Qt_5_9_0_GCC_64bit-Debug/gear360/temp_right.jpg")

    img = cv2.remap(img_right,xMapArr,yMapArr,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("/home/cyx/programes/C++/fisheye/AlgorithmTest/build-UnWarpper-Desktop_Qt_5_9_0_GCC_64bit-Debug/gear360/temp_right_remap.jpg", img)

    cvfs.release()

def rawTest():
    raw_cvfs = cv2.FileStorage('/home/cyx/programes/C++/fisheye/fisheyeStitcher-master/utils/grid_xd_yd_3840x1920.yml.gz', cv2.FileStorage_READ)
    xMapArr = raw_cvfs.getNode("Xd").mat()
    yMapArr = raw_cvfs.getNode("Yd").mat()
    img_r = cv2.imread("/home/cyx/programes/C++/fisheye/fisheyeStitcher-master/scripts/r_img_crop.jpg")
    # img_right = cv2.imread("/home/cyx/programes/C++/fisheye/AlgorithmTest/build-UnWarpper-Desktop_Qt_5_9_0_GCC_64bit-Debug/gear360/360_0006_right_midpoint.jpg")
    img = cv2.remap(img_r, xMapArr, yMapArr, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("test_mls_remap.jpg", img)

    raw_cvfs.release()

def reproduceRemapMatrix():
    route = "/home/cyx/programes/C++/fisheye/fisheyeStitcher-master/scripts/undistortMap.yml"
    route1 = "/home/cyx/programes/C++/fisheye/fisheyeStitcher-master/utils/grid_xd_yd_3840x1920.yml.gz"
    raw_cvfs = cv2.FileStorage(route, cv2.FileStorage_READ)
    
    Xd = raw_cvfs.getNode("Xd").mat()
    Yd = raw_cvfs.getNode("Yd").mat()
    # cvfs = cv2.FileStorage('rebuild.yml', cv2.FileStorage_WRITE)
    cvfs = cv2.FileStorage('undistort.yml', cv2.FileStorage_WRITE)
    # NewXd = Xd / (1920 / 1280)
    # NewYd = Yd / (1770 / 1180)
    NewXd = Xd / (3544 / 2363)
    NewYd = Yd / (1772 / 1280)
    xMapArr = cv2.resize(NewXd, (2363, 1280), cv2.INTER_NEAREST)
    yMapArr = cv2.resize(NewYd, (2363, 1280), cv2.INTER_NEAREST)
    # xMapArr = cv2.resize(NewXd, (1280, 1180), cv2.INTER_NEAREST)
    # yMapArr = cv2.resize(NewYd, (1280, 1180), cv2.INTER_NEAREST)

    cvfs.write("xMapArr", xMapArr)
    cvfs.write("yMapArr", yMapArr)

    raw_cvfs.release()
    cvfs.release()

def undistortFromYml():
    route = "/home/cyx/programes/C++/fisheye/undistort.yml"
    cvfs = cv2.FileStorage(route, cv2.FileStorage_READ)

    xMapArr = cvfs.getNode("xMapArr").mat()
    yMapArr = cvfs.getNode("yMapArr").mat()
    img_raw = cv2.imread("/home/cyx/programes/C++/fisheye/extract_frames/360_0016_left.jpg")

    img = cv2.remap(img_raw, xMapArr, yMapArr, interpolation=INTER_LINEAR)
    cv2.imwrite("undistrot_left.jpg", img)

    cvfs.release()

def cutAndRemap():
    img = cv2.imread("/home/cyx/programes/C++/fisheye/extract_frames/testimg/undistrot_left.jpg")
    img_cut = img[:1180, 542:1822, :]  # height : width : channel
    print(img_cut.shape)
    cv2.imwrite("/home/cyx/programes/C++/fisheye/extract_frames/testimg/undistrot_left_cut.jpg", img_cut)

    cvfs = cv2.FileStorage("/home/cyx/programes/C++/fisheye/rebuild.yml", cv2.FileStorage_READ)

    xMapArr = cvfs.getNode("xMapArr").mat()
    yMapArr = cvfs.getNode("yMapArr").mat()

    img_remap = cv2.remap(img_cut, xMapArr, yMapArr, interpolation=INTER_LINEAR)
    cv2.imwrite("/home/cyx/programes/C++/fisheye/extract_frames/testimg/undistrot_left_cut_remap.jpg", img_remap)

    cvfs.release()
    

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_MBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # print(x,y)
        (img, mPoints) = param
        # img = param._1
        # mPoints = param._2
        mPoints.append([y,x])
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,255,0))
        # cv2.imshow("image", img)

def on_EVENT_FixPoint(event, x, y, flags, param):
    if event == cv2.EVENT_MBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # print(x,y)
        (img, mPoints) = param
        # img = param._1
        # mPoints = param._2
        mPoints.append([y,x])
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,255,0))
        # cv2.imshow("image", img)        

def getPoint():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_1/"
    imgname = "360_0103_left_midpoint"
    img1 = cv2.imread(prefix+imgname+".jpg")

        
    # cvfs = cv2.FileStorage(prefix+"controlPoints_1.yml", cv2.FileStorage_READ)

    # p = cvfs.getNode("p").mat()     # ndarray
    # q = cvfs.getNode("q").mat()
    # print(p.shape)
    # print(p)

    # cvfs.release()

    # p = p.tolist()
    # q = q.tolist()

    p = []
    q = []
    mPoints = []

    cv2.namedWindow("image", WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_FixPoint, (img1, mPoints))

    while(1):
        cv2.imshow("image", img1)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    # print(mPoints)

    cvfs2 = cv2.FileStorage(prefix+"addFixPoints.yml", cv2.FileStorage_WRITE)
    p = p + mPoints
    q = q + mPoints

    cvfs2.write("p", np.array(p))
    cvfs2.write("q", np.array(q))

    cvfs2.release()
    cv2.imwrite(prefix+imgname+"_addFix.jpg", img1)


def cutOverlapArea():
    overlap_width = 148
    pano_width = 3544
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/"
    imgname = "360_0063_left_midpoint"
    img = cv2.imread(prefix + imgname + ".jpg")

    _, overlap_width, _ = img.shape
    overlap_width = overlap_width // 2

    left = img[:, :overlap_width, :]
    cv2.imwrite(prefix + imgname + "_left.jpg", left)
    right = img[:, -overlap_width:, :]
    cv2.imwrite(prefix + imgname + "_right.jpg", right)

def markPointFunc():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_1/"
    imgname1 = "360_0103_left_midpoint_right_down"
    imgname2 = "360_0103_right_midpoint_left_down"

    img1 = cv2.imread(prefix+imgname1+".jpg")
    img2 = cv2.imread(prefix+imgname2+".jpg")

    p = []
    q = []

    cv2.namedWindow("img1", WINDOW_NORMAL)
    cv2.setMouseCallback("img1", on_EVENT_LBUTTONDOWN, (img1, p))
    cv2.namedWindow("img2", WINDOW_NORMAL)
    cv2.setMouseCallback("img2", on_EVENT_LBUTTONDOWN, (img2, q))

    while(1):
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    cv2.imwrite(prefix+imgname1+"_marked.jpg", img1)
    cv2.imwrite(prefix+imgname2+"_marked.jpg", img2)
    
    cvfs = cv2.FileStorage(prefix+"controlPoints.yml", cv2.FileStorage_WRITE)

    print(p)
    print(q)

    cvfs.write("p", np.array(p))
    cvfs.write("q", np.array(q))

    cvfs.release()

def getMarkedPoint():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_1/"
    cvfs = cv2.FileStorage(prefix+"controlPoints.yml", cv2.FileStorage_READ)

    bias = 842      # (y, x)
    p = cvfs.getNode("p").mat()
    q = cvfs.getNode("q").mat()
    for y in q:
        y[1] = y[1] + bias


    print(p)
    print(q)

    cvfs.release()
    cvfs2 = cv2.FileStorage(prefix+"controlPoints_translated.yml", cv2.FileStorage_WRITE)

    cvfs2.write("p", p)
    cvfs2.write("q", q)

    cvfs2.release()

def testYml():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_1/"
    imgname = "360_0103_left_midpoint_right_down_marked"
    cvfs = cv2.FileStorage(prefix+"res.yml", cv2.FileStorage_READ)

    img = cv2.imread(prefix+imgname+".jpg")
    yarr = cvfs.getNode("yarr").mat().astype(np.float32)
    xarr = cvfs.getNode("xarr").mat().astype(np.float32)
    img_deform = cv2.remap(img, xarr, yarr, interpolation=INTER_LINEAR)
    cv2.imwrite(prefix+imgname+"_deform_test.jpg", img_deform)

    cvfs.release()

def rotateImg():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/"
    imgname = "360_0103_right"
    img = cv2.imread(prefix+imgname+".jpg")
    h, w, _ = img.shape
    Rotate_M = cv2.getRotationMatrix2D((w//2,h//2), -0.85, 1)
    img_r = cv2.warpAffine(img, Rotate_M, (w,h))
    cv2.imwrite(prefix+imgname+"_rotate.jpg", img_r)

def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape

    # 圆心
    x0, y0 = src_w//2, src_h//2

    # 数组， 循环每个点
    range_arr = np.array([range(R)])
    # print(type(range_arr[0]))
    # print(range_arr.T)

    theta = Pi - (Pi/R)*(range_arr.T)
    # print(theta)
    temp_theta = np.tan(theta)**2
    # print(temp_theta)

    phi = Pi - (Pi/R)*range_arr
    temp_phi = np.tan(phi)**2

    tempu = r/(temp_phi + 1 + temp_phi/temp_theta)**0.5
    # print(len(tempu), len(tempu[0]))
    tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

    # 用于修正正负号
    flag = np.array([-1] * r + [1] * r)

    # 加0.5是为了四舍五入求最近点
    u = x0 + tempu * flag + 0.5
    v = y0 + tempv * np.array([flag]).T + 0.5

    print(u.shape, v.shape)

    # # 防止数组溢出
    # u[u<0]=0
    # u[u>(src_w-1)] = src_w-1
    # v[v<0]=0
    # v[v>(src_h-1)] = src_h-1

    # 插值
    # dst[:, :, :] = src[v.astype(int),u.astype(int)]
    dst = cv2.remap(src, u.astype(np.float32), v.astype(np.float32), interpolation=INTER_LINEAR)
    return dst

def DoubleLongitude():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/"
    imgname = "360_0103_right_rotate"
    img = cv2.imread(prefix+imgname+".jpg")
    r = int(1020 * 1.1)
    img_ud = undistort(img, r)
    cv2.imwrite(prefix+imgname+"_dl.jpg", img_ud)

def main():
    # testCamera()
    # recordVideo()
    # drawCircle()
    # genMask()
    # DeNoiseTest()
    # findCircle()
    # extractFrame()
    # cutImg()
    # mremap()
    # rawTest()
    # reproduceRemapMatrix()
    # undistortFromYml()
    # cutAndRemap()
    # getPoint()
    # cutOverlapArea()
    # markPointFunc()
    # getMarkedPoint()
    # testYml()
    # rotateImg()
    DoubleLongitude()


if __name__ == '__main__':
    main()