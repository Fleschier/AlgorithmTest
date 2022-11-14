# import os
# os.chdir("fisheye-master")
# curDirtory = os.getcwd()
# print(curDirtory)

from audioop import bias
from cgi import print_form
from concurrent.futures import thread
from readline import write_history_file
from sys import prefix
from time import sleep
from tkinter import Y
from tkinter.tix import X_REGION
import cv2
from cv2 import imshow
from cv2 import WINDOW_NORMAL
from cv2 import namedWindow
from cv2 import waitKey
from cv2 import imwrite
from cv2 import VideoCapture
from cv2 import INTER_LINEAR
from cv2 import CV_16SC3
from torch import Size, int16
import numpy as np

import math

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
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/add_bias/"
    imgname = "0103_right_deform"
    frame = cv2.imread(prefix+imgname+".jpg")
    left = frame[:, :frame.shape[1] // 2]      # 获取双鱼眼左半部分
    right = frame[:, frame.shape[1] // 2:]      # 获取双鱼眼右半部分
    cv2.imwrite(prefix+imgname+"_left.jpg", left)
    cv2.imwrite(prefix+imgname+"_right.jpg", right)
    # down = frame[frame.shape[0]//2:, :]
    # up = frame[:frame.shape[0]//2, :]
    # cv2.imwrite(prefix+imgname+"_down.jpg", down)
    # cv2.imwrite(prefix+imgname+"_up.jpg", up)

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
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/"
    imgname = "0104_right_calib"
    img1 = cv2.imread(prefix+imgname+".jpg")

    # p = []
    # q = []
    mPoints = []

    cv2.namedWindow("image", WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_FixPoint, (img1, mPoints))

    while(1):
        cv2.imshow("image", img1)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    print(mPoints)

    # cvfs2 = cv2.FileStorage(prefix+"addFixPoints.yml", cv2.FileStorage_WRITE)
    # p = p + mPoints
    # q = q + mPoints

    # cvfs2.write("p", np.array(p))
    # cvfs2.write("q", np.array(q))

    # cvfs2.release()
    # cv2.imwrite(prefix+imgname+"_addFix.jpg", img1)


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
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/"
    # imgname1 = "0103_right_calib_deform_right_down"     # p
    # imgname2 = "0103_left_calib_deform_right_down"      # q     # q不变， p --> q
    imgname1 = "0104_right_calib_deform_left_down_p1"     
    imgname2 = "0104_left_calib_deform_left_down_p1"      

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
    
    cvfs = cv2.FileStorage(prefix+"raw_controlPoints.yml", cv2.FileStorage_WRITE)

    print(p)
    print(q)

    cvfs.write("p", np.array(p))
    cvfs.write("q", np.array(q))

    cvfs.release()

def getMarkedPoint():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/marked_data/"
    cvfs = cv2.FileStorage(prefix+"raw_controlPoints.yml", cv2.FileStorage_READ)

    # bias = 842      # (y, x)
    # p = cvfs.getNode("p").mat()
    # q = cvfs.getNode("q").mat()
    # for y in q:
    #     y[1] = y[1] + bias

    # print(p)
    # print(q)

    imgname = "360_0103_left_right_down"
    imgname2 = "360_0103_right_rotate_left_down"
    img1 = cv2.imread(prefix+imgname+".jpg")
    img2 = cv2.imread(prefix+imgname2+".jpg")
    p = cvfs.getNode("p").mat().tolist()
    q = cvfs.getNode("q").mat().tolist()
    p = p[:-2]
    q = q[:-2]
    print(p)
    for i in p:
        cv2.circle(img1, (i[1], i[0]), 2, (0, 255, 0), -1)
    for j in q:
        cv2.circle(img2, (j[1], j[0]), 2, (0, 255, 0), -1)
    cv2.imwrite(prefix+imgname+"_marked.jpg", img1)
    cv2.imwrite(prefix+imgname2+"_marked.jpg", img2)

    cvfs.release()

    cvfs2 = cv2.FileStorage(prefix+"raw_controlPoints_part1.yml", cv2.FileStorage_WRITE)

    cvfs2.write("p", np.array(p))
    cvfs2.write("q", np.array(q))

    cvfs2.release()

def testYml():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test5_undistort_add_bias/"

    # # undistort
    # # imgname = "0104_left"
    # # cvfs = cv2.FileStorage(prefix+"equirectangular_left_bias.yml", cv2.FileStorage_READ)
    # imgname = "0104_right"
    # cvfs = cv2.FileStorage(prefix+"equirectangular_bias.yml", cv2.FileStorage_READ)

    # img = cv2.imread(prefix+imgname+".jpg")
    # yarr = cvfs.getNode("yMapArr").mat().astype(np.float32)
    # xarr = cvfs.getNode("xMapArr").mat().astype(np.float32)
    # img_deform = cv2.remap(img, xarr, yarr, interpolation=INTER_LINEAR)
    # cv2.imwrite(prefix+imgname+"_deform.jpg", img_deform)

    # mls deform
    imgname = "0104_right_deform"
    cvfs = cv2.FileStorage(prefix+"mls.yml", cv2.FileStorage_READ)

    img = cv2.imread(prefix+imgname+".jpg")
    yarr = cvfs.getNode("yarr").mat().astype(np.float32)
    xarr = cvfs.getNode("xarr").mat().astype(np.float32)
    img_deform = cv2.remap(img, xarr, yarr, interpolation=INTER_LINEAR)
    cv2.imwrite(prefix+imgname+"_mls.jpg", img_deform)

    cvfs.release()

def rotateImg():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/"
    imgname = "360_0103_right"
    img = cv2.imread(prefix+imgname+".jpg")
    h, w, _ = img.shape
    Rotate_M = cv2.getRotationMatrix2D((w//2,h//2), -0.8, 1)
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

def MergeYml():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/marked_data_0104/"
    yml1 = "raw_controlPoints_p1.yml"
    yml2 = "raw_controlPoints_p2.yml"
    cvfs1 = cv2.FileStorage(prefix+yml1, cv2.FileStorage_READ)
    cvfs2 = cv2.FileStorage(prefix+yml2, cv2.FileStorage_READ)
    cvfs_w = cv2.FileStorage(prefix+"controlPoints_merged.yml", cv2.FileStorage_WRITE)

    p1 = cvfs1.getNode("p").mat().tolist()
    q1 = cvfs1.getNode("q").mat().tolist()
    p2 = cvfs2.getNode("p").mat().tolist()
    q2 = cvfs2.getNode("q").mat().tolist()
    p = p1 + p2
    q = q1 + q2
    cvfs_w.write("p", np.array(p))
    cvfs_w.write("q", np.array(q))

    cvfs1.release()
    cvfs2.release()
    cvfs_w.release()

def DrawPointsFromYml():
    # prefix="/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/marked_data_0103/"
    # yml = "biasFix_extend.yml"
    prefix="/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/marked_data_0104/"
    yml = "fix_cut_bias.yml"
    name1 = "0104_right_deform_l"
    name2 = "0104_left_calib_deform_left"
    img_l = cv2.imread(prefix+name1+".jpg")
    img_r = cv2.imread(prefix+name2+".jpg")

    cvfs = cv2.FileStorage(prefix+yml, cv2.FileStorage_READ)
    p = cvfs.getNode("p").mat().tolist()
    q = cvfs.getNode("q").mat().tolist()
    # p = p[:-5]
    # print(len(p))
    # q = q[:-3]
    # print(len(q))
    for i in p:
        cv2.circle(img_l, (int(i[1]), int(i[0])), 2, (0, 255, 0), -1)
    for j in q:
        cv2.circle(img_r, (int(j[1]), int(j[0])), 2, (0, 255, 0), -1)

    cvfs.release()
    # cvfs2 = cv2.FileStorage(prefix+"fix_cut_bias.yml", cv2.FileStorage_WRITE)
    # for i in p:
    #     i[1] = i[1] - 18
    # for j in q:
    #     j[1] = j[1] - 18

    # cvfs2.write("p", np.array(p))
    # cvfs2.write("q", np.array(q))
    # cvfs2.release()

    cv2.imwrite(prefix+name1+"_marked.jpg", img_l)
    cv2.imwrite(prefix+name2+"_marked.jpg", img_r)
    

def ForwardEquiRecProjection(x_src, y_src):
    # W = (int)(1920*(360/195))
    # H = 1920
    # f_a = 195
    # f = (math.pi/180)*f_a       # convert degree to radius
    # theta = f*(x_src/W) - 0.5
    # phi = f*(y_src/H) - 0.5
    # x = math.cos(phi)*math.sin(theta)
    # y = math.cos(phi)*math.cos(theta)
    # z = math.sin(phi)
    # lu = (H/f)*(1/math.tan(math.sqrt(x*x + z*z)/y))
    # theta = 1/math.tan(z/x)
    # x_dst = 0.5*W + lu*math.cos(theta)
    # y_dst = 0.5*H + lu*math.sin(theta)

    w_src = 1914
    h_src = 1900
    R195 = 1009
    R180 = 900
    rate = 2*(R195 - R180) / (195 - 180)
    FOV = (180 + (w_src-2*R180)/rate)*(math.pi/180)
    FOV2 = (180 + (h_src-2*R180)/rate)*(math.pi/180)
    theta = math.pi * (x_src/w_src - 0.5)  # -pi ~ pi
    phi = math.pi * (y_src/h_src - 0.5) # -pi/2 ~ pi/2
    x = math.cos(phi)*math.sin(theta)
    y = math.cos(phi)*math.cos(theta)
    z = math.sin(phi)
    theta = math.atan2(z,x)
    phi = math.atan2(math.sqrt(x*x + z*z), y)
    r = w_src * phi / FOV
    r2 = h_src * phi / FOV2
    x_dst = 0.5*w_src + r*math.cos(theta)
    y_dst = 0.5*h_src + r2*math.sin(theta)

    return (x_dst,y_dst)

def RevEquiRecProjection(x_dst, y_dst, w_rad):
    phi = x_dst / w_rad
    theta = -y_dst / w_rad + math.pi/2
    if theta < 0:
        theta = -theta
        phi = phi + math.pi
    if theta > math.pi:
        theta = math.pi - (theta - math.pi)
        phi = phi + math.pi
    s = math.sin(theta)
    v0 = s * math.sin(phi)
    v1 = math.cos(theta)
    r = math.sqrt(v0*v0 + v1*v1)
    theta = w_rad*math.atan2(r, s*math.cos(phi))

    src_x = theta * v0 / r
    src_y = theta * v1 / r
    
    return (src_x, src_y)

def GetUndistortMap():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/"
    yml = "equirectangular.yml"
    cvfs = cv2.FileStorage(prefix+yml, cv2.FileStorage_WRITE)

    # m_wd = (int)(1920*(360/195))
    # m_wd = (int)(1920*(360/183.5))
    # m_wd = m_wd - (m_wd%2)  # make it even
    src_w = 1914
    src_h = 1900
    m_wd = src_w * 2
    m_hd = src_h
    # w_rad = m_wd / (2*math.pi)
    # w2 = m_wd/2 - 0.5
    # h2 = m_hd/2 - 0.5
    # ws2 = src_w/2 - 0.5
    # hs2 = src_h/2 - 0.5
    xMapArr = np.zeros((m_hd, m_wd)).astype(np.float32)
    yMapArr = np.zeros((m_hd, m_wd)).astype(np.float32)

    for y in range(0,m_hd):
        # y_dst = y - h2
        # for x in range(-int(m_wd/4),int(3*m_wd/4)):
        for x in range(0,m_wd):
            # x_dst = x - w2
            # (x_src, y_src) = RevEquiRecProjection(x_dst=x_dst,y_dst=y_dst,w_rad=w_rad)
            x_dst = x + int(m_wd/4)
            (x_src, y_src) = ForwardEquiRecProjection(x_dst, y)
            # x_src = x_src + ws2
            # y_src = y_src + hs2
            xMapArr[y,x] = x_src
            yMapArr[y,x] = y_src
            # xMapArr[y,x] = x_src
            # yMapArr[y,x] = y_src

    # print(type(xMapArr))
    # print(xMapArr)
    cvfs.write("xMapArr", xMapArr)
    cvfs.write("yMapArr", yMapArr)

    cvfs.release()

def DrawTransPointsFromYml():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/marked_data/"
    yml = "controlPoints_merged.yml"
    imgname = "360_0103_left_deform_test"
    cvfs = cv2.FileStorage(prefix+yml, cv2.FileStorage_READ)
    img = cv2.imread(prefix+imgname+".jpg")

    p = cvfs.getNode("p").mat().tolist()
    for i in p:
        # i[0] = i[0] + 960
        # i[1] = i[1] + 960
        (i[1], i[0]) = ForwardEquiRecProjection(i[1]+960, i[0]+960)
        cv2.circle(img, (int(i[1]), int(i[0])), 2, (0, 255, 0), -1)
    cv2.namedWindow("img", WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey()

    cvfs.release()

def FirstStepCalib(prefix, imgname, isDraw = False, linewidth=1):
    # linewidth = 1
    # prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/"
    # imgname = "360_0103"
    img_raw = cv2.imread(prefix+imgname+".jpg")

    # isDraw = False
    # isRotate = False
    
    img_l = img_raw[:, :img_raw.shape[1]//2, :]
    img_r = img_raw[:, img_raw.shape[1]//2:, :]
    C = (957, 950)
    W = 1914
    H = 1900
    R195 = 1009
    R180 = 914
    R165 = R180- (R195 - R180)
    ROUTER = R180+(W//2-R180)
    RINNER = R180 - (ROUTER - R180)

    img_l = img_l[18:1920-2, 2:1920-4, :] # (y,x,channel)      # cut extra pixels
    if isDraw:
        # draw valied area of fisheye with green
        cv2.circle(img_l, C, R195, (0, 255, 0), linewidth)
        # draw 180 degree area of fisheye with read
        cv2.circle(img_l, C, R180, (0, 0, 255), linewidth)
        # draw overlap area
        cv2.circle(img_l, C, R165, (255, 255, 0), linewidth)
        # draw acutally fuse area
        cv2.circle(img_l, C, ROUTER, (255, 255, 255), linewidth)    # out fuse circle
        cv2.circle(img_l, C, RINNER, (255, 255, 255), linewidth)     # inner fuse circle
        # draw line in mid x and y
        cv2.line(img_l, (957,0), (957,H), (255,0,0), linewidth)
        cv2.line(img_l, (0,950), (W,950), (255,0,0), linewidth)
        cv2.imwrite(prefix+imgname[4:]+"_left_calib.jpg", img_l)
    else:
        cv2.imwrite(prefix+imgname[4:]+"_left.jpg", img_l)

    img_r = img_r[20:1920, 0:1920-6, :]      # cut extra pixels

    # first cut to make the fish eye center in the center of the img, then rotate
    # if isRotate:
    #     # rotate right img to calib
    #     Rotate_M = cv2.getRotationMatrix2D(C, -0.6, 1)
    #     img_r = cv2.warpAffine(img_r, Rotate_M, (W,H))
    
    if isDraw:
        # draw valied area of fisheye with green
        cv2.circle(img_r, C, R195, (0, 255, 0), linewidth)
        # draw 180 degree area of fisheye with read
        cv2.circle(img_r, C, R180, (0, 0, 255), linewidth)
        # draw overlap area
        cv2.circle(img_r, C, R165, (255, 255, 0), linewidth)
        # draw acutally fuse area
        cv2.circle(img_r, C, ROUTER, (255, 255, 255), linewidth)    # out fuse circle
        cv2.circle(img_r, C, RINNER, (255, 255, 255), linewidth)     # inner fuse circle
        # draw line in mid x and y
        cv2.line(img_r, (957,0), (957,H), (255,0,0), linewidth)
        cv2.line(img_r, (0,950), (W,950), (255,0,0), linewidth)
        # cv2.imwrite(prefix+imgname[4:]+"_right_rotate_calib.jpg", img_r)
        cv2.imwrite(prefix+imgname[4:]+"_right_calib.jpg", img_r)
    else:
        # cv2.imwrite(prefix+imgname[4:]+"_right_rotate.jpg", img_r)
        cv2.imwrite(prefix+imgname[4:]+"_right.jpg", img_r)

def MergeTest():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test3/"
    imgname1 = "0104_left_deform.jpg"
    imgname2 = "0104_right_deform_mls.jpg"
    img1 = cv2.imread(prefix+imgname1)
    img2 = cv2.imread(prefix+imgname2)

    # img1[:, :1896//2] = img2[:, :1896//2]
    # img1[:, 1896+1900//2:] = img2[:, 1896+1900//2:]

    # cv2.imwrite(prefix+"res.jpg", img1)

    H = 1900
    W = 3796
    overlap = 30
    mask1 = np.concatenate((
        np.zeros((H, 1896//2 - overlap)), 
        np.ones((H, 1896//2 + 2*overlap + 1900//2))*255, 
        np.zeros((H, 1900//2-overlap))), axis=1).astype('uint8')

    mask2 = np.concatenate(
        (np.ones((H, 1896//2 + overlap))*255, 
        np.zeros((H, 1896//2 - 2*overlap + 1900//2)), 
        np.ones((H, 1900//2+overlap))*255), axis=1).astype('uint8')
    
    cv2.imwrite(prefix+"mask1.jpg", mask1)
    cv2.imwrite(prefix+"mask2.jpg", mask2)

    blender = cv2.detail.Blender_createDefault(cv2.detail.BLENDER_MULTI_BAND, try_gpu=False)
    blender.prepare((0,0,W,H))
    blender.feed(img1.astype(np.int16), mask1, (0,0))       # convert to CV_16SC3
    blender.feed(img2.astype(np.int16), mask2, (0,0))

    res = (np.zeros((H,W,3))).astype('uint8')
    res_mask = (np.ones((H,W))*255).astype('uint8')

    out, out_mask = blender.blend(res,res_mask)

    cv2.imwrite(prefix+'out_blended.jpg',out.astype('uint8'))
    cv2.imwrite(prefix+"outmask.jpg", out_mask)

def genRingMask():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_2/"
    imgname1 = "0103_left"
    imgname2 = "0103_right"
    img1 = cv2.imread(prefix+imgname1+".jpg")
    img2 = cv2.imread(prefix+imgname2+".jpg")
    C = (957, 950)
    W = 1914
    H = 1900
    R195 = 1009
    R180 = 914
    R165 = R180- (R195 - R180)
    ROUTER = R180+(W//2-R180)
    RINNER = R180 - (ROUTER - R180)
    inner_ring = np.zeros_like(img1)
    cv2.circle(inner_ring, C, RINNER, (255,255,255), -1)
    outer_ring = np.zeros_like(img1)
    cv2.circle(outer_ring, C, ROUTER, (255,255,255), -1)
    ring_mask = np.zeros_like(img1)
    cv2.bitwise_xor(inner_ring, outer_ring, ring_mask)
    cv2.imwrite(prefix+"ring_mask.jpg", ring_mask)
    cv2.bitwise_and(img1, ring_mask, img1)
    cv2.bitwise_and(img2, ring_mask, img2)
    cv2.imwrite(prefix+imgname1+"_filter.jpg", img1)
    cv2.imwrite(prefix+imgname2+"_filter.jpg", img2)

def CalcbiasFromPointsAndExtend():
    # prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/marked_data_0103/"
    # yml = "raw_controlPoints.yml"
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test4_mark_points/marked_data_0104/"
    yml = "controlPoints_merged.yml"
    cvfs = cv2.FileStorage(prefix+yml, cv2.FileStorage_READ)

    p = cvfs.getNode("p").mat().tolist()
    q = cvfs.getNode("q").mat().tolist()
    bias_xs = []
    bias_ys = []
    for i in range(len(p)):
        bias_ys.append(q[i][0] - p[i][0])
        bias_xs.append(q[i][1] - p[i][1])
    
    bias_x = sum(bias_xs)/len(bias_xs)
    # bias_x = max(bias_xs)
    bias_y = sum(bias_ys[:4])/4
    if bias_x < 0:
        bias_x = int(bias_x - 0.5)
    else:
        bias_x = int(bias_x + 0.5)
    if bias_y < 0:
        bias_y = int(bias_y - 0.5)
    else:
        bias_y = int(bias_y + 0.5)
    print(bias_ys[:4])
    print("bias_x: ", bias_x)
    print("bias_y: ", bias_y)
    cvfs2 = cv2.FileStorage(prefix+"biasFix_extend.yml", cv2.FileStorage_WRITE)
    for i in range(len(p)):     # p --> q
        p[i][1] = p[i][1] + bias_x
        p[i][0] = p[i][0] + 950   # extend to full img
        # add bias to y
        p[i][0] = p[i][0] + bias_y
        # 对称翻转
        y1 = 950 - (p[i][0] - 950)     
        # for each points add an extra points
        p.append([y1, p[i][1]])      #(y,x)
    
    for i in range(len(q)):
        q[i][0] = q[i][0] + 950   # extend to full img
        y1 = 950 - (q[i][0] - 950)     # 对称翻转
        # for each points add an extra points
        q.append([y1, q[i][1]])      #(y,x)
        
    cvfs2.write("p", np.array(p))
    cvfs2.write("q", np.array(q))

    cvfs.release()
    cvfs2.release()

def genUndistortMapWithBias():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test5_undistort_add_bias/"
    raw_ymlname_l = "equirectangular_left"
    raw_ymlname_r = "equirectangular"
    imgname="360_0103"
    FirstStepCalib(prefix, imgname)

    cvfs_l = cv2.FileStorage(prefix+raw_ymlname_l+".yml", cv2.FileStorage_READ)
    cvfs_r = cv2.FileStorage(prefix+raw_ymlname_r+".yml", cv2.FileStorage_READ)
    cvfs_l_new = cv2.FileStorage(prefix+raw_ymlname_l+"_bias.yml", cv2.FileStorage_WRITE)
    cvfs_r_new = cv2.FileStorage(prefix+raw_ymlname_r+"_bias.yml", cv2.FileStorage_WRITE)

    xMapArr = cvfs_r.getNode("xMapArr").mat()
    yMapArr = cvfs_r.getNode("yMapArr").mat()
    r_bias_x = -11
    r_bias_y = -6
    l_bias_x = 18
    l_bias_y = 15

    W = xMapArr.shape[1]
    H = xMapArr.shape[0]
    W2 = W//2
    xMapArr[l_bias_y:, l_bias_x:W2] = xMapArr[:H-l_bias_y, :W2-l_bias_x]
    xMapArr[:H+r_bias_y, W2:W+r_bias_x] = xMapArr[-r_bias_y:, W2-r_bias_x:]
    # xMapArr = xMapArr[:, l_bias_x:W+r_bias_x]
    xMapArr = xMapArr[:, l_bias_x:W+r_bias_x - 3]   # 再减3是为了让宽为4的整数倍
    cvfs_r_new.write("xMapArr", xMapArr)
    yMapArr[l_bias_y:, l_bias_x:W2] = yMapArr[:H-l_bias_y, :W2-l_bias_x]
    yMapArr[:H+r_bias_y, W2:W+r_bias_x] = yMapArr[-r_bias_y:, W2-r_bias_x:]
    yMapArr = yMapArr[:, l_bias_x:W+r_bias_x - 3]
    cvfs_r_new.write("yMapArr", yMapArr)

    cvfs_r.release()
    cvfs_r_new.release()

    xMapArr = cvfs_l.getNode("xMapArr").mat()
    yMapArr = cvfs_l.getNode("yMapArr").mat()
    xMapArr = xMapArr[:, l_bias_x:W+r_bias_x - 3]  # 再减3是为了让宽为4的整数倍
    yMapArr = yMapArr[:, l_bias_x:W+r_bias_x - 3]
    cvfs_l_new.write("xMapArr", xMapArr)
    cvfs_l_new.write("yMapArr", yMapArr)
    cvfs_l.release()
    cvfs_l_new.release()

def MergeMlsYmlAndCut():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test5_undistort_add_bias/mls/"
    yml_l = "res_part_l"
    yml_r = "res_part_r"
    cvfs1 = cv2.FileStorage(prefix+yml_l+".yml", cv2.FileStorage_READ)
    cvfs2 = cv2.FileStorage(prefix+yml_r+".yml", cv2.FileStorage_READ)
    cvfs_w = cv2.FileStorage(prefix+"mls.yml", cv2.FileStorage_WRITE)

    # r_bias_x = -11
    # r_bias_y = -6
    # l_bias_x = 18
    # l_bias_y = 15

    yarr1 = cvfs1.getNode("yarr").mat()
    xarr1 = cvfs1.getNode("xarr").mat()
    yarr2 = cvfs2.getNode("yarr").mat()
    xarr2 = cvfs2.getNode("xarr").mat()
    # rescale to full img
    xarr2 = xarr2 + xarr1.shape[1]
    # create full mls map 
    yarr = np.zeros((yarr1.shape[0], yarr1.shape[1]+yarr2.shape[1]), dtype=int)
    xarr = np.zeros((xarr1.shape[0], xarr1.shape[1]+xarr2.shape[1]), dtype=int)
    yarr[:, :yarr1.shape[1]] = yarr1
    yarr[:, yarr1.shape[1]:] = yarr2
    xarr[:, :xarr1.shape[1]] = xarr1
    xarr[:, xarr1.shape[1]:] = xarr2

    # # cut to fit the bias
    # yarr = yarr[:, l_bias_x:yarr.shape[1]+r_bias_x - 3] # 再减3是为了让宽为4的整数倍
    # xarr = xarr[:, l_bias_x:xarr.shape[1]+r_bias_x - 3]

    cvfs_w.write("yarr", yarr)
    cvfs_w.write("xarr", xarr)

    cvfs1.release()
    cvfs2.release()
    cvfs_w.release()

def AsymmetricCut():
    prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test5_undistort_add_bias/"
    imgname = "0104_right_deform"
    img = cv2.imread(prefix+imgname+".jpg")
    img_l = img[:, :1896]
    img_r = img[:, 1896:1896+1900]
    cv2.imwrite(prefix+imgname+"_l.jpg", img_l)
    cv2.imwrite(prefix+imgname+"_r.jpg", img_r)

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
    # rotateImg()
    # DoubleLongitude()
    # MergeYml()

    # DrawTransPointsFromYml()
    # FirstStepCalib()
    # GetUndistortMap()
    MergeTest()
    # genRingMask()

    # CalcbiasFromPointsAndExtend()
    # DrawPointsFromYml()
    # genUndistortMapWithBias()
    # MergeMlsYmlAndCut()
    # AsymmetricCut()

    # testYml()



if __name__ == '__main__':
    main()