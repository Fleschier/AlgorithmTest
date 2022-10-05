#include <QCoreApplication>

#include"fisheyestitcher.h"

void Show(){
  Mat img;
  VideoCapture cap("./cam2.mp4");
  if(!cap.isOpened()){
      printf("failed to open camera!\n");
      return;
    }
  cap.set(CAP_PROP_FRAME_WIDTH, 2592);
  cap.set(CAP_PROP_FRAME_HEIGHT, 1944);
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
//  cap.set(CAP_PROP_FPS, 30);
  Undistort ud;
  for(int i = 0; i < 10; i++){
      cap.read(img);
//      imshow("test", img);
//      waitKey();
    }
  imwrite("test.jpg", img);
  ud.InitMartix(img);
  Mat dst;
  while(cap.read(img)){
      ud.MatrixUndistort(img, dst);
      imshow("result", dst);
      char c = waitKey(1);
      if(c == 27){
          break;
        }
    }
}

void test(){
  Undistort ud;
  Mat img = imread("/home/fleschier/Programs/CPP/fisheye/test_2592x1944.jpg");
  if(img.empty()){
      printf("read img failed!\n");
    }

  Mat cutted;
  ud.cutFisheye(img, cutted);
  imshow("cutted", cutted);
  imwrite("cutted.jpg", cutted);


  Mat out;
  ud.unDisFishEyeTest(cutted, out);
  imshow("out", out);

  imwrite("test_bilinear.jpg", out);
}

void cutUnwarpedTest(){
  Mat img = imread("./test_bilinear.jpg");
  if(img.empty()){
      printf("failed to read img!\n");
      return;
    }
  Mat gray;
  cvtColor(img, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,15,255,THRESH_BINARY);

  imshow("thresh", thresh);
//  waitKey();

  vector<vector<Point>> contours;
  findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  // find the largest area of contour
  int maxArea = -1;
  vector<Point> MaxContour;
  for(int i = 0; i < contours.size(); i++){
      int area = contourArea(contours[i]);
      if(area > maxArea){
          MaxContour = contours[i];
          maxArea = area;
        }
    }

  Rect rc = boundingRect(MaxContour);
  printf("rc: x:%d, y:%d, width:%d, height:%d\n", rc.x, rc.y, rc.width, rc.height);

  imshow("cutted", img(rc));
  waitKey();
}

void stitcherTest(){
    FishEyeStitcher stitcher;
    if(!stitcher.run()){
        return;
    }

    sleep(20);
    stitcher.stop();
}

int main(int argc, char *argv[])
{
    //  Show();
    //  test();
    //  cutUnwarpedTest();
    stitcherTest();
}
