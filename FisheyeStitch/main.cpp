#include <QCoreApplication>

#include"fisheyestitcher.h"
#include"test/ImageStitch.h"

void Show(){
  Mat img;
  VideoCapture cap("./cam3.mp4");
  if(!cap.isOpened()){
      printf("failed to open camera!\n");
      return;
    }
  cap.set(CAP_PROP_FRAME_WIDTH, 2592);
  cap.set(CAP_PROP_FRAME_HEIGHT, 1944);
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
//  cap.set(CAP_PROP_FPS, 30);
  Undistort ud;
  for(int i = 0; i < 44; i++){
      cap.read(img);
//      imshow("test", img);
//      waitKey();
    }
//  imwrite("test.jpg", img);
  ud.InitMartix(img,img,img);
  Mat dst;
  while(cap.read(img)){

      ud.MatrixUndistort(img, dst,0);
      imshow("result", dst);
      {
        imwrite("./cam3.jpg", dst);
        imwrite("testimg.jpg", img);
        break;
      }

      char c = waitKey(1);
      if(c == 27){
          break;
        }
    }
}

void test(){
  Undistort ud;
  Mat img = imread("./fullcam3.jpg");
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

void record(){
  Mat img;
  VideoCapture cap(0);
  VideoWriter writer;
  writer.open("cam1.mp4", VideoWriter::fourcc('X','2','6','4'), 30, Size(1920,1080));

  if(!cap.isOpened() || !writer.isOpened()){
      printf("failed to open camera!\n");
      return;
    }
  cap.set(CAP_PROP_FRAME_WIDTH, 1920);
  cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
  cap.set(CAP_PROP_FPS, 30);

  while(cap.read(img)){
//      writer.write(img);
      imshow("test", img);

      char c = waitKey(1);
      if(c == 27){
          break;
        }
    }
}

int main(int argc, char *argv[])
{
//      Show();
//      test();
    //  cutUnwarpedTest();
    stitcherTest();
//  fullStitchTest();
//  record();
}
