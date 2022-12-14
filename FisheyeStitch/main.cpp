#include <QCoreApplication>

#include"fisheyestitcher.h"
#include"test/ImageStitch.h"

#include<chrono>

using namespace std;
using namespace chrono;

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

void DulFisheyeTest(){
   string prefix = "/home/cyx/programes/Pictures/gear360/lab_data/new_test_6/";
   string yml_l = "equirectangular_left_bias.yml";
   string yml_r = "equirectangular_bias_mls.yml";
   string testImgname = "0103_right";
   Mat imgIn = imread(prefix+testImgname+".jpg");
   Mat imgOut;
   Mat xMapArr, yMapArr;
   cv::FileStorage cvfs(prefix+yml_r, FileStorage::READ);
   if( cvfs.isOpened())
     {
       cvfs["xMapArr"] >> xMapArr;
       cvfs["yMapArr"] >> yMapArr;
       cvfs.release();
     }
   else
     {
       CV_Error_(cv::Error::StsBadArg,
                 ("Cannot open map file"));
     }
   for(int i = 0; i < 100; i++){
       auto begin = system_clock::now();
       cv::remap(imgIn,imgOut,xMapArr,yMapArr,cv::INTER_LINEAR);
       auto end = system_clock::now();
       auto duration = duration_cast<microseconds>(end - begin);
       cout << "one frame total spends "
            << double(duration.count()) * microseconds::period::num / microseconds::period::den
            << "seconds" << endl;
     }


   cv::imwrite(prefix+testImgname+"_mlsDeform.jpg", imgOut);
   printf("defrom done!\n");

}

int main(int argc, char *argv[])
{
  //      Show();
  //      test();
  //  cutUnwarpedTest();
  //    stitcherTest();
  //  fullStitchTest();
  //  record();
  DulFisheyeTest();
}
