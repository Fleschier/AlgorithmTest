#ifndef UNDISTORT_H
#define UNDISTORT_H

#include<opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

class Undistort
{
private:
  int r;
  Rect rc;
  Rect rcs[3];
  Rect finalrc = Rect(0,0,0,0);
  Eigen::ArrayXXd u;
  Eigen::ArrayXXd v;
  Mat cutted;
  Mat cutImgs[3];
  Mat imgMask;
  Mat masks[3];
  Mat xMapArray, yMapArray;
  Mat unwarpImg;
  Mat unwarpImgs[3];

  void CalcRcCutFunc(Mat& in, int idx);
public:
  Undistort();
  void cutFisheye(Mat& in, Mat& out);
  void unDisFishEyeTest(Mat& in, Mat& out);
  void InitMartix(Mat& in, int Threshold = 20);
  void InitMartix(Mat& in1,Mat& in2,Mat& in3,int radius = 1600/2, int Threshold = 20);
  void MatrixUndistort(Mat& raw, Mat& dst);
  void MatrixUndistort(Mat& raw, Mat& dst, int idx);
};

#endif // UNDISTORT_H
