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
  Rect finalrc = Rect(0,0,0,0);
  Eigen::ArrayXXd u;
  Eigen::ArrayXXd v;
  Mat cutted;
  Mat imgMask;
  Mat imgMask2, imgMask3;
  Mat xMapArray, yMapArray;
  Mat unwarpImg;

  void CalcRcCutFunc(Mat& in);
public:
  Undistort();
  void cutFisheye(Mat& in, Mat& out);
  void unDisFishEyeTest(Mat& in, Mat& out);
  void InitMartix(Mat& in, int Threshold = 20);
  void MatrixUndistort(Mat& raw, Mat& dst);
};

#endif // UNDISTORT_H
