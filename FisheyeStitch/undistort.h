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
  vector<Mat> cutImgs;
  Mat imgMask;
  vector<Mat> masks;
  Mat xMapArray, yMapArray;
  Mat unwarpImg;
  vector<Mat> unwarpImgs;

  void CalcRcCutFunc(Mat& in);
public:
  Undistort();
  void cutFisheye(Mat& in, Mat& out);
  void unDisFishEyeTest(Mat& in, Mat& out);
  void InitMartix(Mat& in, int Threshold = 20);
  void InitMartix(Mat& in,int radius, int MaskIdx = 0,  int Threshold = 20);
  void MatrixUndistort(Mat& raw, Mat& dst);
  void MatrixUndistort(Mat& raw, Mat& dst, int idx);
};

#endif // UNDISTORT_H
