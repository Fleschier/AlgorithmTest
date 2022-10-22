#ifndef UNDISTORT_H
#define UNDISTORT_H

#include<opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// undistort function select
enum UndisMethod{
  DOUBLELONGGITUDE = 0,
  HEMICYLINDER = 1,
  MIDPOINTCIRCLE = 2
};

// distortion models
enum EDistortionModel {
  NODISTORTION = -1,
  EQUIDISTANT,
  EQUISOLID
};

class Undistort
{
private:
  int r;
  Rect _rc;
  Rect _rcs[3];
  Rect _finalrc = Rect(0,0,0,0);
  Eigen::ArrayXXd u;
  Eigen::ArrayXXd v;
  Mat _cutted;
  Mat _cutImgs[3];
  Mat _imgMask;
  Mat _masks[3];
  Mat _xMapArray, _yMapArray;
  Mat _unwarpImg;
  Mat _unwarpImgs[3];

  UndisMethod _method;

  void _CalcRcCutFunc(Mat& in, int idx, int threshold_);
public:
  Undistort();
  void cutFisheye(Mat& in, Mat& out);
  void unDisFishEyeTest(Mat& in, Mat& out);
//  void InitMartix(Mat& in, int Threshold = 5);
  bool InitMartix(Mat& in1,Mat& in2,Mat& in3,int radius = 1600/2, int Threshold = 5, UndisMethod method = DOUBLELONGGITUDE);
//  void MatrixUndistort(Mat& raw, Mat& dst);
  bool MatrixUndistort(Mat& raw, Mat& dst, int idx);
};

#endif // UNDISTORT_H
