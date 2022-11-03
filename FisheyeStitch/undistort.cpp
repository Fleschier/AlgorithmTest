#include "undistort.h"
#include <chrono>
#include <QDebug>

using namespace chrono;
using namespace Eigen;

#define BILINEAR 0
#define OPENCV__ 1
#define NEAREST !BILINEAR & !OPENCV__

#define USEMASK 1
#define DEBUGSHOW 0

#define MANUAL 0

// click count
int g_cClicks = 0;
bool Manual = false;

// mouse click capture
void CallBackFunc( int event, int x, int y, int flags, void* userdata )
{
	if  ( event == EVENT_LBUTTONDOWN )
	{
		vector<Point> *aPoints = (vector<Point> *)userdata;
		Point point;
		point.x = x;
		point.y = y;
		(*aPoints).push_back( point );
#if DEBUGSHOW
		cout << "Point No. " << g_cClicks << " - position (" << x << ", " << y << ")" << endl;
#endif
		g_cClicks++;
	}
	//else if  ( event == EVENT_RBUTTONDOWN )
	//{
	//     cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	//}
	//else if  ( event == EVENT_MBUTTONDOWN )
	//{
	//     cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	//}
	//else if ( event == EVENT_MOUSEMOVE )
	//{
	//     cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	//}
}

Point PointMean( const vector<Point> &aPoints )
{
	Point mean;
	for( unsigned int i = 0; i < aPoints.size(); i++ )
	{
		mean.x += aPoints[i].x;
		mean.y += aPoints[i].y;
	}
	mean.x /= aPoints.size();
	mean.y /= aPoints.size();

	return mean;
}


Undistort::Undistort(){
  _cut_border = 20;
}

void Undistort::cutFisheye(Mat &in, Mat &out){

  _imgMask.create(in.rows, in.cols, CV_8UC3);
  _imgMask.setTo(0);
//  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 710, CV_RGB(255,255,255), -1);
//  cv::circle(imgMask, Point(in.cols/2 + 15, in.rows/2 + 64), 710, CV_RGB(255,255,255), -1);
  cv::circle(_imgMask, Point(in.cols/2 - 30, in.rows/2 + 35), 710, CV_RGB(255,255,255), -1);
  // draw a ring
//  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 410, CV_RGB(0,0,0), -1);
  imshow("mask", _imgMask);
  cv::waitKey();
  bitwise_and(in, _imgMask, in);

  Mat gray;
  cvtColor(in, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,15,255,THRESH_BINARY);

  imshow("thresh", thresh);
//  waitKey();

  vector<vector<Point>> contours;
  findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//  for(auto i : contours){
//      cout << i << endl;
//    }

//  printf("size: %d\n", contours.size());

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
//  r = MAX(rc.width / 2, rc.height / 2);
  r = 1600 / 2;
  printf("rc: x:%d, y:%d, width:%d, height:%d\n", rc.x, rc.y, rc.width, rc.height);

//  in(Rect(507,255,1454,1454)).copyTo(out);
//  in(Rect(512,260,1440,1440)).copyTo(out);
  in(rc).copyTo(out);

}

void Undistort::unDisFishEyeTest(Mat &in, Mat &out){

//  auto start = system_clock::now();

  int R = 2*r;
  double pi = 3.141592653589793;
  out.create(Size(R,R), CV_8UC3);
  out.setTo(0);

  int src_h, src_w;
  src_h = in.rows;
  src_w = in.cols;

  // circle heart
  int x0 = src_w / 2;
  int y0 = src_h / 2;

  // use array to complete coefficient-wise operation**************** important!

  Eigen::ArrayXXd range_arr(1,R);
  for(int i = 0; i < R; i++){
      range_arr(0, i) = i;
    }
  cout << "range_arr:\n" << range_arr.rows() <<" " << range_arr.cols() << endl;

  Eigen::ArrayXXd range_arr_T = range_arr.transpose();
  cout << "range_arr_T:\n" << range_arr_T.rows() <<" " << range_arr_T.cols() << endl;

//  Eigen::MatrixXd theta = Eigen::MatrixXd::Constant(R, 1, pi);
//  theta -= (pi / R) * range_arr_T;
  Eigen::ArrayXXd theta = pi - (pi/R) * range_arr_T;
  cout << "theta:\n" << theta.rows() <<" " << theta.cols() << endl;

//  Eigen::MatrixXd temp_theta(theta);
//  for(int i = 0; i < R; i++){
//      temp_theta(i, 0) = pow(tan(theta(i,0)), 2);
//    }
  Eigen::ArrayXXd temp_theta = pow(tan(theta), 2);
  cout << "temp_theta:\n" << temp_theta.rows() <<" " << temp_theta.cols() << endl;

//  Eigen::MatrixXd phi = Eigen::MatrixXd::Constant(1, R, pi);
//  phi -= (pi/R)*range_arr;
  Eigen::ArrayXXd phi = pi - (pi/R)*range_arr;
  cout << "phi:\n" << phi.rows() <<" " << phi.cols() << endl;

//  Eigen::MatrixXd temp_phi(phi);
//  for(int i = 0; i < R; i++){
//      temp_phi(0 ,i) = pow(tan(phi(0,i)), 2);
//    }
  Eigen::ArrayXXd temp_phi = pow(tan(phi),2);
  cout << "temp_phi:\n" << temp_phi.rows() <<" " << temp_phi.cols() << endl;


  Eigen::ArrayXXd square_temp_phi = (Eigen::MatrixXd::Ones(R,1) * temp_phi.matrix()).array();
  cout << "square_temp_phi:\n" << square_temp_phi.rows() <<" " << square_temp_phi.cols() << endl;
  Eigen::ArrayXXd square_temp_theta = (temp_theta.matrix() * Eigen::MatrixXd::Ones(1,R)).array();
  cout << "square_temp_theta:\n" << square_temp_theta.rows() <<" " << square_temp_theta.cols() << endl;

  Eigen::ArrayXXd temp_u = r/sqrt((square_temp_phi + 1 + square_temp_phi/square_temp_theta));
  cout << "temp_u:\n" << temp_u.rows() <<" " << temp_u.cols() << endl;
  Eigen::ArrayXXd temp_v = r/sqrt((square_temp_theta + 1 + square_temp_theta/square_temp_phi));
  cout << "temp_v:\n" << temp_v.rows() <<" " << temp_v.cols() << endl;

  Eigen::ArrayXXd flag = Eigen::ArrayXXd::Ones(1, R);
  for(int i = 0; i <= r; i++){
      flag(0,i) = -1;
    }
//  cout << flag <<endl;
  flag = (Eigen::MatrixXd::Ones(R,1) * flag.matrix()).array();

  Eigen::ArrayXXd u = x0 + temp_u * flag + 0.5;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  Eigen::ArrayXXd v = y0 + temp_v * flag.transpose() + 0.5;
  cout << "v: " << v.rows() << " " << v.cols() << endl;



#if NEAREST
  u.cast<int>();
  v.cast<int>();
#endif

  auto start = system_clock::now();

#if OPENCV__
//  Mat _xMapArray(u.cols(), u.rows(), CV_32FC1), _yMapArray(v.cols(), v.rows(), CV_32FC1);
  _xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
  _yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
  for(int i = 0; i < u.rows(); i++){
      for(int j = 0; j < u.cols(); j++){
          _xMapArray.at<float>(i,j)=u(i,j);
          _yMapArray.at<float>(i,j)=v(i,j);
        }
    }
  remap(in, out, _xMapArray, _yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
#endif

#if NEAREST
  for(int i = 0; i < out.rows; i++){
//      uchar* ptr_in = in.ptr<uchar>(i);
      uchar* ptr_out = out.ptr<uchar>(i);
      for(int j = 0; j < out.cols; j++){
          // in.at<uchar>(row,col)  //gray
          // in.at<Vec3b>(row,col)[i] //color
          if(u(i,j) <= in.cols-1 && u(i,j) >= 0 && v(i,j) <= in.rows-1 && v(i,j) >= 0){
              ptr_out[j*3] = in.at<Vec3b>(v(i,j), u(i,j))[0];
              ptr_out[j*3+1] = in.at<Vec3b>(v(i,j), u(i,j))[1];
              ptr_out[j*3+2] = in.at<Vec3b>(v(i,j), u(i,j))[2];
            }

        }
    }
#endif
#if BILINEAR
  double s1, s2, s3, s4;
//  printf("%.6f\n", u(1,1));
  for(int i = 0; i < out.rows; i++){
      uchar* ptr_out = out.ptr<uchar>(i);
      for(int j = 0; j < out.cols; j++){
          if(u(i,j) <= in.cols-1 && u(i,j) >= 0 && v(i,j) <= in.rows-1 && v(i,j) >= 0){
              int x0 = u(i,j);
              int y0 = v(i,j);
//              if(in.at<Vec3b>(y0, x0)[0] <= 10 && in.at<Vec3b>(y0, x0)[1] <= 10 && in.at<Vec3b>(y0, x0)[2] <= 10){
//                  ptr_out[j*3] = in.at<Vec3b>(y0, x0)[0];
//                  ptr_out[j*3+1] = in.at<Vec3b>(y0, x0)[1];
//                  ptr_out[j*3+2] = in.at<Vec3b>(y0, x0)[2];
//                  continue;
//                }
              int x1 = (u(i,j)+1 <= in.cols-1) ? u(i,j)+1 : in.cols-1;
              int y1 = (v(i,j)+1 <= in.rows-1) ? v(i,j)+1 : in.rows-1;
              s1 = (u(i,j) - x0) * (v(i,j) - y0);
              s2 = (x1 - u(i,j)) * (v(i,j) - y0);
              s3 = (u(i,j) - x0) * (y1 - v(i,j));
              s4 = (x1 - u(i,j)) * (y1 - v(i,j));
//              printf("s1=%.6f s2=%.6f s3=%.6f s4=%.6f", s1, s2, s3, s4);

              ptr_out[j*3] = in.at<Vec3b>(y0, x0)[0]*s4 + in.at<Vec3b>(y0, x1)[0]*s3 +
                  in.at<Vec3b>(y1, x0)[0]*s2 + in.at<Vec3b>(y1, x1)[0]*s1;
              ptr_out[j*3+1] = in.at<Vec3b>(y0, x0)[1]*s4 + in.at<Vec3b>(y0, x1)[1]*s3 +
                  in.at<Vec3b>(y1, x0)[1]*s2 + in.at<Vec3b>(y1, x1)[1]*s1;
              ptr_out[j*3+2] = in.at<Vec3b>(y0, x0)[2]*s4 + in.at<Vec3b>(y0, x1)[2]*s3 +
                  in.at<Vec3b>(y1, x0)[2]*s2 + in.at<Vec3b>(y1, x1)[2]*s1;
            }

        }
    }
#endif
//  Eigen::ArrayXXd test1(3,3), test2(3,3);
//  test1 << 1,2,3,
//            4,5,6,
//            7,8,9;
//  test2 << 1,2,3,
//      4,5,6,
//      7,8,10;
//  cout << pow(tan(test1/test2),2) << endl;

//  Eigen::MatrixXd test1(1,5), test2(5,1);
//  test1 << 1,2,3,4,5;
//  test2 << 10,
//          11,
//          12,
//          13,
//          14;
//  test1 = MatrixXd::Ones(5,1) * test1;
//  test2 = test2 * MatrixXd::Ones(1,5);
////  cout << MatrixXd::Ones(5,1) * test1 << endl;
////  cout << test2 * MatrixXd::Ones(1,5) << endl;
//  cout << "test2/test1: \n" << test2.array()/test1.array() << endl;
//  cout << "test1 + test2/test1: \n" << test1.array()+test2.array()/test1.array() << endl;

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "undistort spends: "
             << double(duration.count()) * microseconds::period::num / microseconds::period::den
             << "seconds" << endl;

}

void Undistort::_CalcRcCutFunc(Mat &in, int idx, int threshold_){
  Mat gray;
  cvtColor(in, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,threshold_,255,THRESH_BINARY);

#if DEBUGSHOW
  imshow("thresh", thresh);
  waitKey();
#endif
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

  if(idx < 0){
      _finalrc = boundingRect(MaxContour);
      printf("finalrc: x:%d, y:%d, width:%d, height:%d\n", _finalrc.x, _finalrc.y, _finalrc.width, _finalrc.height);

  }
  else{
      _rcs[idx] = boundingRect(MaxContour);
      printf("rcs[%d]: x:%d, y:%d, width:%d, height:%d\n",idx, _rcs[idx].x, _rcs[idx].y, _rcs[idx].width, _rcs[idx].height);

  }

}

// deprecated
//void Undistort::InitMartix(Mat& in, int Threshold){

//#if USEMASK
//  _imgMask.create(in.rows, in.cols, CV_8UC3);
//  _imgMask.setTo(0);
////  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 710, CV_RGB(255,255,255), -1);
////  cv::circle(imgMask, Point(in.cols/2 + 15, in.rows/2 + 64), 710, CV_RGB(255,255,255), -1);
//  cv::circle(_imgMask, Point(in.cols/2 - 30, in.rows/2 + 35), 710, CV_RGB(255,255,255), -1);
//  bitwise_and(in, _imgMask, in);
//#endif

//  Mat gray;
//  cvtColor(in, gray, COLOR_RGB2GRAY);

//  Mat thresh;
//  threshold(gray, thresh,Threshold,255,THRESH_BINARY);

//  vector<vector<Point>> contours;
//  findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//  // find the largest area of contour
//  int maxArea = -1;
//  vector<Point> MaxContour;
//  for(int i = 0; i < contours.size(); i++){
//      int area = contourArea(contours[i]);
//      if(area > maxArea){
//          MaxContour = contours[i];
//          maxArea = area;
//        }
//    }

//  _rc = boundingRect(MaxContour);
////  _rc = Rect(507,255,1454,1454);
////  _rc = Rect(512,260,1440,1440);
////  r = MAX(_rc.width / 2, _rc.height / 2) + 50;
//  r = 1600/2;

//  int R = 2*r;
//  _cutted = Mat::zeros(Size(_rc.width,_rc.height), CV_8UC3);

//  //  double pi = 3.141592653589793;
//  double pi = CV_PI;

//  int src_h, src_w;
//  src_h = _cutted.rows;
//  src_w = _cutted.cols;
//  printf("src_h: %d , src_w: %d\n", src_h, src_w);

//  // circle heart
//  int x0 = src_w / 2;
//  int y0 = src_h / 2;
//  printf("x0: %d, y0: %d\n", x0, y0);

//  // use array to complete coefficient-wise operation**************** important!

//  Eigen::ArrayXXd range_arr(1,R);
//  for(int i = 0; i < R; i++){
//      range_arr(0, i) = i;
//    }
//  cout << "range_arr:\n" << range_arr.rows() <<" " << range_arr.cols() << endl;

//  Eigen::ArrayXXd range_arr_T = range_arr.transpose();
//  cout << "range_arr_T:\n" << range_arr_T.rows() <<" " << range_arr_T.cols() << endl;

////  Eigen::MatrixXd theta = Eigen::MatrixXd::Constant(R, 1, pi);
////  theta -= (pi / R) * range_arr_T;
//  Eigen::ArrayXXd theta = pi - (pi/R) * range_arr_T;
//  cout << "theta:\n" << theta.rows() <<" " << theta.cols() << endl;

////  Eigen::MatrixXd temp_theta(theta);
////  for(int i = 0; i < R; i++){
////      temp_theta(i, 0) = pow(tan(theta(i,0)), 2);
////    }
//  Eigen::ArrayXXd temp_theta = pow(tan(theta), 2);
//  cout << "temp_theta:\n" << temp_theta.rows() <<" " << temp_theta.cols() << endl;

////  Eigen::MatrixXd phi = Eigen::MatrixXd::Constant(1, R, pi);
////  phi -= (pi/R)*range_arr;
//  Eigen::ArrayXXd phi = pi - (pi/R)*range_arr;
//  cout << "phi:\n" << phi.rows() <<" " << phi.cols() << endl;

////  Eigen::MatrixXd temp_phi(phi);
////  for(int i = 0; i < R; i++){
////      temp_phi(0 ,i) = pow(tan(phi(0,i)), 2);
////    }
//  Eigen::ArrayXXd temp_phi = pow(tan(phi),2);
//  cout << "temp_phi:\n" << temp_phi.rows() <<" " << temp_phi.cols() << endl;


//  Eigen::ArrayXXd square_temp_phi = (Eigen::MatrixXd::Ones(R,1) * temp_phi.matrix()).array();
//  cout << "square_temp_phi:\n" << square_temp_phi.rows() <<" " << square_temp_phi.cols() << endl;
//  Eigen::ArrayXXd square_temp_theta = (temp_theta.matrix() * Eigen::MatrixXd::Ones(1,R)).array();
//  cout << "square_temp_theta:\n" << square_temp_theta.rows() <<" " << square_temp_theta.cols() << endl;

//  Eigen::ArrayXXd temp_u = r/sqrt((square_temp_phi + 1 + square_temp_phi/square_temp_theta));
//  cout << "temp_u:\n" << temp_u.rows() <<" " << temp_u.cols() << endl;
//  Eigen::ArrayXXd temp_v = r/sqrt((square_temp_theta + 1 + square_temp_theta/square_temp_phi));
//  cout << "temp_v:\n" << temp_v.rows() <<" " << temp_v.cols() << endl;

//  Eigen::ArrayXXd flag = Eigen::ArrayXXd::Ones(1, R);
//  for(int i = 0; i <= r; i++){
//      flag(0,i) = -1;
//    }
////  cout << flag <<endl;
//  flag = (Eigen::MatrixXd::Ones(R,1) * flag.matrix()).array();

//#if NEAREST
//  u = x0 + temp_u * flag + 0.5;
//  cout << "u: " << u.rows() << " " << u.cols() << endl;
//  v = y0 + temp_v * flag.transpose() + 0.5;
//  cout << "v: " << v.rows() << " " << v.cols() << endl;
//  u.cast<int>();
//  v.cast<int>();
//#endif

//#if BILINEAR
//  u = x0 + temp_u * flag;
//  cout << "u: " << u.rows() << " " << u.cols() << endl;
//  v = y0 + temp_v * flag.transpose();
//  cout << "v: " << v.rows() << " " << v.cols() << endl;
//#endif

//#if OPENCV__
//  u = x0 + temp_u * flag;
//  cout << "u: " << u.rows() << " " << u.cols() << endl;
//  v = y0 + temp_v * flag.transpose();
//  cout << "v: " << v.rows() << " " << v.cols() << endl;
////  Mat _xMapArray(u.cols(), u.rows(), CV_32FC1), _yMapArray(v.cols(), v.rows(), CV_32FC1);
//  _xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
//  _yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
//  for(int i = 0; i < u.rows(); i++){
//      for(int j = 0; j < u.cols(); j++){
//          _xMapArray.at<float>(i,j)=u(i,j);
//          _yMapArray.at<float>(i,j)=v(i,j);
//        }
//    }
//#endif

//  if(_finalrc.width == 0){
//      // init final rc
//      Mat dst;
//#if USEMASK
//      bitwise_and(in, _imgMask, in);
//#endif
//      in(_rc).copyTo(_cutted);
//      remap(_cutted, dst, _xMapArray, _yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
//            cv::Scalar(0, 0, 0));
//      _CalcRcCutFunc(dst, -1, Threshold);
//    }

//  // init unwarpImg
//  _unwarpImg.create(Size(2*r,2*r), CV_8UC3);
//  _unwarpImg.setTo(0);
//}

bool Undistort::InitMartix(Mat& in1,Mat& in2,Mat& in3,int radius, int Threshold, UndisMethod method){
#if USEMASK
  _imgMask.create(in1.rows, in1.cols, CV_8UC3);
  _imgMask.setTo(0);
  cv::circle(_imgMask, Point(in1.cols/2 - 30, in1.rows/2 + 30), 710, CV_RGB(255,255,255), -1);
  _imgMask.copyTo(_masks[0]);
  _imgMask.setTo(0);
  cv::circle(_imgMask, Point(in1.cols/2 + 15, in1.rows/2 + 64), 710, CV_RGB(255,255,255), -1);
  _imgMask.copyTo(_masks[1]);
  _imgMask.setTo(0);
  cv::circle(_imgMask, Point(in1.cols/2 - 30, in1.rows/2 + 35), 710, CV_RGB(255,255,255), -1);
  _imgMask.copyTo(_masks[2]);
  bitwise_and(in1, _masks[0], in1);
  bitwise_and(in2, _masks[1], in2);
  bitwise_and(in3, _masks[2], in3);
#if DEBUGSHOW
  imshow("mask1", _masks[0]);
  imshow("mask2", _masks[1]);
  imshow("mask3", _masks[2]);
#endif
#endif

  // calc rect info
  this->_CalcRcCutFunc(in1, 0, Threshold);
  this->_CalcRcCutFunc(in2, 1, Threshold);
  this->_CalcRcCutFunc(in3, 2, Threshold);

  _cutted = Mat::zeros(Size(_rcs[0].width,_rcs[0].height), CV_8UC3);

  _method = method;
  printf("method: %d ", _method);
  vector<string> methodname = {"DOUBLELONGITUDE", "HEMICYLINDER", "MIDPOINTCIRCLE"};
  cout << methodname.at((int)_method) << endl;
  if(DOUBLELONGGITUDE == _method){
      r = radius;
      int R = 2*r;
      double pi = CV_PI;

      int src_h, src_w;
      src_h = _cutted.rows;
      src_w = _cutted.cols;
      printf("src_h: %d , src_w: %d\n", src_h, src_w);

      // circle heart
      int x0 = src_w / 2;
      int y0 = src_h / 2;
      printf("x0: %d, y0: %d\n", x0, y0);

      Eigen::ArrayXXd range_arr(1,R);
      for(int i = 0; i < R; i++){
          range_arr(0, i) = i;
        }
      cout << "range_arr:\n" << range_arr.rows() <<" " << range_arr.cols() << endl;

      Eigen::ArrayXXd range_arr_T = range_arr.transpose();
      cout << "range_arr_T:\n" << range_arr_T.rows() <<" " << range_arr_T.cols() << endl;

      Eigen::ArrayXXd theta = pi - (pi/R) * range_arr_T;
      cout << "theta:\n" << theta.rows() <<" " << theta.cols() << endl;

      Eigen::ArrayXXd temp_theta = pow(tan(theta), 2);
      cout << "temp_theta:\n" << temp_theta.rows() <<" " << temp_theta.cols() << endl;

      Eigen::ArrayXXd phi = pi - (pi/R)*range_arr;
      cout << "phi:\n" << phi.rows() <<" " << phi.cols() << endl;

      Eigen::ArrayXXd temp_phi = pow(tan(phi),2);
      cout << "temp_phi:\n" << temp_phi.rows() <<" " << temp_phi.cols() << endl;

      Eigen::ArrayXXd square_temp_phi = (Eigen::MatrixXd::Ones(R,1) * temp_phi.matrix()).array();
      cout << "square_temp_phi:\n" << square_temp_phi.rows() <<" " << square_temp_phi.cols() << endl;
      Eigen::ArrayXXd square_temp_theta = (temp_theta.matrix() * Eigen::MatrixXd::Ones(1,R)).array();
      cout << "square_temp_theta:\n" << square_temp_theta.rows() <<" " << square_temp_theta.cols() << endl;

      Eigen::ArrayXXd temp_u = r/sqrt((square_temp_phi + 1 + square_temp_phi/square_temp_theta));
      cout << "temp_u:\n" << temp_u.rows() <<" " << temp_u.cols() << endl;
      Eigen::ArrayXXd temp_v = r/sqrt((square_temp_theta + 1 + square_temp_theta/square_temp_phi));
      cout << "temp_v:\n" << temp_v.rows() <<" " << temp_v.cols() << endl;

      Eigen::ArrayXXd flag = Eigen::ArrayXXd::Ones(1, R);
      for(int i = 0; i <= r; i++){
          flag(0,i) = -1;
        }
      flag = (Eigen::MatrixXd::Ones(R,1) * flag.matrix()).array();

      u = x0 + temp_u * flag;
      cout << "u: " << u.rows() << " " << u.cols() << endl;
      v = y0 + temp_v * flag.transpose();
      cout << "v: " << v.rows() << " " << v.cols() << endl;

      _xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
      _yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
      for(int i = 0; i < u.rows(); i++){
          for(int j = 0; j < u.cols(); j++){
              _xMapArray.at<float>(i,j)=u(i,j);
              _yMapArray.at<float>(i,j)=v(i,j);
            }
        }
    }
  else if(HEMICYLINDER == _method){
      int nWidth = _cutted.cols;
      int nHeight = _cutted.rows;
      int Cx, Cy;
      EDistortionModel eDistModel = EQUIDISTANT;
      // distortion parameters
      Cx = nWidth / 2;
      Cy = nHeight / 2;
      double F = (double)nWidth / CV_PI;

      // generate transformation map
      _xMapArray.create(Size(nWidth, nHeight), CV_32FC1);
      _yMapArray.create(Size(nWidth, nHeight), CV_32FC1);

      for( int v = 0; v < nHeight; v++ )
      {
              for( int u = 0; u < nWidth; u++ )
              {
                      // implement hemi-cylinder target model
                      double xt = double( u );
                      double yt = double( v - Cy );

		      double r = (double)nWidth / CV_PI;
		      double alpha = double( nWidth - xt ) / r;
		      double xp = r * cos( alpha );
		      double yp = /*((double)nWidth / (double)nHeight) **/ yt;
		      double zp = r * fabs( sin( alpha ) );

		      double rp = sqrt( xp * xp + yp * yp );
		      double theta = atan( rp / zp );

		      double x1;
		      double y1;
		      // select lens distortion model
		      switch( eDistModel )
		      {
		      case( EQUIDISTANT ):
			      x1 = F * theta * xp / rp;
			      y1 = F * theta * yp / rp;
			      break;
		      case( EQUISOLID ):
			      x1 = 2.0 * F * sin( theta / 2.0 ) * xp / rp;
			      y1 = 2.0 * F * sin( theta / 2.0 ) * yp / rp;
			      break;
		      case( NODISTORTION ):
		      default:
			      x1 = xt;
			      y1 = yt;
			      break;
		      };

		      _xMapArray.at<float>( v, u ) = (float)x1 + (float)Cx;
		      _yMapArray.at<float>( v, u ) = (float)y1 + (float)Cy;
	      }
      }
    }
  else if(MIDPOINTCIRCLE == _method){

      int nWidth = _cutted.cols;
      int nHeight = _cutted.rows;
      int Cx, Cy;

      Cx = nWidth / 2;
      Cy = nHeight / 2;
      int R = MAX(Cx, Cy);
      _xMapArray.create(Size(nWidth, nHeight), CV_32FC1);
      _yMapArray.create(Size(nWidth, nHeight), CV_32FC1);

      if(Manual){
          cout << "\nPlease mark 12 points on the boundary of the circle\n\n";

          // show input frame for marking circle
          Mat tempFrame = in1.clone();
          imshow( "Input Frame", tempFrame );

          // initialize point list
          g_cClicks = 0;
          unsigned int nNumPoints = 12;
          vector<Point> aPoints;
          vector<bool> aPointDrawn;

          //set the callback function for mouse event
          setMouseCallback( "Input Frame", CallBackFunc, &aPoints );

          // wait till all 12 points input and any button is pressed
          do
            {
              waitKey(250);

              while( aPointDrawn.size() < aPoints.size() )
                {
                  aPointDrawn.push_back( false );
                }

              for( unsigned int iPoints = 0; iPoints < aPointDrawn.size(); iPoints++ )
                {
                  if( !aPointDrawn[iPoints] )
                    {
                      // draw cross mark at selected point locations
                      int s = 5;
                      line( tempFrame, Point( aPoints[iPoints].x - s, aPoints[iPoints].y - s),
                            Point( aPoints[iPoints].x + s, aPoints[iPoints].y + s), CV_RGB( 255, 0, 0 ) );
                      line( tempFrame, Point( aPoints[iPoints].x - s, aPoints[iPoints].y + s),
                            Point( aPoints[iPoints].x + s, aPoints[iPoints].y - s), CV_RGB( 255, 0, 0 ) );
                      //circle( tempFrame, aPoints[iPoints], 3, CV_RGB( 255, 0, 0 ) );
                      aPointDrawn[iPoints] = true;
                    }
                }

              imshow( "Input Frame", tempFrame );
            }while( aPoints.size() < nNumPoints );

          // reset the mouse callback
          setMouseCallback( "Input Frame", NULL, NULL );

          // compute summations
          Point mean = PointMean( aPoints );
          int sum_xi = 0, sum_yi = 0, sum_xi_2 = 0, sum_yi_2 = 0, sum_xi_3 = 0, sum_yi_3 = 0;
          int sum_xi_yi = 0, sum_xi_yi_2 = 0, sum_xi_2_yi = 0;
          for( unsigned int iPoints = 0; iPoints < nNumPoints; iPoints++ )
            {
              int xi = aPoints[iPoints].x - mean.x;
              int yi = aPoints[iPoints].y - mean.y;

              sum_xi += xi;
              sum_yi += yi;
              sum_xi_2 += xi * xi;
              sum_yi_2 += yi * yi;
              sum_xi_3 += xi * xi * xi;
              sum_yi_3 += yi * yi * yi;
              sum_xi_yi += xi * yi;
              sum_xi_yi_2 += xi * yi * yi;
              sum_xi_2_yi += xi * xi * yi;
            }

          // frame circle fitting as least squares problem
          Mat D( 3, 3, CV_64FC1 ), E( 3, 1, CV_64FC1 ), Q( 3, 1, CV_64FC1 );

          D.at<double>(0, 0) = double(2 * sum_xi);
          D.at<double>(0, 1) = double(2 * sum_yi);
          D.at<double>(0, 2) = double(nNumPoints);
          D.at<double>(1, 0) = double(2 * sum_xi_2);
          D.at<double>(1, 1) = double(2 * sum_xi_yi);
          D.at<double>(1, 2) = double(sum_xi);
          D.at<double>(2, 0) = double(2 * sum_xi_yi);
          D.at<double>(2, 1) = double(2 * sum_yi_2);
          D.at<double>(2, 2) = double(sum_yi);

          E.at<double>(0, 0) = double(sum_xi_2 + sum_yi_2);
          E.at<double>(1, 0) = double(sum_xi_3 + sum_xi_yi_2);
          E.at<double>(2, 0) = double(sum_xi_2_yi + sum_yi_3);

          // solve the least squares
          solve( D, E, Q , DECOMP_LU );
          double A = Q.at<double>(0, 0);
          double B = Q.at<double>(1, 0);
          double C = Q.at<double>(2, 0);

          // compute parameters
          Cx = (int)A + mean.x;
          Cy = (int)B + mean.y;
          R = (int)sqrt (C + A * A + B * B);

#if DEBUGSHOW
          cout << "Cx = " << Cx << ", Cy = " << Cy << ", R = " << R << endl;

          // draw computed circle
          circle( tempFrame, Point( Cx, Cy ), R, CV_RGB( 0, 255, 0 ) );
          imshow( "Input Frame", tempFrame );
          cv::waitKey();
#endif
        }

      // generate transformation map
      for( int u = 0; u < nWidth; u++ )
        {
          for( int v = 0; v < nHeight; v++ )
            {
              float x1, y1;
              float xt = float(u - Cx);
              float yt = float(v - Cy);
              if( xt != 0 ) // non limiting case
                {
                  float AO1 = (xt * xt + float(R * R)) / (2.0f * xt);
                  float AB = sqrt( xt * xt + float(R * R) );
                  float AP = yt;
                  float PE = float(R - yt);

                  float a = AP / PE;
                  float b = 2.0f * asin( AB / (2.0f * AO1) );

                  float alpha = a * b / (a + 1.0f);
                  x1 = xt - AO1 + AO1 * cos( alpha );
                  y1 = AO1 * sin( alpha );
                }
              else // limiting case
                {
                  x1 = (float)xt;
                  y1 = (float)yt;
                }
              _xMapArray.at<float>( v, u ) = x1 + (float)Cx;
              _yMapArray.at<float>( v, u ) = y1 + (float)Cy;
            }
        }
    }
  else{
      printf("no Support function to distort!\n");
      return false;
    }

  if(_finalrc.width == 0){
      // init final rc
      if(DOUBLELONGGITUDE == _method){
          Mat dst;
          in1(_rcs[0]).copyTo(_cutted);
          remap(_cutted, dst, _xMapArray, _yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
          _CalcRcCutFunc(dst, -1, Threshold);
        }
      else if(HEMICYLINDER == _method || MIDPOINTCIRCLE == _method){
          _finalrc = Rect(0,0,_cutted.cols, _cutted.rows);
        }
      else{
          printf("Unsupported undistort method: %d\n", (int)_method);
          return false;
        }

  }

  // init unwarpImg
  for(int i = 0; i < 3; i++){
      if(DOUBLELONGGITUDE == _method){
          _unwarpImgs[i].create(Size(2*r,2*r), CV_8UC3);
        }
      else if(HEMICYLINDER == _method){
          _unwarpImgs[i].create(Size(_cutted.cols, _cutted.rows), CV_8UC3);
        }
      else if(MIDPOINTCIRCLE == _method){
          _unwarpImgs[i].create(Size(_cutted.cols, _cutted.rows), CV_8UC3);
//          if(_cpy_mask.cols == 0){
//              _cpy_mask.create(Size(_cutted.cols, _cutted.rows), CV_8UC1);
//              _cpy_mask.setTo(0);
//              _cpy_mask(Rect(_cut_border,0,_cpy_mask.cols-2*_cut_border,_cpy_mask.rows)).setTo(255);
//            }
        }
      else{
          printf("Unsupported undistort method:%d\n", (int)_method);
          return false;
        }
      _unwarpImgs[i].setTo(0);
    }

  printf("Undistort parameters all init done!\n");
  return true;
}

// deprecated
//void Undistort::MatrixUndistort(Mat &raw, Mat &dst){
//#if USEMASK
//  bitwise_and(raw, _imgMask, raw);
//#endif

//  if(_finalrc.width != 0){
//      if(dst.cols != _finalrc.width || dst.rows != _finalrc.height){
//          dst.create(Size(_finalrc.width, _finalrc.height), CV_8UC3);
//          dst.setTo(0);
//        }
//    }
//  else{
////      if(dst.cols != 2*r || dst.rows != 2*r || dst.channels() != 3){
////          dst.create(Size(2*r,2*r), CV_8UC3);
////          dst.setTo(0);
////        }
//      printf("uninitialize final rc, please check!...\n");
//      return;
//    }

//  raw(_rc).copyTo(_cutted);

//  imshow("cutted", _cutted);
//  waitKey(1);

//  remap(_cutted, _unwarpImg, _xMapArray, _yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
//        cv::Scalar(0, 0, 0));

//  _unwarpImg(_finalrc).copyTo(dst);
////  _unwarpImg.copyTo(dst);

//}

bool Undistort::MatrixUndistort(Mat &raw, Mat &dst, int idx){
#if USEMASK
  bitwise_and(raw, _masks[idx], raw);
#endif

//  if(DOUBLELONGGITUDE == _method){
//      printf("test\n");
//      if(_finalrc.width != 0){
//          if(dst.cols != _finalrc.width || dst.rows != _finalrc.height){
//              dst.create(Size(_finalrc.width, _finalrc.height), CV_8UC3);
//              dst.setTo(0);
//            }
//        }
//      else{
//          printf("uninitialize final rc, please check!\n");
//          return false;
//        }
//    }
//  else{
//      if(dst.cols != _cutted.cols){
//          dst.create(_cutted.size(), CV_8U);
//          printf("resize dst\n");
//        }
//    }


  raw(_rcs[idx]).copyTo(_cutImgs[idx]);

  remap(_cutImgs[idx], _unwarpImgs[idx], _xMapArray, _yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));

  if(DOUBLELONGGITUDE == _method){
      _unwarpImgs[idx](_finalrc).copyTo(dst);
    }
  else if(MIDPOINTCIRCLE == _method){
      _unwarpImgs[idx](Rect(_cut_border,0,_unwarpImgs[idx].cols-2*_cut_border, _unwarpImgs[idx].rows)).copyTo(dst);
//      _unwarpImgs[idx].copyTo(dst, _cpy_mask);
    }
  else if(HEMICYLINDER == _method){
      _unwarpImgs[idx].copyTo(dst);
    }
  else{
      printf("undefined undistort method!\n");
      return false;
    }


#if DEBUGSHOW
    switch (idx) {
    case 0:
        printf("0\n");
        imshow("0", dst);
        break;
    case 1:
        printf("1\n");
        imshow("1", dst);
        break;
    case 2:
        printf("2\n");
        imshow("2", dst);
        break;
    default:
        break;
    }
    cv::waitKey();
#endif
    return true;
}
