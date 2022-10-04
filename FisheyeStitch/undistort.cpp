#include "undistort.h"
#include <chrono>
#include <QDebug>

using namespace chrono;
using namespace Eigen;

#define BILINEAR 0
#define OPENCV__ 1
#define NEAREST !BILINEAR & !OPENCV__
#define USEMASK 1


Undistort::Undistort(){

}

void Undistort::cutFisheye(Mat &in, Mat &out){

  imgMask.create(in.rows, in.cols, CV_8UC3);
  imgMask.setTo(0);
  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 715, CV_RGB(255,255,255), -1);
  bitwise_and(in, imgMask, in);

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
//  Mat xMapArray(u.cols(), u.rows(), CV_32FC1), yMapArray(v.cols(), v.rows(), CV_32FC1);
  xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
  yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
  for(int i = 0; i < u.rows(); i++){
      for(int j = 0; j < u.cols(); j++){
          xMapArray.at<float>(i,j)=u(i,j);
          yMapArray.at<float>(i,j)=v(i,j);
        }
    }
  remap(in, out, xMapArray, yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
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

void Undistort::CalcRcCutFunc(Mat &in){
  Mat gray;
  cvtColor(in, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,15,255,THRESH_BINARY);

//  imshow("thresh", thresh);
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

  finalrc = boundingRect(MaxContour);
  printf("rc: x:%d, y:%d, width:%d, height:%d\n", finalrc.x, finalrc.y, finalrc.width, finalrc.height);

}

void Undistort::InitMartix(Mat& in, int Threshold){

#if USEMASK
  imgMask.create(in.rows, in.cols, CV_8UC3);
  imgMask.setTo(0);
//  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 710, CV_RGB(255,255,255), -1);
//  cv::circle(imgMask, Point(in.cols/2 + 20, in.rows/2 + 62), 710, CV_RGB(255,255,255), -1);
  cv::circle(imgMask, Point(in.cols/2 + 5, in.rows/2 - 50), 710, CV_RGB(255,255,255), -1);
  bitwise_and(in, imgMask, in);
#endif

  Mat gray;
  cvtColor(in, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,Threshold,255,THRESH_BINARY);

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

  rc = boundingRect(MaxContour);
//  rc = Rect(507,255,1454,1454);
//  rc = Rect(512,260,1440,1440);
//  r = MAX(rc.width / 2, rc.height / 2) + 50;
  r = 1600/2;

  int R = 2*r;
  cutted = Mat::zeros(Size(rc.width,rc.height), CV_8UC3);

  //  double pi = 3.141592653589793;
  double pi = CV_PI;

  int src_h, src_w;
  src_h = cutted.rows;
  src_w = cutted.cols;
  printf("src_h: %d , src_w: %d\n", src_h, src_w);

  // circle heart
  int x0 = src_w / 2;
  int y0 = src_h / 2;
  printf("x0: %d, y0: %d\n", x0, y0);

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

#if NEAREST
  u = x0 + temp_u * flag + 0.5;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose() + 0.5;
  cout << "v: " << v.rows() << " " << v.cols() << endl;
  u.cast<int>();
  v.cast<int>();
#endif

#if BILINEAR
  u = x0 + temp_u * flag;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose();
  cout << "v: " << v.rows() << " " << v.cols() << endl;
#endif

#if OPENCV__
  u = x0 + temp_u * flag;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose();
  cout << "v: " << v.rows() << " " << v.cols() << endl;
//  Mat xMapArray(u.cols(), u.rows(), CV_32FC1), yMapArray(v.cols(), v.rows(), CV_32FC1);
  xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
  yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
  for(int i = 0; i < u.rows(); i++){
      for(int j = 0; j < u.cols(); j++){
          xMapArray.at<float>(i,j)=u(i,j);
          yMapArray.at<float>(i,j)=v(i,j);
        }
    }
#endif

  if(finalrc.width == 0){
      // init final rc
      Mat dst;
#if USEMASK
      bitwise_and(in, imgMask, in);
#endif
      in(rc).copyTo(cutted);
      remap(cutted, dst, xMapArray, yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0));
      CalcRcCutFunc(dst);
    }

  // init unwarpImg
  unwarpImg.create(Size(2*r,2*r), CV_8UC3);
  unwarpImg.setTo(0);
}

void Undistort::InitMartix(Mat& in,int radius, int MaskIdx, int Threshold){
#if USEMASK
  imgMask.create(in.rows, in.cols, CV_8UC3);
  imgMask.setTo(0);
  cv::circle(imgMask, Point(in.cols/2 - 30, in.rows/2 + 30), 710, CV_RGB(255,255,255), -1);
  masks.push_back(imgMask);
  imgMask.setTo(0);
  cv::circle(imgMask, Point(in.cols/2 + 20, in.rows/2 + 62), 710, CV_RGB(255,255,255), -1);
  masks.push_back(imgMask);
  imgMask.setTo(0);
  cv::circle(imgMask, Point(in.cols/2 + 5, in.rows/2 - 50), 710, CV_RGB(255,255,255), -1);
  masks.push_back(imgMask);
  bitwise_and(in, masks[MaskIdx], in);
#endif

  Mat gray;
  cvtColor(in, gray, COLOR_RGB2GRAY);

  Mat thresh;
  threshold(gray, thresh,Threshold,255,THRESH_BINARY);

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

  rc = boundingRect(MaxContour);
  r = radius;

  int R = 2*r;
  cutted = Mat::zeros(Size(rc.width,rc.height), CV_8UC3);

  //  double pi = 3.141592653589793;
  double pi = CV_PI;

  int src_h, src_w;
  src_h = cutted.rows;
  src_w = cutted.cols;
  printf("src_h: %d , src_w: %d\n", src_h, src_w);

  // circle heart
  int x0 = src_w / 2;
  int y0 = src_h / 2;
  printf("x0: %d, y0: %d\n", x0, y0);

  // use array to complete coefficient-wise operation**************** important!

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

#if NEAREST
  u = x0 + temp_u * flag + 0.5;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose() + 0.5;
  cout << "v: " << v.rows() << " " << v.cols() << endl;
  u.cast<int>();
  v.cast<int>();
#endif

#if BILINEAR
  u = x0 + temp_u * flag;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose();
  cout << "v: " << v.rows() << " " << v.cols() << endl;
#endif

#if OPENCV__
  u = x0 + temp_u * flag;
  cout << "u: " << u.rows() << " " << u.cols() << endl;
  v = y0 + temp_v * flag.transpose();
  cout << "v: " << v.rows() << " " << v.cols() << endl;

  xMapArray.create(Size(u.cols(), u.rows()), CV_32FC1);
  yMapArray.create(Size(v.cols(), v.rows()), CV_32FC1);
  for(int i = 0; i < u.rows(); i++){
      for(int j = 0; j < u.cols(); j++){
          xMapArray.at<float>(i,j)=u(i,j);
          yMapArray.at<float>(i,j)=v(i,j);
        }
    }
#endif

  if(finalrc.width == 0){
      // init final rc
      Mat dst;
      in(rc).copyTo(cutted);
      remap(cutted, dst, xMapArray, yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0));
      CalcRcCutFunc(dst);
    }

  // init unwarpImg
  unwarpImg.create(Size(2*r,2*r), CV_8UC3);
  unwarpImg.setTo(0);
  for(int i = 0; i < 3; i++){
      unwarpImgs.push_back(unwarpImg);
      cutImgs.push_back(Mat());
    }

  printf("Undistort parameters all init done!\n");
}

void Undistort::MatrixUndistort(Mat &raw, Mat &dst){
#if USEMASK
  bitwise_and(raw, imgMask, raw);
#endif

  if(finalrc.width != 0){
      if(dst.cols != finalrc.width || dst.rows != finalrc.height){
          dst.create(Size(finalrc.width, finalrc.height), CV_8UC3);
          dst.setTo(0);
        }
    }
  else{
//      if(dst.cols != 2*r || dst.rows != 2*r || dst.channels() != 3){
//          dst.create(Size(2*r,2*r), CV_8UC3);
//          dst.setTo(0);
//        }
      printf("uninitialize final rc, please check!\n");
      return;
    }

  raw(rc).copyTo(cutted);

  imshow("cutted", cutted);
  waitKey(1);

#if NEAREST
  for(int i = 0; i < dst.rows; i++){
      uchar* ptr_out = dst.ptr<uchar>(i);
      for(int j = 0; j < dst.cols; j++){
          if(u(i,j) >= 0 && u(i,j) <= cutted.cols-1 && v(i,j) >= 0 && v(i,j) <= cutted.rows-1){
              ptr_out[j*3] = cutted.at<Vec3b>(v(i,j), u(i,j))[0];
              ptr_out[j*3+1] = cutted.at<Vec3b>(v(i,j), u(i,j))[1];
              ptr_out[j*3+2] = cutted.at<Vec3b>(v(i,j), u(i,j))[2];
            }

        }
    }
#endif

#if BILINEAR
  double s1, s2, s3, s4;
  for(int i = 0; i < dst.rows; i++){
      uchar* ptr_out = dst.ptr<uchar>(i);
      for(int j = 0; j < dst.cols; j++){
          if(u(i,j) >= 0 && u(i,j) <= cutted.cols-1 && v(i,j) >= 0 && v(i,j) <= cutted.rows-1){
              int x0 = u(i,j);
              int y0 = v(i,j);
              int x1 = (u(i,j)+1 <= cutted.cols-1) ? u(i,j)+1 : cutted.cols-1;
              int y1 = (v(i,j)+1 <= cutted.rows-1) ? v(i,j)+1 : cutted.rows-1;
              s1 = (u(i,j) - x0) * (v(i,j) - y0);
              s2 = (x1 - u(i,j)) * (v(i,j) - y0);
              s3 = (u(i,j) - x0) * (y1 - v(i,j));
              s4 = (x1 - u(i,j)) * (y1 - v(i,j));

              ptr_out[j*3] = cutted.at<Vec3b>(y0, x0)[0]*s4 + cutted.at<Vec3b>(y0, x1)[0]*s3 +
                  cutted.at<Vec3b>(y1, x0)[0]*s2 + cutted.at<Vec3b>(y1, x1)[0]*s1;
              ptr_out[j*3+1] = cutted.at<Vec3b>(y0, x0)[1]*s4 + cutted.at<Vec3b>(y0, x1)[1]*s3 +
                  cutted.at<Vec3b>(y1, x0)[1]*s2 + cutted.at<Vec3b>(y1, x1)[1]*s1;
              ptr_out[j*3+2] = cutted.at<Vec3b>(y0, x0)[2]*s4 + cutted.at<Vec3b>(y0, x1)[2]*s3 +
                  cutted.at<Vec3b>(y1, x0)[2]*s2 + cutted.at<Vec3b>(y1, x1)[2]*s1;
            }

        }
    }
#endif

#if OPENCV__
  remap(cutted, unwarpImg, xMapArray, yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
#endif
  unwarpImg(finalrc).copyTo(dst);

}

void Undistort::MatrixUndistort(Mat &raw, Mat &dst, int idx){
#if USEMASK
  bitwise_and(raw, masks[idx], raw);
#endif

  if(finalrc.width != 0){
      if(dst.cols != finalrc.width || dst.rows != finalrc.height){
          dst.create(Size(finalrc.width, finalrc.height), CV_8UC3);
          dst.setTo(0);
        }
    }
  else{
      printf("uninitialize final rc, please check!\n");
      return;
    }

  raw(rc).copyTo(cutImgs[idx]);

#if NEAREST
  for(int i = 0; i < dst.rows; i++){
      uchar* ptr_out = dst.ptr<uchar>(i);
      for(int j = 0; j < dst.cols; j++){
          if(u(i,j) >= 0 && u(i,j) <= cutted.cols-1 && v(i,j) >= 0 && v(i,j) <= cutted.rows-1){
              ptr_out[j*3] = cutted.at<Vec3b>(v(i,j), u(i,j))[0];
              ptr_out[j*3+1] = cutted.at<Vec3b>(v(i,j), u(i,j))[1];
              ptr_out[j*3+2] = cutted.at<Vec3b>(v(i,j), u(i,j))[2];
            }

        }
    }
#endif

#if BILINEAR
  double s1, s2, s3, s4;
  for(int i = 0; i < dst.rows; i++){
      uchar* ptr_out = dst.ptr<uchar>(i);
      for(int j = 0; j < dst.cols; j++){
          if(u(i,j) >= 0 && u(i,j) <= cutted.cols-1 && v(i,j) >= 0 && v(i,j) <= cutted.rows-1){
              int x0 = u(i,j);
              int y0 = v(i,j);
              int x1 = (u(i,j)+1 <= cutted.cols-1) ? u(i,j)+1 : cutted.cols-1;
              int y1 = (v(i,j)+1 <= cutted.rows-1) ? v(i,j)+1 : cutted.rows-1;
              s1 = (u(i,j) - x0) * (v(i,j) - y0);
              s2 = (x1 - u(i,j)) * (v(i,j) - y0);
              s3 = (u(i,j) - x0) * (y1 - v(i,j));
              s4 = (x1 - u(i,j)) * (y1 - v(i,j));

              ptr_out[j*3] = cutted.at<Vec3b>(y0, x0)[0]*s4 + cutted.at<Vec3b>(y0, x1)[0]*s3 +
                  cutted.at<Vec3b>(y1, x0)[0]*s2 + cutted.at<Vec3b>(y1, x1)[0]*s1;
              ptr_out[j*3+1] = cutted.at<Vec3b>(y0, x0)[1]*s4 + cutted.at<Vec3b>(y0, x1)[1]*s3 +
                  cutted.at<Vec3b>(y1, x0)[1]*s2 + cutted.at<Vec3b>(y1, x1)[1]*s1;
              ptr_out[j*3+2] = cutted.at<Vec3b>(y0, x0)[2]*s4 + cutted.at<Vec3b>(y0, x1)[2]*s3 +
                  cutted.at<Vec3b>(y1, x0)[2]*s2 + cutted.at<Vec3b>(y1, x1)[2]*s1;
            }

        }
    }
#endif

#if OPENCV__
  remap(cutImgs[idx], unwarpImg, xMapArray, yMapArray, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
#endif
  unwarpImg(finalrc).copyTo(dst);
}
