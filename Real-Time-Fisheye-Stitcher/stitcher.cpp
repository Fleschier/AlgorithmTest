#include "stitcher.h"
#include <chrono>

using namespace std;
using namespace chrono;
using namespace cv;

#define DEBUGSHOW 1

#ifndef TESTGENERATE
#define TESTGENERATE 1
#endif

uint FishEyeStitcher::current_idx = 0;

FishEyeStitcher::FishEyeStitcher(){

}

bool FishEyeStitcher::Init(){
  _overlap = 40;

  rc_l = Rect(2, 18, 1920-6, 1920-20);
  rc_r = Rect(0+1920, 20, 1920-6, 1920-20);
  cv::FileStorage cvfs("./maps/equirectangular_left_bias.yml.gz", cv::FileStorage::READ);
  if(!cvfs.isOpened()){
      CV_Error_(cv::Error::StsBadArg,("Cannot open map file"));
    }
  cvfs["xMapArr"] >> _xMapArr_l;
  cvfs["yMapArr"] >> _yMapArr_l;
  cvfs.release();
  cvfs.open("./maps/equirectangular_bias_mls.yml.gz", cv::FileStorage::READ);
  if(!cvfs.isOpened()){
      CV_Error_(cv::Error::StsBadArg,("Cannot open map file"));
    }
  cvfs["xMapArr"] >> _xMapArr_r;
  cvfs["yMapArr"] >> _yMapArr_r;
  cvfs.release();
#if DEBUGSHOW
  printf("_xMapArr_l: %d x %d x %d\n", _xMapArr_l.cols, _xMapArr_l.rows, _xMapArr_l.channels());
  printf("_yMapArr_l: %d x %d x %d\n", _yMapArr_l.cols, _yMapArr_l.rows, _yMapArr_l.channels());
  printf("_xMapArr_r: %d x %d x %d\n", _xMapArr_r.cols, _xMapArr_r.rows, _xMapArr_r.channels());
  printf("_yMapArr_r: %d x %d x %d\n", _yMapArr_r.cols, _yMapArr_r.rows, _yMapArr_r.channels());
#endif
  _valid_area_mask = cv::Mat::zeros(Size(1914, 1900), CV_8UC3);
  cv::circle(_valid_area_mask, Point(957,950), 1914/2, CV_RGB(255,255,255), -1);
  //  _blend_mask_l = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
  //  _blend_mask_l(Rect(1896/2 - _overlap,0, 1896/2+1900/2 + 2*_overlap, 1900)).setTo(255);
  //  _blend_mask_r = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
  //  _blend_mask_r(Rect(0,0, 1896/2 + _overlap, 1900)).setTo(255);
  //  _blend_mask_r(Rect(1896+1900/2 - _overlap, 0, 1900/2 + _overlap, 1900)).setTo(255);
  Rect blend_l_area = Rect(1896/2 - _overlap,0,2*_overlap,_xMapArr_l.rows);
  Rect blend_r_area = Rect(1896+1900/2 - _overlap,0,2*_overlap,_xMapArr_l.rows);
  _blend_mask_l = cv::Mat::ones(Size(2*_overlap, _xMapArr_l.rows), CV_8U);
  _blend_mask_l.setTo(255);
  _blend_mask_r = cv::Mat::ones(Size(2*_overlap,_xMapArr_l.rows), CV_8U);
  _blend_mask_r.setTo(255);

  Ptr<detail::Blender> blender;    //定义图像融合器

  blender = detail::Blender::createDefault(detail::Blender::NO, false);    //简单融合方法

  _band_num = 5;
  blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, false);    //多频段融合
  detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
  mb->setNumBands(_band_num);   //设置频段数，即金字塔层数
#if TESTGENERATE
  Mat test = imread("/home/cyx/programes/Pictures/gear360/lab_data/360_0107.jpg");
  Mat img_l = test(Rect(rc_l));
  bitwise_and(img_l, _valid_area_mask, img_l);
  cv::imshow("img_l", img_l);
  Mat img_r = test(Rect(rc_r));
  bitwise_and(img_r, _valid_area_mask, img_r);
  cv::imshow("img_r", img_r);

  Mat ud_l, ud_r;
  cv::remap(img_l, ud_l, _xMapArr_l, _yMapArr_l, CV_INTER_LINEAR);
  cv::remap(img_r, ud_r, _xMapArr_r, _yMapArr_r, CV_INTER_LINEAR);
  cv::imshow("ud_l", ud_l);
  cv::imshow("ud_r", ud_r);

  Mat roi_l, roi_r;
  for(int i = 0; i < 10; i++){
      auto begin = system_clock::now();
//      blender->prepare(Rect(0,0,_xMapArr_l.cols,_xMapArr_l.rows));    //生成全景图像区域
//      blender->feed(ud_l, _blend_mask_l, Point(0,0));
//      blender->feed(ud_r, _blend_mask_r, Point(0,0));
//      blender->blend(_pano, _pano_mask);
      blender->prepare(Rect(0,0,2*_overlap,_xMapArr_r.rows));    //only blend overlap area
      blender->feed(ud_l(blend_r_area), _blend_mask_l, Point(0,0));
      blender->feed(ud_r(blend_r_area), _blend_mask_r, Point(0,0));
      blender->blend(roi_r, _pano_mask);
      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - begin);
      cout << "one frame blend spends "
           << double(duration.count()) * microseconds::period::num / microseconds::period::den
           << "seconds" << endl;
    }
  blender->prepare(Rect(0,0,2*_overlap,_xMapArr_l.rows));    //only blend overlap area
  blender->feed(ud_l(blend_l_area), _blend_mask_l, Point(0,0));
  blender->feed(ud_r(blend_l_area), _blend_mask_r, Point(0,0));
  blender->blend(roi_l, _pano_mask);

  ud_r.copyTo(_pano);
  ud_l(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)).copyTo(_pano(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)));

  Mat roi_l_u, roi_r_u;
  roi_l.convertTo(roi_l_u, CV_8U);
  roi_r.convertTo(roi_r_u, CV_8U);

  roi_l_u.copyTo(_pano(blend_l_area));
  roi_r_u.copyTo(_pano(blend_r_area));

  auto begin1 = system_clock::now();
  int ProcessWidth = _overlap;
  OptimizeSeam(ud_l, 1896/2-_overlap, ud_r, 1896/2+_overlap-ProcessWidth, _pano, roi_l_u, ProcessWidth, 0);
  OptimizeSeam(ud_l, 1896+1900/2-_overlap, ud_r, 1896+1900/2+_overlap-ProcessWidth, _pano, roi_r_u, ProcessWidth, 1);
  auto end1 = system_clock::now();
  auto duration1 = duration_cast<microseconds>(end1 - begin1);
  cout << "one frame step two blend spends: "
       << double(duration1.count()) * microseconds::period::num / microseconds::period::den
       << "seconds" << endl;

//  OptimizeSeam(ud_l, 1896/2-_overlap, ud_r, 1896+1900/2-_overlap,_pano,2*_overlap);

  cv::imshow("pano", _pano);
  cv::imwrite("/home/cyx/programes/Pictures/gear360/lab_data/360_0107_pano.jpg", _pano);
//  cv::imwrite("/home/cyx/programes/Pictures/gear360/lab_data/360_0103_roi_r.jpg", roi_r);

  cv::waitKey();
#endif
}

bool FishEyeStitcher::OptimizeSeam(cv::Mat& img1, int begin1, cv::Mat& img2, int begin2, cv::Mat& pano, cv::Mat& roi, int ProcessWidth, int position){
  double alpha = 1.0;//img1中像素的权重

#if DEBUGSHOW
  cout << "pano size: " << pano.size << endl;
  cout << "img1 size: " << img1.size << endl;
  cout << "img2 size: " << img2.size << endl;
  cout << "roi size: " << roi.size << endl;
  cout << "begin1: " << begin1 << endl;
  cout << "begin2: " << begin2 << endl;
  cout << "pano type: " << pano.type() << endl;
  cout << "img1 type: " << img1.type() << endl;
  cout << "img2 type: " << img2.type() << endl;   // above type = 16 (CV_8UC3)
  cout << "roi type: " << roi.type() << endl;   // attention! roi.type = 19 (CV_16SC3)
#endif

  for (int row = 0; row < pano.rows; row++){
      uchar* p1;
      uchar* p2;
      uchar* t = roi.ptr<uchar>(row);
      uchar* d = pano.ptr<uchar>(row);

////      printf("row: %d, pano.rows: %d\n", row, pano.rows);
//      if(row >= pano.rows){
//          printf("error!!!!!!!!!!!!!!!\n");
//          break;
//        }

      // process left overlap area
      if(0 == position){
          p2 = img1.ptr<uchar>(row);  //获取第i行的首地址
          p1 = img2.ptr<uchar>(row);
        }
      // process right overlap area
      else if(1 == position){
          p1 = img1.ptr<uchar>(row);  //获取第i行的首地址
          p2 = img2.ptr<uchar>(row);
        }
      else{
          CV_Error_(cv::Error::StsBadArg,("Wrong fuse postion input!"));
        }
      // process left part of multi-band edge
      for (int col = begin1; col < begin1+ProcessWidth; col++){
          //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
          alpha = (ProcessWidth - (col - begin1))*1.0 / ProcessWidth;
          int k = (col-begin1);
          d[col * 3] = p1[col * 3] * alpha + t[k * 3] * (1 - alpha);
          d[col * 3 + 1] = p1[col * 3 + 1] * alpha + t[k * 3 + 1] * (1 - alpha);
          d[col * 3 + 2] = p1[col * 3 + 2] * alpha + t[k * 3 + 2] * (1 - alpha);

        }
      // process right part of multi-band edge
      for (int col = begin2; col < begin2+ProcessWidth; col++){
          alpha = (ProcessWidth - (col - begin2))*1.0 / ProcessWidth;
          int k = col - begin2 + roi.cols - ProcessWidth;
          d[col * 3] = t[k * 3] * alpha + p2[col * 3] * (1 - alpha);
          d[col * 3 + 1] = t[k * 3 + 1] * alpha + p2[col * 3 + 1] * (1 - alpha);
          d[col * 3 + 2] = t[k * 3 + 2] * alpha + p2[col * 3 + 2] * (1 - alpha);

        }
    }
  return true;
}

bool FishEyeStitcher::OptimizeSeam(Mat &img1, int begin1, Mat &img2, int begin2, Mat &pano, int ProcessWidth){
  double alpha = 1.0;//img1中像素的权重

#if DEBUGSHOW
  cout << "pano size: " << pano.size << endl;
  cout << "img1 size: " << img1.size << endl;
  cout << "img2 size: " << img2.size << endl;
  cout << "begin1: " << begin1 << endl;
  cout << "begin2: " << begin2 << endl;
  cout << "pano type: " << pano.type() << endl;
  cout << "img1 type: " << img1.type() << endl;
  cout << "img2 type: " << img2.type() << endl;
#endif

  for (int row = 0; row < pano.rows; row++){
      uchar* p1 = img1.ptr<uchar>(row);  //获取第i行的首地址
      uchar* p2 = img2.ptr<uchar>(row);
      uchar* d = pano.ptr<uchar>(row);

      // process left part of multi-band edge
      for (int col = begin1; col < begin1+ProcessWidth; col++){
          //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
          alpha = (ProcessWidth - (col - begin1))*1.0 / ProcessWidth;
          d[col * 3] = p2[col * 3] * alpha + p1[col * 3] * (1 - alpha);
          d[col * 3 + 1] = p2[col * 3 + 1] * alpha + p1[col * 3 + 1] * (1 - alpha);
          d[col * 3 + 2] = p2[col * 3 + 2] * alpha + p1[col * 3 + 2] * (1 - alpha);

        }
      // process right part of multi-band edge
      for (int col = begin2; col < begin2+ProcessWidth; col++){
          alpha = (ProcessWidth - (col - begin2))*1.0 / ProcessWidth;
          d[col * 3] = p1[col * 3] * alpha + p2[col * 3] * (1 - alpha);
          d[col * 3 + 1] = p1[col * 3 + 1] * alpha + p2[col * 3 + 1] * (1 - alpha);
          d[col * 3 + 2] = p1[col * 3 + 2] * alpha + p2[col * 3 + 2] * (1 - alpha);

        }
    }
  return true;
}
