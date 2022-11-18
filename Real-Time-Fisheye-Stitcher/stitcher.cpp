#include "stitcher.h"
#include <chrono>

using namespace std;
using namespace chrono;
using namespace cv;

#define DEBUGSHOW 1
#define TESTGENERATE 1

uint FishEyeStitcher::current_idx = 0;

FishEyeStitcher::FishEyeStitcher(){

}

bool FishEyeStitcher::Init(){
  _overlap = 30;

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

#if TESTGENERATE
  _blend_mask_l = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
  _blend_mask_l(Rect(1896/2 - _overlap,0, 1896/2+1900/2 + 2*_overlap, 1900)).setTo(255);
  _blend_mask_r = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
  _blend_mask_r(Rect(0,0, 1896/2 + _overlap, 1900)).setTo(255);
  _blend_mask_r(Rect(1896+1900/2 - _overlap, 0, 1900/2 + _overlap, 1900)).setTo(255);

  Ptr<detail::Blender> blender;    //定义图像融合器

  blender = detail::Blender::createDefault(detail::Blender::NO, false);    //简单融合方法
  //  //羽化融合方法
  //  blender = detail::Blender::createDefault(detail::Blender::FEATHER, false);
  //  //dynamic_cast多态强制类型转换时候使用
  //  detail::FeatherBlender* fb = dynamic_cast<detail::FeatherBlender*>(static_cast<detail::Blender*>(blender));
  //  fb->setSharpness(0.005);    //设置羽化锐度

  _band_num = 2;
  blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, false);    //多频段融合
  detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
  mb->setNumBands(_band_num);   //设置频段数，即金字塔层数

  //  blender->prepare(Rect(0,0,_xMapArr_l.cols,_xMapArr_l.rows));    //生成全景图像区域
//  cv::imshow("_valid_area_mask", _valid_area_mask);
//  cv::imshow("_blend_mask_l", _blend_mask_l);
//  cv::imshow("_blend_mask_r", _blend_mask_r);
  Mat test = imread("/home/cyx/programes/Pictures/gear360/lab_data/360_0103.jpg");
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

//  ud_l.convertTo(ud_l, CV_16S);
//  ud_r.convertTo(ud_r, CV_16S);
  for(int i = 0; i < 10; i++){
      auto begin = system_clock::now();
//      ud_l.convertTo(ud_l, CV_16S);
//      ud_r.convertTo(ud_r, CV_16S);
      blender->prepare(Rect(0,0,_xMapArr_l.cols,_xMapArr_l.rows));    //生成全景图像区域
      blender->feed(ud_l, _blend_mask_l, Point(0,0));
      blender->feed(ud_r, _blend_mask_r, Point(0,0));
      blender->blend(_pano, _pano_mask);
      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - begin);
      cout << "one frame blend spends "
           << double(duration.count()) * microseconds::period::num / microseconds::period::den
           << "seconds" << endl;
    }

//  cv::imshow("pano", _pano);
  cv::imwrite("/home/cyx/programes/Pictures/gear360/lab_data/360_0103_pano.jpg", _pano);

  cv::waitKey();
#endif
}
