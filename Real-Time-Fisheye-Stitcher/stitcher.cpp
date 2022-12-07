#include "stitcher.h"
#include <chrono>

using namespace std;
using namespace chrono;
using namespace cv;

#define DEBUGSHOW 0

#ifndef TESTGENERATE
#define TESTGENERATE 1
#endif

// Polynomial Coefficients
#define    P1_    -7.5625e-17
#define    P2_     1.9589e-13
#define    P3_    -1.8547e-10
#define    P4_     6.1997e-08
#define    P5_    -6.9432e-05
#define    P6_     0.9976
//#define    P1_     -6.82564199647755E-17
//#define    P2_     1.45146475845449E-13
//#define    P3_     -8.36765527228239E-11
//#define    P4_     -1.6793129839737E-08
//#define    P5_     -4.67623718248117E-05
//#define    P6_     0.995854937248903

#define _PI 3.1415926

uint FishEyeStitcher::current_idx = 0;

FishEyeStitcher::FishEyeStitcher(){

}

bool FishEyeStitcher::Init(){
  _overlap = 45;

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
  __genScaleMap();
  printf("init success!\n");
  return true;
}

bool FishEyeStitcher::__captureThread(){
  std::unique_lock<std::mutex> lock_l(_dulfish_mut_l);
  std::unique_lock<std::mutex> lock_r(_dulfish_mut_r);
  _cap_cond.wait(lock_l);
}

bool FishEyeStitcher::__preProcessThread(int area_idx){
  Mat single_fish;

  // process left fish eye
  if(0 == area_idx){
      std::unique_lock<std::mutex> lock(_dulfish_mut_l);
      // if too much img suspend, wait for single to contiunue
      single_fish = _dul_fish_img(rc_l);
      _preprocess_cond.wait(lock, [this]{return (_unwarped_l.length() <= MAXQUEUELEN);});
      cv::remap(single_fish, _unwarp_img_l, _xMapArr_l, _yMapArr_l, INTER_LINEAR);
      //      if(lock.owns_lock())    // 如果持有锁
      //        lock.unlock();  // 解锁，然后执行任务
      _unwarped_l.push(std::move(single_fish));   // if buff would release?
      _fuse_cond.notify_one();
    }
  // process right fish eye
  else if(1 == area_idx){
      std::unique_lock<std::mutex> lock(_dulfish_mut_r);

      single_fish = _dul_fish_img(rc_r);
      cv::remap(single_fish, _unwarp_img_r, _xMapArr_r, _yMapArr_r, INTER_LINEAR);
      _preprocess_cond.wait(lock, [this]{return (_unwarped_l.length() <= MAXQUEUELEN);});
      _unwarped_r.push(std::move(single_fish));
      _fuse_cond.notify_one();
    }
  else{
      CV_Error_(cv::Error::StsBadArg,("Undefined Fisheye area idx!"));
    }
}

#if TESTGENERATE
void FishEyeStitcher::TestGenerate(int type){

  Mat test = imread("/home/fleschier/programes/Pictures/gear360/lab_data/360_0108.jpg");
  Mat img_l = test(rc_l);
  bitwise_and(img_l, _valid_area_mask, img_l);
  Mat img_r = test(rc_r);
  bitwise_and(img_r, _valid_area_mask, img_r);
  Mat img_l_comp, img_r_comp;

  bool pre_light_compen = false;
  if(pre_light_compen){
      auto cmpe_begin = system_clock::now();
      __compenLightFO(img_l, img_l_comp);
      auto cmpe_end = system_clock::now();
      auto cmpe_duration = duration_cast<microseconds>(cmpe_end - cmpe_begin);
      cout << "one frame light compensation spends: "
           << double(cmpe_duration.count()) * microseconds::period::num / microseconds::period::den
           << "seconds" << endl;
      __compenLightFO(img_r, img_r_comp);

//      cv::imshow("img_l_comp", img_l_comp);
//      cv::imshow("img_r_comp", img_r_comp);
    }
  else{
      img_l_comp = img_l;
      img_r_comp = img_r;
    }

  Mat ud_l, ud_r;
  cv::remap(img_l_comp, ud_l, _xMapArr_l, _yMapArr_l, INTER_LINEAR);
  cv::remap(img_r_comp, ud_r, _xMapArr_r, _yMapArr_r, INTER_LINEAR);
//  cv::imshow("ud_l", ud_l);
//  cv::imshow("ud_r", ud_r);

  ud_r.copyTo(_pano);
  ud_l(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)).copyTo(_pano(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)));

  // multi band blend shrink version
  if(0 == type){
      Ptr<detail::Blender> blender;    //定义图像融合器
      _band_num = 5;
      blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, false);    //多频段融合
      detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
      mb->setNumBands(_band_num);   //设置频段数，即金字塔层数

      Rect blend_l_area = Rect(1896/2 - _overlap,0,2*_overlap,_xMapArr_l.rows);
      Rect blend_r_area = Rect(1896+1900/2 - _overlap,0,2*_overlap,_xMapArr_l.rows);
      _blend_mask_l = cv::Mat::ones(Size(2*_overlap, _xMapArr_l.rows), CV_8U);
      _blend_mask_l.setTo(255);
      _blend_mask_r = cv::Mat::ones(Size(2*_overlap,_xMapArr_l.rows), CV_8U);
      _blend_mask_r.setTo(255);
      Mat roi_l, roi_r;
      for(int i = 0; i < 10; i++){
          auto begin = system_clock::now();
          blender->prepare(Rect(0,0,2*_overlap,_xMapArr_r.rows));    //only blend overlap area
          blender->feed(ud_l(blend_r_area), _blend_mask_l, Point(0,0));
          blender->feed(ud_r(blend_r_area), _blend_mask_r, Point(0,0));
          blender->blend(roi_r, _pano_mask);
          auto end = system_clock::now();
          auto duration = duration_cast<microseconds>(end - begin);
          cout << "one frame two-step blend spends "
               << double(duration.count()) * microseconds::period::num / microseconds::period::den
               << "seconds" << endl;
        }
      blender->prepare(Rect(0,0,2*_overlap,_xMapArr_l.rows));    //only blend overlap area
      blender->feed(ud_l(blend_l_area), _blend_mask_l, Point(0,0));
      blender->feed(ud_r(blend_l_area), _blend_mask_r, Point(0,0));
      blender->blend(roi_l, _pano_mask);

      Mat roi_l_u, roi_r_u;
      roi_l.convertTo(roi_l_u, CV_8U);
      roi_r.convertTo(roi_r_u, CV_8U);

      roi_l_u.copyTo(_pano(blend_l_area));
      roi_r_u.copyTo(_pano(blend_r_area));

      auto begin1 = system_clock::now();
      int ProcessWidth = 10;
      __optimizeSeam(ud_l, 1896/2-_overlap, ud_r, 1896/2+_overlap-ProcessWidth, _pano, roi_l_u, ProcessWidth, 0);
      __optimizeSeam(ud_l, 1896+1900/2-_overlap, ud_r, 1896+1900/2+_overlap-ProcessWidth, _pano, roi_r_u, ProcessWidth, 1);
      auto end1 = system_clock::now();
      auto duration1 = duration_cast<microseconds>(end1 - begin1);
      cout << "one frame step two blend spends: "
           << double(duration1.count()) * microseconds::period::num / microseconds::period::den
           << "seconds" << endl;
    }

  // full multi-band blender
  else if(1 == type){
      Ptr<detail::Blender> blender;    //定义图像融合器
      _band_num = 5;
      blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, false);    //多频段融合
      detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
      mb->setNumBands(_band_num);   //设置频段数，即金字塔层数
      //      _blend_mask_l = cv::Mat::zeros(_xMapArr_l.size(), CV_8U);
      //      _blend_mask_r = cv::Mat::zeros(_xMapArr_r.size(), CV_8U);
      //      cv::remap(_valid_area_mask, _blend_mask_l, _xMapArr_l, _yMapArr_l, INTER_NEAREST);
      //      cv::remap(_valid_area_mask, _blend_mask_r, _xMapArr_r, _yMapArr_r, INTER_NEAREST);
      //      cvtColor(_blend_mask_l, _blend_mask_l, COLOR_RGB2GRAY);
      //      cvtColor(_blend_mask_r, _blend_mask_r, COLOR_BGR2GRAY);
      //      imshow("mask_l", _blend_mask_l);
      //      imshow("mask_r", _blend_mask_r);
      _blend_mask_l = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
      _blend_mask_l(Rect(1896/2 - _overlap,0, 1896/2+1900/2 + 2*_overlap, 1900)).setTo(255);
      _blend_mask_r = cv::Mat::zeros(Size(_xMapArr_l.cols, _xMapArr_l.rows), CV_8U);
      _blend_mask_r(Rect(0,0, 1896/2 + _overlap, 1900)).setTo(255);
      _blend_mask_r(Rect(1896+1900/2 - _overlap, 0, 1900/2 + _overlap, 1900)).setTo(255);

      for(int i = 0; i < 10; i++){
          auto begin = system_clock::now();
          blender->prepare(Rect(0,0,_xMapArr_l.cols,_xMapArr_l.rows));    //生成全景图像区域
          blender->feed(ud_l, _blend_mask_l, Point(0,0));
          blender->feed(ud_r, _blend_mask_r, Point(0,0));
          blender->blend(_pano, _pano_mask);
          auto end = system_clock::now();
          auto duration = duration_cast<microseconds>(end - begin);
          cout << "one frame multi-band blend spends "
               << double(duration.count()) * microseconds::period::num / microseconds::period::den
               << "seconds" << endl;
        }
    }

  // simple linear fuse
  else if(2 == type){
      auto begin = system_clock::now();
      __optimizeSeam(ud_l, 1896/2-_overlap, ud_r, 1896+1900/2-_overlap,_pano,2*_overlap);
      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - begin);
      cout << "one frame simple blend spends "
           << double(duration.count()) * microseconds::period::num / microseconds::period::den
           << "seconds" << endl;
    }

  auto cmpen_begin = system_clock::now();
  int template_area_width = 200;
  int bias_width = 30;
  int bottom_top_cut = 0;
//  __compenLightFallOff(Rect(1896/2-_overlap,0,2*_overlap,_pano.rows),
//                       Rect(1896/2-_overlap-template_area_width,0,template_area_width,_pano.rows),
//                       Rect(1896/2+_overlap,0,template_area_width,_pano.rows),
//                       1);
//  __compenLightFallOff(Rect(1896/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
//                       Rect(1896/2-_overlap-bias_width-template_area_width,0,template_area_width+bias_width,_pano.rows),
//                       Rect(1896/2+_overlap,0,template_area_width+bias_width,_pano.rows));
  __compenLightFallOff(Rect(1896/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
                       Rect(1896/2-_overlap-bias_width-template_area_width,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut),
                       Rect(1896/2+_overlap,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut));
  auto cmpen_end = system_clock::now();
  auto cmpen_duration = duration_cast<microseconds>(cmpen_end - cmpen_begin);
  cout << "one frame light compensation spends: "
       << double(cmpen_duration.count()) * microseconds::period::num / microseconds::period::den
       << "seconds" << endl;
//  __compenLightFallOff(Rect(1896+1900/2-_overlap,0,2*_overlap,_pano.rows),
//                       Rect(1896+1900/2-_overlap-template_area_width,0,template_area_width,_pano.rows),
//                       Rect(1896+1900/+_overlap,0,template_area_width,_pano.rows),
//                       1);
//  __compenLightFallOff(Rect(1896+1900/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
//                       Rect(1896+1900/2-_overlap-bias_width-template_area_width,0,template_area_width+bias_width,_pano.rows),
//                       Rect(1896+1900/2+_overlap,0,template_area_width+bias_width,_pano.rows));
  __compenLightFallOff(Rect(1896+1900/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
                       Rect(1896+1900/2-_overlap-bias_width-template_area_width,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut),
                       Rect(1896+1900/2+_overlap,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut));


  //  cv::imshow("pano", _pano);
  cv::imwrite("/home/fleschier/programes/Pictures/gear360/lab_data/360_0108_pano.jpg", _pano);
  //  cv::imwrite("/home/cyx/programes/Pictures/gear360/lab_data/360_0103_roi_r.jpg", roi_r);

//  cv::waitKey();
}

void FishEyeStitcher::TestGenerateVideo(){
  string prefix = "/home/fleschier/programes/Videos/gear360/";
  string videoname = "360_0108";
  VideoCapture cap;
  cap.open(prefix+videoname+".MP4");
  VideoWriter writer;
  writer.open(prefix+videoname+"_pano.mp4", VideoWriter::fourcc('M','J','P','G'), 30, Size(_xMapArr_l.cols, _xMapArr_l.rows));
  if(!cap.isOpened()){
      printf("failed to open video!\n");
    }
  Mat test;
  while(cap.read(test)){
      Mat img_l = test(rc_l);
      bitwise_and(img_l, _valid_area_mask, img_l);
//      cv::imshow("img_l", img_l);
      Mat img_r = test(rc_r);
      bitwise_and(img_r, _valid_area_mask, img_r);
//      cv::imshow("img_r", img_r);

      Mat img_l_comp, img_r_comp;
      bool pre_light_compen = false;
      if(pre_light_compen){
          __compenLightFO(img_l, img_l_comp);
          __compenLightFO(img_r, img_r_comp);
        }
      else{
          img_l_comp = img_l;
          img_r_comp = img_r;
        }

      Mat ud_l, ud_r;
      cv::remap(img_l_comp, ud_l, _xMapArr_l, _yMapArr_l, INTER_LINEAR);
      cv::remap(img_r_comp, ud_r, _xMapArr_r, _yMapArr_r, INTER_LINEAR);

      ud_r.copyTo(_pano);
      ud_l(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)).copyTo(_pano(Rect(1896/2,0, 1896/2+1900/2, _pano.rows)));
       __optimizeSeam(ud_l, 1896/2-_overlap, ud_r, 1896+1900/2-_overlap,_pano,2*_overlap);

       // light fall off compnesation
       int template_area_width = 600;
       int bias_width = 50;
       int bottom_top_cut = 300;
       __compenLightFallOff(Rect(1896/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
                            Rect(1896/2-_overlap-bias_width-template_area_width,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut),
                            Rect(1896/2+_overlap,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut));
       __compenLightFallOff(Rect(1896+1900/2-_overlap-bias_width,0,2*(_overlap+bias_width),_pano.rows),
                            Rect(1896+1900/2-_overlap-bias_width-template_area_width,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut),
                            Rect(1896+1900/2+_overlap,bottom_top_cut,template_area_width+bias_width,_pano.rows-2*bottom_top_cut));

       writer.write(_pano);
    }


}

#endif

bool FishEyeStitcher::__optimizeSeam(cv::Mat& img1, int begin1, cv::Mat& img2, int begin2, cv::Mat& pano, cv::Mat& roi, int ProcessWidth, int position){
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
          d[col * 3] = p1[col * 3] * alpha + t[k * 3] * (1 - alpha) + 0.5;
          d[col * 3 + 1] = p1[col * 3 + 1] * alpha + t[k * 3 + 1] * (1 - alpha) + 0.5;
          d[col * 3 + 2] = p1[col * 3 + 2] * alpha + t[k * 3 + 2] * (1 - alpha) + 0.5;

        }
      // process right part of multi-band edge
      for (int col = begin2; col < begin2+ProcessWidth; col++){
          alpha = (ProcessWidth - (col - begin2))*1.0 / ProcessWidth;
          int k = col - begin2 + roi.cols - ProcessWidth;
          d[col * 3] = t[k * 3] * alpha + p2[col * 3] * (1 - alpha) + 0.5;
          d[col * 3 + 1] = t[k * 3 + 1] * alpha + p2[col * 3 + 1] * (1 - alpha) + 0.5;
          d[col * 3 + 2] = t[k * 3 + 2] * alpha + p2[col * 3 + 2] * (1 - alpha) + 0.5;

        }
    }
  return true;
}

bool FishEyeStitcher::__optimizeSeam(Mat &img1, int begin1, Mat &img2, int begin2, Mat &pano, int ProcessWidth){
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
          d[col * 3] = MIN(p2[col * 3] * alpha + p1[col * 3] * (1 - alpha) + 0.5, 255);
          d[col * 3 + 1] = MIN(p2[col * 3 + 1] * alpha + p1[col * 3 + 1] * (1 - alpha) + 0.5, 255);
          d[col * 3 + 2] = MIN(p2[col * 3 + 2] * alpha + p1[col * 3 + 2] * (1 - alpha) + 0.5, 255);

        }
      // process right part of multi-band edge
      for (int col = begin2; col < begin2+ProcessWidth; col++){
          alpha = (ProcessWidth - (col - begin2))*1.0 / ProcessWidth;
          d[col * 3] = MIN(p1[col * 3] * alpha + p2[col * 3] * (1 - alpha) + 0.5, 255);
          d[col * 3 + 1] = MIN(p1[col * 3 + 1] * alpha + p2[col * 3 + 1] * (1 - alpha) + 0.5, 255);
          d[col * 3 + 2] = MIN(p1[col * 3 + 2] * alpha + p2[col * 3 + 2] * (1 - alpha) + 0.5, 255);

        }
    }
  return true;
}

//!
//! @brief  Fisheye Light Fall-off Compensation
//! @param  in_img  LFO-uncompensated image
//! @param  return  LFO-compensated image
//!
void FishEyeStitcher::__compenLightFO(cv::Mat& imgin, cv::Mat& imgout)
{
  cv::Mat out_img_double;

//  cv::Mat out_img_double(imgin.size(), imgin.type());
//  cv::Mat rgb_ch[3];
//  cv::Mat rgb_ch_double[3];
//  cv::split(imgin, rgb_ch);
//  rgb_ch[0].convertTo(rgb_ch_double[0], _scale_map.type());
//  rgb_ch[1].convertTo(rgb_ch_double[1], _scale_map.type());
//  rgb_ch[2].convertTo(rgb_ch_double[2], _scale_map.type());
//  //
//  rgb_ch_double[0] = rgb_ch_double[0].mul(_scale_map); // element-wise multiplication
//  rgb_ch_double[1] = rgb_ch_double[1].mul(_scale_map);
//  rgb_ch_double[2] = rgb_ch_double[2].mul(_scale_map);
//  cv::merge(rgb_ch_double, 3, out_img_double);

  imgin.convertTo(out_img_double, _scale_map.type());
  out_img_double = out_img_double.mul(_scale_map);

  out_img_double.convertTo(imgout, CV_8UC3);

}   // compenLightFO()

//!
//! @brief Fisheye Light Fall-off Compensation: Scale_Map Construction
//! @param R_pf  everse profile model
//!
//!     Update member m_scale_map
//!
void FishEyeStitcher::__genScaleMap()
{
    // TODO: remove duplicate params

    //------------------------------------------------------------------------//
    // Generate R_pf (reverse light fall-off profile)                         //
    //------------------------------------------------------------------------//
    int H = 1900;
    int W = 1914;
    int W_ = W/2;
    int H_ = H/2;
    cv::Mat x_coor = cv::Mat::zeros(1, W_, CV_32F);
    cv::Mat temp(x_coor.size(), x_coor.type());

    for (int i = 0; i < W_; ++i)
    {
        x_coor.at<float>(0, i) = i;
    }

    //-----------------------------------------------------------------------//
    //  R_pf = P1_ * (x_coor.^5.0) + P2_ * (x_coor.^4.0) +                   //
    //         P3_ * (x_coor.^3.0) + P4_ * (x_coor.^2.0) +                   //
    //         P5_ * x_coor        + P6_;                                    //
    //-----------------------------------------------------------------------//
    cv::Mat R_pf = cv::Mat::zeros(x_coor.size(), x_coor.type());
    cv::pow(x_coor, 5.0, temp);
    R_pf = R_pf + P1_ * temp;
    cv::pow(x_coor, 4.0, temp);
    R_pf = R_pf + P2_ * temp;
    cv::pow(x_coor, 3.0, temp);
    R_pf = R_pf + P3_ * temp;
    cv::pow(x_coor, 2.0, temp);
    R_pf = R_pf + P4_ * temp;
    R_pf = R_pf + P5_ * x_coor + P6_;

    // PF_LUT
    cv::divide(1, R_pf, R_pf); //element-wise inverse

    //------------------------------------------------------------------------//
    // Generate scale map                                                     //
    //------------------------------------------------------------------------//
    // Create IV quadrant map
    cv::Mat scale_map_quad_4 = cv::Mat::zeros(H_, W_, R_pf.type());
    float da = R_pf.at<float>(0, W_ - 1);
    int x, y;
    float r, a, b;

    for (x = 0; x < W_; ++x)
    {
        for (y = 0; y < H_; ++y)
        {
            r = std::floor(sqrt(std::pow(x, 2) + std::pow(y, 2)));
            if (r >= (W_ - 1))
            {
                scale_map_quad_4.at<float>(y, x) = da;
            }
            else
            {
                a = R_pf.at<float>(0, r);
                if ((x < W_) && (y < H_)) // within boundaries
                    b = R_pf.at<float>(0, r + 1);
                else // on boundaries
                    b = R_pf.at<float>(0, r);
                scale_map_quad_4.at<float>(y, x) = (a + b) / 2.0f;
            }
        } // x()
    } // y()

    // Assume Optical Symmetry & Flip
    cv::Mat scale_map_quad_1(scale_map_quad_4.size(), scale_map_quad_4.type());
    cv::Mat scale_map_quad_2(scale_map_quad_4.size(), scale_map_quad_4.type());
    cv::Mat scale_map_quad_3(scale_map_quad_4.size(), scale_map_quad_4.type());
    //
    cv::flip(scale_map_quad_4, scale_map_quad_1, 0); // quad I, up-down or around x-axis
    cv::flip(scale_map_quad_4, scale_map_quad_3, 1); // quad III, left-right or around y-axis
    cv::flip(scale_map_quad_1, scale_map_quad_2, 1); // quad II, up-down or around x-axis
    //
    cv::Mat quad_21, quad_34;
    cv::hconcat(scale_map_quad_2, scale_map_quad_1, quad_21);
    cv::hconcat(scale_map_quad_3, scale_map_quad_4, quad_34);
    //
//    cv::vconcat(quad_21, quad_34, _scale_map);
    Mat single_channel;
    Mat channels[3];
    cv::vconcat(quad_21, quad_34, single_channel);

    channels[0] = single_channel.clone();
    channels[1] = single_channel.clone();
    channels[2] = single_channel.clone();
    cv::merge(channels, 3, _scale_map);

//    FileStorage cvfs;
//    cvfs.open("./scalemap_test.yml", FileStorage::WRITE);
//    cvfs.write("_scale_map",channels[2]);
//    cvfs.release();

}   // genScaleMap()

double ValueBGR2GRAY(uint B, uint G, uint R){
//  return (15*B + 75*G + 38*R) >> 7;
  return (0.114*B + 0.587*G + 0.299*R);
}

void FishEyeStitcher::__compenLightFallOff(cv::Rect FixArea, cv::Rect TemplateArea_l, cv::Rect TemplateArea_r){

  cv::Mat fixArea_gray, tpl_l, tpl_r;

  cvtColor(_pano(FixArea), fixArea_gray, COLOR_BGR2GRAY);
  cvtColor(_pano(TemplateArea_l), tpl_l, COLOR_BGR2GRAY);
  cvtColor(_pano(TemplateArea_r), tpl_r, COLOR_BGR2GRAY);
  double mean_l, mean_r, mean_FixArea;
  mean_l = cv::mean(tpl_l)[0];
  mean_r = cv::mean(tpl_r)[0];
  mean_FixArea = cv::mean(fixArea_gray)[0];

#if DEBUGSHOW
  cout << "mean_l: " << mean_l << endl;
  cout << "mean_r: " << mean_r << endl;
#endif

  bool useEnhanced_algo = true;

  if(useEnhanced_algo){ // two-setp compensation
      int mid_col = FixArea.x + FixArea.width/2;
      for(int row = 0; row < FixArea.height; row++){
          int row_pre = MAX(row - 1, 0);
          int row_next = MIN(row + 1, _pano.rows - 1);
          uchar* d_pre = _pano.ptr<uchar>(row_pre);
          uchar* d_next = _pano.ptr<uchar>(row_next);
          uchar* d = _pano.ptr<uchar>(row);
          for(int col = FixArea.x; col < FixArea.x+FixArea.width; col++){
              /*
               * for each point E, considering the neareast 8 points
               * | A | B | C |
               * | D | E | F |
               * | G | H | I |
               *
               * using Gaussian kernel paramenter as the weight of each element of the matrix
               *
               *              | 0.3679 | 0.6065 | 0.3679 |
               * (1/4.8796) * | 0.6065 | 1.0000 | 0.6065 |
               *              | 0.3679 | 0.6065 | 0.3679 |
               */

              int col_pre = MAX(col - 1, 0);
              int col_next = MIN(col + 1, _pano.cols-1);
              // top left point
              double A = ValueBGR2GRAY(d_pre[col_pre*3], d_pre[col_pre*3+1], d_pre[col_pre*3+2]);
              // top point
              double B = ValueBGR2GRAY(d_pre[col*3], d_pre[col*3+1], d_pre[col*3+2]);
              // top right point
              double C = ValueBGR2GRAY(d_pre[col_next*3], d_pre[col_next*3+1], d_pre[col_next*3+2]);
              // left point
              double D = ValueBGR2GRAY(d[col_pre*3], d[col_pre*3+1], d[col_pre*3+2]);
              // current point
              double E = ValueBGR2GRAY(d[col*3], d[col*3+1], d[col*3+2]);
              // right point
              double F = ValueBGR2GRAY(d[col_next*3], d[col_next*3+1], d[col_next*3+2]);
              // bottom left point
              double G = ValueBGR2GRAY(d_next[col_pre*3], d_next[col_pre*3+1], d_next[col_pre*3+2]);
              // bottom point
              double H = ValueBGR2GRAY(d_next[col*3], d_next[col*3+1], d_next[col*3+2]);
              // bottom right point
              double I = ValueBGR2GRAY(d_next[col_next*3], d_next[col_next*3+1], d_next[col_next*3+2]);

              double k = 1 / 4.8796;
              double w1 = k*0.3679;
              double w2 = k*0.6065;
              double w3 = k*0.3679;
              double w4 = k*0.6065;
              double w5 = k*1.0000;
              double w6 = k*0.6065;
              double w7 = k*0.3679;
              double w8 = k*0.6065;
              double w9 = k*0.3679;

              double neareast_target_gray =
                  w1*A + w2*B + w3*C +
                  w4*D + w5*E + w6*F +
                  w7*G + w8*H + w9*I;

//              double new_scale = 0.8*neareast_target_gray / E + 0.2*compensatoin_rate[idx];
              double alpha = (col - FixArea.x)*1.0 / FixArea.width*1.0;
              double target_gray = alpha * mean_l + (1-alpha) * mean_r;
              double new_scale =
                  0.4 * neareast_target_gray / E +
                 // 0.05 * neareast_target_gray / ((A + B + D) / 3) +
                  0.6 * target_gray / mean_FixArea;
//              cout << "neareast_target_gray: " << neareast_target_gray << endl;
//              cout << "E: " << E << endl;
//              cout << "new_scale: " << new_scale << endl;
//              cout << "target_gray: " << target_gray << endl;
//              cout << "mean_FixArea: " << mean_FixArea << endl;
//              cout << "current_meam_grays[" << idx << "]: " << current_meam_grays[idx] << endl;
              new_scale = MAX(1.0, new_scale);
//              int mthreshold = 155;
//              if(target_gray > mthreshold){
//                  new_scale = (new_scale - 1) * (1 - (target_gray-mthreshold)*1.0/target_gray) + 1.0;
//                }

              double k1 = abs(col - mid_col) * 1.0 / ((FixArea.width * 1.0)/2.0) ; // col --> 0 or 2*mid_col , k1 --> 1
              k1 = 1-k1;
              // double k1 = cos((_PI/2.0*mid_col)*(col-mid_col));       // col --> mid_col , k1 --> 1
              d[col * 3] = k1 * MIN(d[col * 3]*new_scale, 255.0) + (1 - k1) * d[col * 3] + 0.5;
              d[col * 3 + 1] = k1 * MIN(d[col * 3 + 1]*new_scale, 255.0) + (1 - k1) * d[col * 3 + 1] + 0.5;
              d[col * 3 + 2] = k1 * MIN(d[col * 3 + 2]*new_scale, 255.0) + (1 - k1) * d[col * 3 + 2] + 0.5;

            } // cols
        } // rows
    } // two-step light fall-off compensation

}
