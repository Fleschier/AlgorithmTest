#ifndef STITCHER_H
#define STITCHER_H

#include "opencv2/opencv.hpp"
#include "threadsafe_queue.hpp"
#include <atomic>

#define MAXQUEUELEN 10

struct IMG{
  std::atomic<uint> idx;
  cv::Mat imgdata;
};

class FishEyeStitcher
{
private:
  static uint current_idx;
  cv::Rect rc_l, rc_r;    // valid rectangular area of raw fisheye img (dule fisheye)
  cv::Mat _valid_area_mask;   // valid circle area mask of cutted single fisheye
  cv::Mat _xMapArr_l, _yMapArr_l;
  cv::Mat _xMapArr_r, _yMapArr_r;
  cv::Mat _pano, _pano_mask;
  cv::Mat _unwarp_img_l, _unwarp_img_r;
  threadsafe_queue<cv::Mat> _unwarped_l, _unwarped_r;
  cv::Mat _blend_mask_l, _blend_mask_r;
  int _overlap;
  int _band_num;
public:
  FishEyeStitcher();
  bool Init();
};

#endif // STITCHER_H
