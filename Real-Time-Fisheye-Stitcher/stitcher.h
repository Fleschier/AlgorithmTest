#ifndef STITCHER_H
#define STITCHER_H

#include "opencv2/opencv.hpp"

class Stitcher
{
private:
  cv::Mat _xMapArr_l, _yMapArr_l;
  cv::Mat _xMapArr_r, _yMapArr_r;
  cv::Mat _pano;
  cv::Mat _unwarp_img_l, _unwarp_img_r;
  cv::Mat _mask_l, _mask_r;
public:
  Stitcher();
};

#endif // STITCHER_H
