#ifndef FISHEYESTITCHER_H
#define FISHEYESTITCHER_H

#include"undistort.h"
#include<atomic>
#include<mutex>
#include<condition_variable>

class FishEyeStitcher
{
private:
  Undistort ud;
  vector<VideoCapture> caps = {VideoCapture("./cam1.mp4"), VideoCapture("./cam2.mp4"), VideoCapture("./cam3.mp4")};
  vector<Mat> PreparedImgs = {Mat(), Mat(), Mat()};

  // multi threads --------------------------
  atomic_bool startCapture = false;
  mutex mut1, mut2, mut3;
  condition_variable data_cond;
  // ------------------------

public:
  FishEyeStitcher();
  bool Init();
  bool PreProcess(int idx, Mat& dst);
  bool run();
  bool stop();
  void Stitch();

};

#endif // FISHEYESTITCHER_H
