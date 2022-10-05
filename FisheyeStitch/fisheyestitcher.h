#ifndef FISHEYESTITCHER_H
#define FISHEYESTITCHER_H

#include"undistort.h"
#include<atomic>
#include"threadpool/thread_pool.h"

class FishEyeStitcher
{
private:
  Undistort ud;
  thread_pool pool;
  VideoCapture caps[3];
  Mat capImgs[3];
  Mat PreparedImgs[3];
  std::vector<std::future<bool>> flgs;
  atomic_bool isInit;

  // multi threads --------------------------
  atomic_bool startCapture;
  // ------------------------

  Mat pano;

public:
  FishEyeStitcher();
  ~FishEyeStitcher();
  bool Init();
  bool PreProcess(int idx);
  bool run();
  bool stop();
  void Stitch();

};

#endif // FISHEYESTITCHER_H
