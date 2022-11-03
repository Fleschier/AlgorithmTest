#ifndef FISHEYESTITCHER_H
#define FISHEYESTITCHER_H

#include"undistort.h"
#include<atomic>
#include"threadpool/thread_pool.h"

class FishEyeStitcher
{
private:
  Undistort _ud;
  thread_pool _pool;
  VideoCapture _caps[3];
  Mat _capImgs[3];
  Mat _PreparedImgs[3];
  std::vector<std::future<bool>> _flgs;
  atomic_bool _isInit;

  // multi threads --------------------------
  atomic_bool _startCapture;
  // ------------------------

  Mat _pano;

  uint _ovlps[3];
//  int _cut_border;

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
