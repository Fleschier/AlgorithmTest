#include "fisheyestitcher.h"

FishEyeStitcher::FishEyeStitcher(){

}

bool FishEyeStitcher::Init(){
  Mat img;
  for(int i = 0; i < 3; i++){
      if(!caps[i].isOpened()){
          printf("failed to open cam%d!\n", i+1);
          return false;
        }
    }
  if(!caps[0].read(img)){
      printf("failed to read img!\n");
      return false;
    }
  ud.InitMartix(img, 1600/2);
  return true;
}

bool FishEyeStitcher::run(){
  startCapture = true;
  return true;
}

bool FishEyeStitcher::stop(){
  startCapture = false;
  return true;
}

bool FishEyeStitcher::PreProcess(int idx, Mat& dst){
  Mat img;
  while(startCapture){
      if(!caps[i].read(img)){
          caps[i].set(CAP_PROP_POS_AVI_RATIO, 0);
          caps[i].read(img);
        }
      ud.MatrixUndistort(img, dst, idx);
    }

}

void FishEyeStitcher::Stitch(){

}
