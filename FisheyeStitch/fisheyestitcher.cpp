#include "fisheyestitcher.h"
#include<chrono>
using namespace chrono;

#define DEBUGSHOW 0
#define RUNTIME 0
#define STORERESULT 0

FishEyeStitcher::FishEyeStitcher(){
    isInit = Init();
}

FishEyeStitcher::~FishEyeStitcher(){
}

bool FishEyeStitcher::Init(){
  startCapture = false;
  caps[0].open("./cam1.mp4");
  caps[1].open("./cam2.mp4");
  caps[2].open("./cam3.mp4");
  Mat img1, img2, img3;
  for(int i = 0; i < 3; i++){
      if(!caps[i].isOpened()){
          printf("failed to open cam%d!\n", i+1);
          return false;
        }
    }
  if(!caps[0].read(img1) || !caps[1].read(img2) || !caps[2].read(img3)){
      printf("failed to read img!\n");
      return false;
    }
  ud.InitMartix(img1, img2, img3);
  return true;
}

bool FishEyeStitcher::run(){
    if(!isInit){
        printf("something wrong while init!\n");
        return false;
    }
  startCapture = true;
  thread t = std::thread(&FishEyeStitcher::Stitch, this);
  if(t.joinable()){
      t.detach();
  }
  printf("success run the stitcher!\n");
  return true;
}

bool FishEyeStitcher::stop(){
  startCapture = false;
  return true;
}

bool FishEyeStitcher::PreProcess(int idx){
#if RUNTIME
        auto start = system_clock::now();
#endif

    if(!caps[idx].read(capImgs[idx])){
        caps[idx].set(CAP_PROP_POS_AVI_RATIO, 0);
        caps[idx].read(capImgs[idx]);
    }

    ud.MatrixUndistort(capImgs[idx], PreparedImgs[idx], idx);
#if RUNTIME
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "preprocess one frame spends "
             << double(duration.count()) * microseconds::period::num / microseconds::period::den
             << "seconds" << endl;
#endif

#if DEBUGSHOW
    switch (idx) {
    case 0:
        imshow("0", PreparedImgs[idx]);
        break;
    case 1:
        imshow("1", PreparedImgs[idx]);
        break;
    case 2:
        imshow("2", PreparedImgs[idx]);
        break;
    default:
        printf("wrong idx!\n");
        break;
    }
    cv::waitKey();
#endif
    return true;
}

void FishEyeStitcher::Stitch(){

    namedWindow("pano", WINDOW_NORMAL);
    while(startCapture){

#if RUNTIME
        auto start = system_clock::now();
#endif

        for(int i = 0;  i < 3; i++){
            auto flg = pool.submit(std::bind(&FishEyeStitcher::PreProcess, this, i));
            flgs.emplace_back(std::move(flg));
        }
        for(auto & flg: flgs){    //future对象不能复制，所以auto 后面要加& 来表示取引用
            if(flg.wait_for(std::chrono::seconds(1)) == std::future_status::timeout){
                cout << "执行挂起的任务\n";
                pool.run_pending_task();    // 执行挂起的任务
              }
            flg.wait();
          }

#if DEBUGSHOW
        imshow("preparedImgs[0]", PreparedImgs[0]);
        cv::waitKey();
#endif

        int width = PreparedImgs[0].cols + PreparedImgs[1].cols + PreparedImgs[2].cols;
        int height = MAX(MAX(PreparedImgs[0].rows, PreparedImgs[1].rows),PreparedImgs[2].rows);

        if(pano.cols != width){
            printf("pano.cols: %d, target width: %d, reallocate...\n",pano.cols, width);
            pano.create(cv::Size(width, height), CV_8UC3);
            pano.setTo(0);
        }

        PreparedImgs[1].copyTo(pano(Rect(0,0,PreparedImgs[0].cols, PreparedImgs[0].rows)));
        PreparedImgs[0].copyTo(pano(Rect(PreparedImgs[0].cols,0,PreparedImgs[1].cols, PreparedImgs[1].rows)));
        PreparedImgs[2].copyTo(pano(Rect(PreparedImgs[0].cols+PreparedImgs[1].cols,0,PreparedImgs[2].cols, PreparedImgs[2].rows)));

#if STORERESULT
        static int count = 0;
        if(++count >= 10){
            imwrite("cam1_u.jpg", PreparedImgs[0]);
            imwrite("cam2_u.jpg", PreparedImgs[1]);
            imwrite("cam3_u.jpg", PreparedImgs[2]);
            imwrite("pano.jpg", pano);
            break;
        }
#endif

#if RUNTIME
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "one frame total spends "
             << double(duration.count()) * microseconds::period::num / microseconds::period::den
             << "seconds" << endl;
#endif
        imshow("pano", pano);
        cv::waitKey(27);

        flgs.clear();
    }

}
