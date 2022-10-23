#include "fisheyestitcher.h"
#include<chrono>
using namespace chrono;

#define DEBUGSHOW 0
#define RUNTIME 1
#define STORERESULT 0

FishEyeStitcher::FishEyeStitcher(){
    _isInit = Init();
}

FishEyeStitcher::~FishEyeStitcher(){
}

bool FishEyeStitcher::Init(){
  _startCapture = false;
  _caps[0].open("./cam1.mp4");
  _caps[1].open("./cam2.mp4");
  _caps[2].open("./cam3.mp4");
  Mat img1, img2, img3;
  for(int i = 0; i < 3; i++){
      if(!_caps[i].isOpened()){
          printf("failed to open cam%d!\n", i+1);
          return false;
        }
    }
  if(!_caps[0].read(img1) || !_caps[1].read(img2) || !_caps[2].read(img3)){
      printf("failed to read img!\n");
      return false;
    }
  for(int i = 0; i < 3; i++){
      _ovlps[i] = 300;
    }
  _cut_border = 20;
  return _ud.InitMartix(img1, img2, img3, 1600/2, 5, MIDPOINTCIRCLE);
}

bool FishEyeStitcher::run(){
    if(!_isInit){
        printf("something wrong while init!\n");
        return false;
    }
  _startCapture = true;
  thread t = std::thread(&FishEyeStitcher::Stitch, this);
  if(t.joinable()){
      t.detach();
  }
  printf("success run the stitcher!\n");
  return true;
}

bool FishEyeStitcher::stop(){
  _startCapture = false;
  return true;
}

bool FishEyeStitcher::PreProcess(int idx){
#if RUNTIME
        auto start = system_clock::now();
#endif

    if(!_caps[idx].read(_capImgs[idx])){
        _caps[idx].set(CAP_PROP_POS_AVI_RATIO, 0);
        _caps[idx].read(_capImgs[idx]);
    }

    _ud.MatrixUndistort(_capImgs[idx], _PreparedImgs[idx], idx);
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
        imshow("0", _PreparedImgs[idx]);
        break;
    case 1:
        imshow("1", _PreparedImgs[idx]);
        break;
    case 2:
        imshow("2", _PreparedImgs[idx]);
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
    while(_startCapture){

#if RUNTIME
        auto start = system_clock::now();
#endif

        for(int i = 0;  i < 3; i++){
            auto flg = _pool.submit(std::bind(&FishEyeStitcher::PreProcess, this, i));
            _flgs.emplace_back(std::move(flg));
        }
        for(auto & flg: _flgs){    //future对象不能复制，所以auto 后面要加& 来表示取引用
            if(flg.wait_for(std::chrono::seconds(1)) == std::future_status::timeout){
                cout << "执行挂起的任务\n";
                _pool.run_pending_task();    // 执行挂起的任务
              }
            flg.wait();
          }

#if DEBUGSHOW
        imshow("preparedImgs[0]", _PreparedImgs[0]);
        cv::waitKey();
#endif

        if(_pano.empty()){
            int width = _PreparedImgs[0].cols + _PreparedImgs[1].cols + _PreparedImgs[2].cols - _ovlps[0] - _ovlps[1] - _ovlps[2];
            int height = MAX(MAX(_PreparedImgs[0].rows, _PreparedImgs[1].rows), _PreparedImgs[2].rows);

            printf("pano.cols: %d, target width: %d, reallocate...\n",_pano.cols, width);
            _pano.create(cv::Size(width, height), CV_8UC3);
            _pano.setTo(0);
        }

        Size size1 = _PreparedImgs[1].size();
        Size size2 = _PreparedImgs[0].size();
        Size size3 = _PreparedImgs[2].size();
        _PreparedImgs[1](Rect(size1.width/2, 0,size1.width/2-_cut_border, size1.height))
            .copyTo(_pano(Rect(0,0, size1.width/2-_cut_border, size1.height)));
        _PreparedImgs[0](Rect(_ovlps[0],0,size2.width-_ovlps[0]-_cut_border,size2.height))
            .copyTo(_pano(Rect(size1.width/2-_cut_border,0, size2.width-_ovlps[0]-_cut_border, size2.height)));
        _PreparedImgs[2](Rect(_ovlps[1],0,size3.width-_ovlps[1]-_cut_border,size3.height))
            .copyTo(_pano(Rect(size1.width/2+size2.width-_ovlps[0]-2*_cut_border,0,size3.width-_ovlps[1]-_cut_border, size3.height)));
        _PreparedImgs[1](Rect(_ovlps[2],0,size1.width/2-_ovlps[2],size1.height))
            .copyTo(_pano(Rect(size1.width/2+size2.width-_ovlps[0]+size3.width-_ovlps[1]-3*_cut_border,0,size1.width/2-_ovlps[2],size1.height)));

#if STORERESULT
        static int count = 0;
        if(++count >= 10){
            imwrite("cam1_u.jpg", _PreparedImgs[0]);
            imwrite("cam2_u.jpg", _PreparedImgs[1]);
            imwrite("cam3_u.jpg", _PreparedImgs[2]);
            imwrite("pano.jpg", _pano);
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
        imshow("pano", _pano);
        cv::waitKey(27);

        _flgs.clear();
    }

}
