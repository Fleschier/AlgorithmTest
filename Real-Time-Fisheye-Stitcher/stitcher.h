#ifndef STITCHER_H
#define STITCHER_H

#include "opencv2/opencv.hpp"
#include "ThreadPool/thread_pool.h"
#include <atomic>
#include <mutex>

#define MAXQUEUELEN 10

#ifndef TESTGENERATE
#define TESTGENERATE 1
#endif

class ReaderWriterLock: public std::mutex{
private:
    std::mutex _mut;
    std::condition_variable _cond;
    int _readWaiting = 0;
    int _writeWaiting = 0;
    int _reading = 0;
    int _writing = 0;
    bool _prefer_writer = false;    // prefer reader
public:
    ReaderWriterLock(bool isPreferWriter):_prefer_writer(isPreferWriter){}

    void ReadLock(){
        std::unique_lock<std::mutex> lock(_mut);
        ++_readWaiting;
        _cond.wait(lock, [&](){return _writing <= 0 && (!_prefer_writer || _writeWaiting <= 0);});
        ++_reading;
        --_readWaiting;
    }

    void WriteLock(){
        std::unique_lock<std::mutex> lock(_mut);
        ++_writeWaiting;
        _cond.wait(lock, [&](){return _reading <= 0 && _writing <= 0;});
        ++_writing;
        --_writeWaiting;
    }

    void ReadUnlock(){
        std::unique_lock<std::mutex> lock(_mut);
        --_reading;
        // if no reader remained, awake a writer
        if(_reading <= 0){
            _cond.notify_one();
        }
    }

    void WriterUnlock(){
        std::unique_lock<std::mutex> lock(_mut);
        --_writing;
        // awake all reader, writer
        _cond.notify_all();
    }

};

class FishEyeStitcher
{
private:
    thread_pool _mpool;
    static uint current_idx;
    cv::Rect rc_l, rc_r;    // valid rectangular area of raw fisheye img (dule fisheye)
    cv::Mat _valid_area_mask;   // valid circle area mask of cutted single fisheye
    cv::Mat _xMapArr_l, _yMapArr_l;
    cv::Mat _xMapArr_r, _yMapArr_r;
    cv::Mat _pano;
    cv::Mat _unwarp_img_l, _unwarp_img_r;
    threadsafe_queue<cv::Mat> _unwarped_l, _unwarped_r;
    int _overlap;
#if TESTGENERATE
    cv::Mat _pano_mask;
    cv::Mat _blend_mask_l, _blend_mask_r;
    int _band_num;
#endif
    cv::Mat _dul_fish_img;
    std::mutex _dulfish_mut_l, _dulfish_mut_r;    // simulate read-write lock
    std::condition_variable _cap_cond;
    std::atomic_bool _cap_ready, pre_l_ready, pre_r_ready;
    std::condition_variable _preprocess_cond;
    std::condition_variable _fuse_cond;

    cv::Mat _scale_map;

    int _divid_idx_l = 1896/2;
    int _divid_idx_r = 1896+1900/2;
    int _img_l_w = 1896/2+1900/2;

private:
    // param: position: 0 ==> left, 1 ==> right
    bool __optimizeSeam(cv::Mat& img1, int begin1, cv::Mat& img2, int begin2, cv::Mat& pano, cv::Mat& roi, int ProcessWidth, int position);
    bool __optimizeSeam(cv::Mat& img1, int begin1, cv::Mat& img2, int begin2, cv::Mat& pano, int ProcessWidth1, int ProcessWidth2);
    bool __captureThread();
    // param: area_idx: 0 ==> left_fisheye ; 1 ==> right_fisheye
    bool __preProcessThread(int area_idx);
    // Light Fall-off Compensation
    void __compenLightFO(cv::Mat& imgin, cv::Mat& imgout);
    void __genScaleMap();
    // dynamic real-time light fall-off compensatoin function
    void __compenLightFallOff(cv::Rect FixArea, cv::Rect TemplateArea_l, cv::Rect TemplateArea_r);
    bool __histoConvert(cv::Mat& img_l, cv::Mat& img_r);
    bool __histoConvert_multiThread(cv::Mat& img_l, cv::Mat& img_r);
    bool __histo_tool_func(cv::Mat& inputImage, cv::Rect& rc_in, cv::Rect& rc_tpl, bool isDescend);
    void __blockHistoConvert(cv::Mat& img_l, cv::Mat& img_r);
    void __globalLightCompo(cv::Mat& img_l, cv::Mat& img_r);

    void __biasFuseAscend(cv::Mat& img1, cv::Mat& img2, cv::Rect rc1, cv::Rect rc2);
    void __biasFuseDescend(cv::Mat& img1, cv::Mat& img2, cv::Rect rc1, cv::Rect rc2);
public:
    FishEyeStitcher();
    bool Init();
#if TESTGENERATE
    void TestGenerate(int type = 0);
    void TestGenerateVideo();
#endif
};

#endif // STITCHER_H
