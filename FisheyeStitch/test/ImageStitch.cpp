#include"ImageStitch.h"

#include <fstream>
#include <string>
#include<iostream>
//#include "opencv2/opencv_modules.hpp"
//#include <opencv2/core/utility.hpp>
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/stitching/detail/autocalib.hpp"
//#include "opencv2/stitching/detail/blenders.hpp"
//#include "opencv2/stitching/detail/timelapsers.hpp"
//#include "opencv2/stitching/detail/camera.hpp"
//#include "opencv2/stitching/detail/exposure_compensate.hpp"
//#include "opencv2/stitching/detail/matchers.hpp"
//#include "opencv2/stitching/detail/motion_estimators.hpp"
//#include "opencv2/stitching/detail/seam_finders.hpp"
//#include "opencv2/stitching/detail/warpers.hpp"
//#include "opencv2/stitching/warpers.hpp"
//// #include <opencv2/nofree/nofree.hpp>
#include<opencv2/xfeatures2d.hpp>

#include<opencv2/opencv.hpp>

#include<stdlib.h> // itoa atoi

#include<chrono>

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace chrono;



bool readCamera(const string& filename, Mat& cameraMatrix, Mat& distCoeffs,  float& ratio);

int fullStitchTest()
{
  string outputPrefix = "./DebugOutput/";

    vector<Mat> imgs;
//    ifstream fin("../img.txt");
//    string img_name;
//    while(getline(fin, img_name))
//    {
//        Mat img = imread(img_name);
//        //resize(img, img, Size(), 0.25, 0.25);
//        imgs.push_back(img);
//    }

//    imgs.push_back(imread("/home/cyx/programes/Pictures/l0001.jpg"));
//    imgs.push_back(imread("/home/cyx/programes/Pictures/r0001.jpg"));

//    imgs.push_back(imread("/home/cyx/programes/C++/fisheye/extract_frames/fullcam1_unwarp.jpg"));
////    imgs.push_back(imread("/home/cyx/programes/C++/fisheye/extract_frames/fullcam2_unwarp.jpg"));
//    imgs.push_back(imread("/home/cyx/programes/C++/fisheye/extract_frames/fullcam3_unwarp.jpg"));

    imgs.push_back(imread("/home/cyx/programes/Pictures/123.jpg"));
    imgs.push_back(imread("/home/cyx/programes/Pictures/456.jpg"));

    int num_images = imgs.size();    //图像数量
    cout<<"图像数量为"<<num_images<<endl;
    cout<<"图像读取完毕"<<endl;
    // ----------------------- extract features ----------------------------
    Ptr<FeaturesFinder> finder;    //定义特征寻找器
//    finder = new SurfFeaturesFinder();    //应用SURF方法寻找特征
    //finder = new OrbFeaturesFinder();    //应用ORB方法寻找特征
    finder = new SiftFeaturesFinder();
    vector<ImageFeatures> features(num_images);    //表示图像特征
    for (int i =0 ;i<num_images;i++){
        (*finder)(imgs[i], features[i]);    //特征检测
        Mat img_keypoints;
        drawKeypoints( imgs[i], features[i].keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        char filename[20];
        sprintf(filename, "keypoints%d.jpg", i);
        imwrite(outputPrefix+string(filename), img_keypoints);
      }
    cout<<"特征提取完毕"<<endl;

    // ----------------------- match features -------------------------------
    vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
    BestOf2NearestMatcher matcher(false, 0.6f, 6, 6);    //定义特征匹配器，2NN方法
//    AffineBestOf2NearestMatcher matcher(false, false, 0.3f, 6);
    matcher(features, pairwise_matches);    //进行特征匹配
    printf("pairwise_matches size: %d\n", pairwise_matches.size());
    /*打印图像之间的匹配关系匹配*/
    for(size_t i=0; i<num_images; i++)
        for(size_t j=0; j<num_images; j++)
        {
            if(pairwise_matches.at(i*num_images+j).H.empty())
                continue;
            cout<<"第"<<i<<"匹配"<<j<<"幅图片置信度为"
                <<pairwise_matches[i*num_images+j].confidence<<endl;
            cout << "Homography: " << pairwise_matches[i*num_images+j].H << endl;

            try{
              Mat matchInfoImg;
//              printf("try to draw matches...\n");
              drawMatches(imgs[i], features[i].keypoints, imgs[j], features[j].keypoints, pairwise_matches[i*num_images+j].matches, matchInfoImg);
              char filename[50];
              sprintf(filename, "matchInfoImg%d%d.jpg", i, j);
              imwrite(outputPrefix+string(filename), matchInfoImg);
              matchInfoImg.release();
            }
            catch(...){
              printf("draw matchs failed!\n");
              continue;
            }

        }
    cout<<"特征匹配完毕"<<endl;

    // ------------------------ camera param estimate -----------------------
    HomographyBasedEstimator estimator;    //定义参数评估器
//    AffineBasedEstimator estimator;
    vector<CameraParams> cameras;    //表示相机参数，内参加外参
    estimator(features, pairwise_matches, cameras);    //进行相机参数评估

    for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }
    cout<<"相机参数预测完毕"<<endl;

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout<<"第"<<i<<"焦距为"<<cameras[i].focal<<endl;
    }


    // -------------------------- Bundle Adjust ------------------------------
    // 在一部可以计算重映射误差，想办法让他可以输出出来
    Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
    //adjuster->setRefinementMask();
    adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
    //adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

    adjuster->setConfThresh(0.6f);    //设置匹配置信度，该值设为1
    (*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数


    /*查看进行光束平差法之后的树节点数量和核心节点位置*/
//    const int node_number = static_cast<int>(features.size());
//    cout<<"树节点数量"<<node_number<<endl;
//    //Graph span_tree;
//    //std::vector<int> span_tree_centers;
//    findMaxSpanningTree(node_number, pairwise_matches, span_tree, span_tree_centers);
//    for(size_t i=0; i<span_tree_centers.size(); i++)
//        cout<<"核心节点包括"<<span_tree_centers[i]<<endl;


    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)    //复制相机的旋转参数
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //进行波形校正
    for (size_t i = 0; i < cameras.size(); ++i)    //相机参数赋值
        cameras[i].R = rmats[i];
    rmats.clear();    //清变量

    cout<<"利用光束平差法进行相机矩阵更新"<<endl;

    // ------------------------ remap ----------------------------------------
    vector<Point> corners(num_images);    //表示映射变换后图像的左上角坐标
    vector<UMat> masks_warped(num_images);    //表示映射变换后的图像掩码
    vector<UMat> images_warped(num_images);    //表示映射变换后的图像
    vector<Size> sizes(num_images);    //表示映射变换后的图像尺寸
    vector<UMat> masks(num_images);    //表示源图的掩码

    for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
    {
        masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
        masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
    }

    Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
//    warper_creator = new cv::SphericalWarper();
    warper_creator = makePtr<cv::PlaneWarper>();     //平面投影
//    warper_creator = new cv::CylindricalWarper();    //柱面投影
//    warper_creator = new cv::SphericalWarper();    //球面投影
//    warper_creator = new cv::FisheyeWarper();    //鱼眼投影
//    warper_creator = new cv::StereographicWarper();    //立方体投影

    //定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
    vector<double> focals;

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout<<"第"<<i<<"焦距为"<<cameras[i].focal<<endl;
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    cout<<"最终选择的图像的焦距为"<<warped_image_scale<<endl;
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));

    auto remap_start = system_clock::now();

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
        //对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
//        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, images_warped[i]);
        cout << "warped corners[" << i <<"]: " << corners[i] << endl;
        sizes[i] = images_warped[i].size();    //得到尺寸
        cout<<"width:    "<<sizes[i].width<<"height:   "<<sizes[i].height<<endl;
        //得到变换后的图像掩码
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

        try{
          char filename[50];
          sprintf(filename, "images_warped[%d].jpg", i);
          imwrite(outputPrefix+string(filename), images_warped[i]);
          sprintf(filename, "masks[%d].jpg", i);
          imwrite(outputPrefix+string(filename), masks[i]);
          sprintf(filename, "masks_warped[%d].jpg", i);
          imwrite(outputPrefix+string(filename), masks_warped[i]);
        }
        catch(...){
          printf("write img failed!\n");
        }

    }

    imgs.clear();    //清变量
    masks.clear();
    cout<<"图像映射完毕"<<endl;

    auto remap_end = system_clock::now();
    auto remap_duration = duration_cast<microseconds>(remap_end - remap_start);
    cout << "remap spends "
         << double(remap_duration.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    // ---------------------------- exposure compensation ---------------
    //创建曝光补偿器，应用增益补偿方法
    auto exposure_start = system_clock::now();
    Ptr<ExposureCompensator> compensator =
            ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    compensator->feed(corners, images_warped, masks_warped);    //得到曝光补偿器
    for(int i=0;i<num_images;++i)    //应用曝光补偿器，对图像进行曝光补偿
    {
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
    }
    cout<<"图像曝光完毕"<<endl;

    auto exposure_end = system_clock::now();
    auto exposure_duration = duration_cast<microseconds>(exposure_end - exposure_start);
    cout << "exposure spends "
         << double(exposure_duration.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    //在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个副本masks_seam
    // ----------------------------- seam finder -------------------------
    vector<UMat> masks_seam(num_images);
    for(int i = 0; i<num_images;i++)
        masks_warped[i].copyTo(masks_seam[i]);

    Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
    //seam_finder = new NoSeamFinder();    //无需寻找接缝线
    //seam_finder = new VoronoiSeamFinder();    //逐点法
    seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
    //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    //图割法
    //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
    //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);

    auto seam_start = system_clock::now();

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)    //图像数据类型转换
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    images_warped.clear();    //清内存
    //得到接缝线的掩码图像masks_seam
    seam_finder->find(images_warped_f, corners, masks_seam);

    Mat tmpImg;
    for (int i=0; i<masks_seam.size(); i++)
    {
        masks_seam[i].copyTo(tmpImg);
        imwrite(outputPrefix + "masks_seam" + to_string(i) + ".jpg", tmpImg);
    }
    tmpImg.release();

    cout<<"拼缝优化完毕"<<endl;

    auto seam_end = system_clock::now();
    auto seam_duration = duration_cast<microseconds>(seam_end - seam_start);
    cout << "seam finder spends "
         << double(seam_duration.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    // --------------------------- blender -------------------------------------
    vector<Mat> images_warped_s(num_images);
    Ptr<Blender> blender;    //定义图像融合器

    //blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
    //羽化融合方法
//    blender = Blender::createDefault(Blender::FEATHER, false);
//    //dynamic_cast多态强制类型转换时候使用
//    FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
//    fb->setSharpness(0.005);    //设置羽化锐度

    blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
    mb->setNumBands(8);   //设置频段数，即金字塔层数

    blender->prepare(corners, sizes);    //生成全景图像区域
    for(auto& c: corners){
        cout << c << endl;
      }
    for(auto& s: sizes){
        cout << s << endl;
      }
    cout<<"生成全景图像区域"<<endl;
    //在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，因此我们还需要在接缝线两侧开辟一块区域用于融合处理，这一处理过程对羽化方法尤为关键
    //应用膨胀算法缩小掩码面积
    vector<Mat> dilate_img(num_images);
    vector<Mat> masks_seam_new(num_images);
    Mat tem;
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素

    for(int k=0;k<num_images;k++)
    {
        images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
        dilate(masks_seam[k], masks_seam_new[k], element);    //膨胀运算

        imwrite(outputPrefix+"mask_seam_new"+to_string(k)+".jpg", masks_seam_new[k]);

        //映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
        //resize(dilated_mask, tem, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        masks_warped[k].copyTo(tem);
        masks_seam_new[k] = masks_seam_new[k] & tem;

        imwrite(outputPrefix+"mask_seam_new_&_masks_warped"+to_string(k)+".jpg", masks_seam_new[k]);

//        masks_seam_new[k].copyTo(tmpImg);
//        imwrite(to_string(k) + "masks_seam_new.jpg", masks_seam_new[k]);
//        namedWindow("mask_seam_new", WINDOW_NORMAL);
//        imshow("mask_seam_new", masks_seam_new[k]);
//        waitKey(0);

        blender->feed(images_warped_s[k], masks_seam_new[k], corners[k]);    //初始化数据
        cout << "corners[" << k << "]: " << corners[k] << endl;
        cout<<"处理完成"<<k<<"图片"<<endl;
    }

    masks_seam.clear();    //清内存

    images_warped_s.clear();

    masks_warped.clear();

    images_warped_f.clear();


    Mat result, result_mask;
    //完成融合操作，得到全景图像result和它的掩码result_mask


    auto blend_start = system_clock::now();

    blender->blend(result, result_mask);

    auto blend_end = system_clock::now();
    auto blend_duration = duration_cast<microseconds>(blend_end - blend_start);
    cout << "blender spends "
         << double(blend_duration.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

    imwrite(outputPrefix + "result_mask.jpg", result_mask);
    imwrite(outputPrefix + "result.jpg", result);    //存储全景图像

    return 0;
}
