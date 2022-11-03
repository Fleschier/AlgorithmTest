#ifndef MULTIBANDBLENDERTEST_H
#define MULTIBANDBLENDERTEST_H

#include<opencv2/opencv.hpp>
#include<vector>

class MultiBandBlenderTest
{
public:
  MultiBandBlenderTest();
  static void MultiBandBlend(cv::Mat img1, cv::Mat mask1, cv::Mat img2, cv::Mat mask2, cv::Rect ROI, cv::Mat& res, cv::Mat res_mask);
};

// 图像金字塔信息对象
class CImagePyramidInfo
{
private:
    cv::Mat m_matMeanFocus;  // 平均梯度网格矩阵
    std::vector<cv::Mat> m_vecGaussian;  // 高斯金字塔
    std::vector<cv::Mat> m_vecLaplac;  // 拉普拉斯金字塔
    std::vector<cv::Mat> m_vecMask;  // 掩码金字塔

public:
    CImagePyramidInfo(const cv::Mat& matImage, int nGradSize = 100)
    {
        __makeMeanFocus(matImage, m_matMeanFocus, nGradSize);

        cv::Mat matImageFloat;
        matImage.convertTo(matImageFloat, CV_32F, 1.0 / 255);

        __makeGaussian(matImageFloat, m_vecGaussian, 5);
        __makeLaplac(m_vecGaussian, m_vecLaplac);
    }
    virtual ~CImagePyramidInfo()
    {
    }

    // 构建掩码金字塔
    void makeMask(const cv::Mat& matMask)
    {
        __makeMask(matMask, m_vecMask, m_vecGaussian.size());
    }

    // 获取平均梯度网格矩阵
    const cv::Mat& getMeanFocus() const
    {
        return m_matMeanFocus;
    }

    // 获取高斯金字塔
    const std::vector<cv::Mat>& getGaussian() const
    {
        return m_vecGaussian;
    }

    // 获取拉普拉斯金字塔
    const std::vector<cv::Mat>& getLaplac() const
    {
        return m_vecLaplac;
    }

    // 获取掩码金字塔
    const std::vector<cv::Mat>& getMask() const
    {
        return m_vecMask;
    }

private:

    // 构建平均梯度网格矩阵
    static void __makeMeanFocus(const cv::Mat& matImage, cv::Mat& matMeanFocus, int nGradSize = 100)
    {
        // 只接受CV_8UC3和CV_8UC1格式的图像
        if (matImage.type() != CV_8UC3 && matImage.type() != CV_8UC1)
            return;

        // 计算网格数
        int nGradRows = matImage.rows / nGradSize;
        if (matImage.rows % nGradSize)
            ++nGradRows;
        int nGradCols = matImage.cols / nGradSize;
        if (matImage.cols % nGradSize)
            ++nGradCols;

        matMeanFocus = cv::Mat(nGradRows, nGradCols, CV_32FC1, cv::Scalar::all(0.0));

        // 计算每个网格的平均梯度
        for (int nRow = 0; nRow < nGradRows; ++nRow)
        {
            for (int nCol = 0; nCol < nGradCols; ++nCol)
            {
                int nOffsetX = nCol * nGradSize;
                int nOffsetY = nRow * nGradSize;
                int nWidth = nOffsetX + nGradSize > matImage.cols ? matImage.cols - nOffsetX : nGradSize;
                int nHeight = nOffsetY + nGradSize > matImage.rows ? matImage.rows - nOffsetY : nGradSize;

                cv::Mat matImageROISrc = matImage(cv::Rect(nOffsetX, nOffsetY, nWidth, nHeight));
                cv::Mat matImageROIDst;

                // 将BGR转换为灰度图像
                if (matImage.type() == CV_8UC3)
                {
                    matImageROIDst = cv::Mat(matImageROISrc.size(), CV_8UC1);
                    cv::cvtColor(matImageROISrc, matImageROIDst, cv::COLOR_BGR2GRAY);
                }
                else
                {
                    matImageROIDst = matImageROISrc.clone();
                }

                cv::GaussianBlur(matImageROIDst, matImageROIDst, cv::Size(3, 3), 0, 0);
                cv::Mat matImageROISobel;
                cv::Sobel(matImageROIDst, matImageROISobel, CV_8UC1, 1, 1, 3);

                matMeanFocus.at<float>(nRow, nCol) = cv::mean(matImageROISobel)[0];
            }
        }
    }

    // 构建高斯金字塔
    static void __makeGaussian(const cv::Mat& matImage, std::vector<cv::Mat>& vecGaussian, int nLayerCount = 5)
    {
        vecGaussian.resize(nLayerCount);

        for (int nLayerIndex = 0; nLayerIndex < nLayerCount; ++nLayerIndex)
        {
            if (nLayerIndex == 0)
            {
                vecGaussian[nLayerIndex] = matImage.clone();
            }
            else
            {
                cv::Mat matImageDown;
                cv::pyrDown(vecGaussian[nLayerIndex - 1], matImageDown);
                vecGaussian[nLayerIndex] = matImageDown;
            }
        }
    }

    // 构建拉普拉斯金字塔
    static void __makeLaplac(const std::vector<cv::Mat>& vecGaussian, std::vector<cv::Mat>& vecLaplac)
    {
        vecLaplac.resize(vecGaussian.size());

        for (int nLayerIndex = vecLaplac.size() - 1; nLayerIndex >= 0; --nLayerIndex)
        {
            if (nLayerIndex == vecLaplac.size() - 1)
            {
                vecLaplac[nLayerIndex] = vecGaussian[nLayerIndex].clone();
            }
            else
            {
                cv::Mat matImageUp;
                cv::pyrUp(vecGaussian[nLayerIndex + 1], matImageUp, vecGaussian[nLayerIndex].size());
                vecLaplac[nLayerIndex] = vecGaussian[nLayerIndex] - matImageUp;
            }
        }
    }

    // 构建掩码金字塔
    static void __makeMask(const cv::Mat& matMask, std::vector<cv::Mat>& vecMask, int nLayerCount = 5)
    {
        vecMask.resize(nLayerCount);

        for (int nLayerIndex = 0; nLayerIndex < vecMask.size(); ++nLayerIndex)
        {
            if (nLayerIndex == 0)
            {
                vecMask[nLayerIndex] = matMask.clone();
            }
            else
            {
                cv::Mat matMaskDown;
                cv::pyrDown(vecMask[nLayerIndex - 1], matMaskDown);
                vecMask[nLayerIndex] = matMaskDown;
            }
        }
    }
};

// 多聚焦多频段图像融合算法
static bool multiFocusBandBlender(const std::vector<cv::Mat>& vecImage, cv::Mat& matResult, int nGradSize = 100)
{
    if (vecImage.size() < 2)
        return false;

    // 检查每张图片的大小和格式是否一致
    for (int i = 1; i < vecImage.size(); ++i)
    {
        if (vecImage[i].size() != vecImage[i - 1].size())
            return false;

        if (vecImage[i].type() != vecImage[i - 1].type())
            return false;
    }

    // 检查每张图片的格式是否为CV_8UC3和CV_8UC1格式
    for (int i = 0; i < vecImage.size(); ++i)
    {
        if (vecImage[i].type() != CV_8UC3 && vecImage[i].type() != CV_8UC1)
            return false;
    }

    // 为每张图像建立高斯和拉普拉斯金字塔

    std::vector<CImagePyramidInfo*> vecImagePyramidInfo(vecImage.size());
    for (int i = 0; i < vecImage.size(); ++i)
    {
        vecImagePyramidInfo[i] = new CImagePyramidInfo(vecImage[i], nGradSize);
    }

    // 合成聚焦网格矩阵

    std::vector<cv::Mat> vecMeanFocus(vecImagePyramidInfo.size());
    std::vector<cv::Mat> vecBestIndex(vecImagePyramidInfo.size());
    for (int i = 0; i < vecImagePyramidInfo.size(); ++i)
    {
        vecMeanFocus[i] = vecImagePyramidInfo[i]->getMeanFocus();
        vecBestIndex[i] = cv::Mat(vecMeanFocus[i].size(), CV_8UC1, cv::Scalar::all(0));
    }

    for (int nRow = 0; nRow < vecMeanFocus[0].rows; ++nRow)
    {
        for (int nCol = 0; nCol < vecMeanFocus[0].cols; ++nCol)
        {
            // 找梯度最大图片
            float fMax = -1.0;
            int nIndex = -1;
            for (int i = 0; i < vecMeanFocus.size(); ++i)
            {
                float fMeanFocus = vecMeanFocus[i].at<float>(nRow, nCol);
                if (fMeanFocus > fMax)
                {
                    fMax = fMeanFocus;
                    nIndex = i;
                }
            }

            // 标记该图片当前网格为选中
            vecBestIndex[nIndex].at<uchar>(nRow, nCol) = 1;
        }
    }

    // 构建掩码金字塔

    std::vector<cv::Mat> vecMaskPyramid(vecBestIndex.size());

    for (int i = 0; i < vecBestIndex.size(); ++i)
    {
        vecMaskPyramid[i] = cv::Mat(vecImage[i].size(), (vecImage[i].channels() == 1 ? CV_32FC1 : CV_32FC3), cv::Scalar::all(0.0));

        for (int nRow = 0; nRow < vecBestIndex[i].rows; ++nRow)
        {
            for (int nCol = 0; nCol < vecBestIndex[i].cols; ++nCol)
            {
                if (vecBestIndex[i].at<uchar>(nRow, nCol) == 1)
                {
                    int nOffsetX = nCol * nGradSize;
                    int nOffsetY = nRow * nGradSize;
                    int nWidth = nOffsetX + nGradSize > vecImage[0].cols ? vecImage[0].cols - nOffsetX : nGradSize;
                    int nHeight = nOffsetY + nGradSize > vecImage[0].rows ? vecImage[0].rows - nOffsetY : nGradSize;

                    cv::Mat matROI = vecMaskPyramid[i](cv::Rect(nOffsetX, nOffsetY, nWidth, nHeight));
                    matROI.setTo(cv::Scalar::all(1.0));
                }
            }
        }
    }

    for (int i = 0; i < vecImagePyramidInfo.size(); ++i)
    {
        vecImagePyramidInfo[i]->makeMask(vecMaskPyramid[i]);
    }

    // 逐层融合拉普拉斯金字塔

    // 层次遍历
    for (int nLayerIndex = vecImagePyramidInfo[0]->getLaplac().size() - 1; nLayerIndex >= 0; --nLayerIndex)
    {
        cv::Mat matBlendLayer;

        // 图片遍历
        for (int i = 0; i < vecImagePyramidInfo.size(); ++i)
        {
            cv::Mat matLaplac = vecImagePyramidInfo[i]->getLaplac()[nLayerIndex];
            cv::Mat matMask = vecImagePyramidInfo[i]->getMask()[nLayerIndex];

            if (matBlendLayer.empty())
            {
                matBlendLayer = matLaplac.mul(matMask);
            }
            else
            {
                matBlendLayer += matLaplac.mul(matMask);
            }
        }

        if (matResult.empty())
        {
            matResult = matBlendLayer;
        }
        else
        {
            cv::pyrUp(matResult, matResult, matBlendLayer.size());
            matResult += matBlendLayer;
        }
    }

    cv::convertScaleAbs(matResult, matResult, 255);

    for (auto iter = vecImagePyramidInfo.begin(); iter != vecImagePyramidInfo.end(); ++iter)
    {
        delete (*iter);
        (*iter) = NULL;
    }

    return true;
}


void blendTest(cv::Mat& img1, cv::Rect roi1, cv::Mat& img2, cv::Rect roi2, cv::Mat& dst, cv::Rect dst_roi){
  const int nGradSize = 100;

  std::vector<cv::Mat> vecImage;
  vecImage.push_back(img1(roi1));
  vecImage.push_back(img2(roi2));
  cv::Mat matResult;
//  dst(dst_roi).copyTo(matResult);
  multiFocusBandBlender(vecImage, matResult, nGradSize);
//  matResult.copyTo(dst(dst_roi));

  cv::imwrite("full2_result.jpg", matResult);
}

void MultiBlend(cv::Mat img1, cv::Mat img2){
  cv::Mat img1s, img2s;
  img1.convertTo(img1s, CV_16S);
  img2.convertTo(img2s, CV_16S);

  cv::Mat mask1, mask2;
  mask1 = cv::Mat::zeros(img1s.size(), CV_8U);
  mask1.setTo(255);
//  mask1(cv::Rect(0,0,mask1.cols, mask1.rows/2)).setTo(255);
  mask2 = cv::Mat::zeros(img2s.size(), CV_8U);
//  mask2(cv::Rect(0,0,mask2.cols, mask2.rows/2)).setTo(255);
  mask2.setTo(255);
  cv::detail::MultiBandBlender blender( false, 5);

  blender.prepare(cv::Rect(0, 0, MAX(img1s.cols, img2s.cols), MAX(img1s.rows, img2s.rows)));
  blender.feed(img1s, mask1, cv::Point(0,0));
  blender.feed(img2s, mask2, cv::Point(0,0));

  cv::Mat result_s, result_mask;
  blender.blend(result_s, result_mask);
  cv::Mat result;
  result_s.convertTo(result, CV_8U);

//  cv::imshow( "result",result);
  cv::imwrite( "baboon_lena.jpg",result);
//  cv::waitKey();
}

#endif // MULTIBANDBLENDERTEST_H
