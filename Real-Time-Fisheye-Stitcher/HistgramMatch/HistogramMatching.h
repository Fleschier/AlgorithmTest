
#ifndef HISTOGRAMMATCH_H
#define HISTOGRAMMATCH_H

//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv.hpp"
#include<string>

#define LEVEL 256

void image2Histogram(const cv::Mat& image, float histogram[]);
void histogram2CumulativeHistogram(float histogram[],
    float cumulativeHistogram[]);

void histogramMatchingChannel(const cv::Mat& inputChannel,
    const cv::Mat& desiredChannel, cv::Mat& outputChannel);

bool histogramMatching(const cv::Mat& inputImage,
    const cv::Mat& desiredImage, cv::Mat& outputImage);

int maintest(std::string input, std::string reference, std::string output);

#endif //HISTOGRAMMATCH_H
