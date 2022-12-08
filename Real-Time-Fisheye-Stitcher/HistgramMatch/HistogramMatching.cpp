
#include "HistogramMatching.h"
#include <iostream>

#include<chrono>

using namespace std;
using namespace chrono;
using namespace cv;

int maintest(string input, string reference, string output)
{

	cv::Mat inputImage = cv::imread(input);
	cv::Mat referenceImage = cv::imread(reference);
	cv::Mat outputImage;

    auto begin1 = system_clock::now();
	bool result = histogramMatching(inputImage, referenceImage, outputImage);
    auto end1 = system_clock::now();
    auto duration1 = duration_cast<microseconds>(end1 - begin1);
    cout << "one frame histo convert spends "
         << double(duration1.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;

	if (result && cv::imwrite(output, outputImage))
	{
		std::cout << "Match succeeded";
	}
	else
	{
		std::cerr << "Match failed";
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}


void image2Histogram(const cv::Mat& image, float histogram[])
{
	int size = image.rows * image.cols;
	for (int i = 0; i < LEVEL; i++)
	{
		histogram[i] = 0;
	}

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			histogram[(int)image.at<uchar>(y, x)]++;
		}
	}

    // normalize histogram to [0,1]
	for (int i = 0; i < LEVEL; i++)
	{
		histogram[i] = histogram[i] / size;
	}
}

void histogram2CumulativeHistogram(float histogram[], float cumulativeHistogram[])
{
	cumulativeHistogram[0] = histogram[0];
	for (int i = 1; i < LEVEL; i++)
	{
		cumulativeHistogram[i] = histogram[i] + cumulativeHistogram[i - 1];
	}
}

void histogramMatchingChannel(const cv::Mat& inputChannel,
	const cv::Mat& desiredChannel, cv::Mat& outputChannel)
{
	if (inputChannel.channels() != 1 || desiredChannel.channels() != 1) {
		std::cerr << "HistogramMatching Function Error. "
			<< "The Input Image or Desired Image does not have only one channel"
			<< std::endl;

		std::cerr << "Input Image Channels: "
			<< inputChannel.channels() << std::endl;

		std::cerr << "Desired Image Channels: "
			<< desiredChannel.channels() << std::endl;
		return;
	}

	float inputHistogram[LEVEL], inputHistogramCumulative[LEVEL];
	image2Histogram(inputChannel, inputHistogram);
	histogram2CumulativeHistogram(inputHistogram, inputHistogramCumulative);

	float desiredHistogram[LEVEL], desiredHistogramCumulative[LEVEL];
	image2Histogram(desiredChannel, desiredHistogram);
	histogram2CumulativeHistogram(desiredHistogram, desiredHistogramCumulative);

	float outputHistogram[LEVEL];
	for (int i = 0; i < LEVEL; i++)
	{
		int j = 0;
		do {
			outputHistogram[i] = (float)j;
			j++;
		} while (inputHistogramCumulative[i] > desiredHistogramCumulative[j]);
	}

	outputChannel = inputChannel.clone();
	for (int y = 0; y < inputChannel.rows; y++)
	{
		for (int x = 0; x < inputChannel.cols; x++)
		{
			outputChannel.at<uchar>(y, x) =
                MIN(MAX((int)(outputHistogram[inputChannel.at<uchar>(y, x)] + 0.5),0), 255);
		}
	}
}

bool histogramMatching(const cv::Mat& inputImage,
	const cv::Mat& referenceImage, cv::Mat& outputImage)
{
//    std::vector<cv::Mat> inputChannels, referenceChannels, outputChannels;
//    cv::split(inputImage, inputChannels);
//    cv::split(referenceImage, referenceChannels);
//    if (inputChannels.size() != referenceChannels.size())
//    {
//        std::cerr << "Channel of input image isn't the same reference image" << std::endl;
//        return false;
//    }

//    for (int c = 0; c < inputChannels.size(); c++)
//    {
//        cv::Mat output;
//        histogramMatchingChannel(inputChannels[c], referenceChannels[c], output);
//        outputChannels.push_back(output);
//    }
//    cv::merge(outputChannels, outputImage);

    // convert to HSV , only modify V which means light, vary from 0 - 100
    cv::Mat HSV_input, HSV_ref;
    cvtColor(inputImage, HSV_input, COLOR_BGR2HSV);
    cvtColor(referenceImage, HSV_ref, COLOR_BGR2HSV);
    std::vector<cv::Mat> inputChannels, referenceChannels, outputChannels;
    cv::split(HSV_input, inputChannels);
    cv::split(HSV_ref, referenceChannels);
    if (inputChannels.size() != referenceChannels.size())
    {
        std::cerr << "Channel of input image isn't the same reference image" << std::endl;
        return false;
    }


    cv::Mat output;
    histogramMatchingChannel(inputChannels[2], referenceChannels[2], output);

    inputChannels[2] = output;
    cv::merge(inputChannels, outputImage);
    cvtColor(outputImage, outputImage, COLOR_HSV2BGR);

	return true;
}
