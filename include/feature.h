#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>
#include "functions.h"
#include "global.h"
using namespace cv;

Mat charFeaturesForANNTrain(const Mat &in, int data_size);
Mat getProjectedMat(const Mat &in,int data_size);
Mat charFeaturesForANNChGrayTrain(const Mat &in);

#endif