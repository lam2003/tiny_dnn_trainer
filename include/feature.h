#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
#include "functions.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::core;
using namespace std;
using namespace cv;

void charFeatureForCNN(const Mat &in,int data_size,double min_v,double max_v,vector<vec_t> &out_vec);

#endif