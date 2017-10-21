#ifndef TRAINER_H
#define TRAINER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "tiny_dnn/tiny_dnn.h"
#include "feature.h"
#include "utils.h"
#include "global.h"
#include "functions.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::core;
using namespace std;
using namespace cv;



class CNNTrainer
{
public:
    CNNTrainer();
    void preprocessTrainData();
    void train();
    void recognize(const Mat &in);
private:
    Mat getSyntheticMat(const Mat &in);
    vector<label_t> train_labels,test_labels;
    vector<vec_t> train_images,test_images;
    network<sequential> net;
    adagrad optimizer;
    const char *sample_path;
    const char *model_path;

};
#endif  