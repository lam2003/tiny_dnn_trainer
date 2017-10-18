#ifndef TRAINER_H
#define TRAINER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "functions.h"
#include "utils.h"
#include "global.h"
#include "feature.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

class Trainer
{
public:
    Trainer();
    virtual ~Trainer();
    virtual void train() = 0;
    virtual void test() = 0;
private:
    virtual Ptr<TrainData> preprocessTrainData(int min_input_num) = 0;
};

class ANNTrainer : public Trainer
{
public:
    explicit ANNTrainer(const char *sample_path,const char *xml_path,int type = 0);
    virtual ~ANNTrainer();
    virtual void train();
    virtual void test();
private:
    int identity(const Mat &in);
    int identityChinese(const Mat &in);
    Mat getSyntheticMat(const Mat &in);
    virtual Ptr<TrainData> preprocessTrainData(int min_input_num);
    Ptr<ANN_MLP> ann_ptr;
    const char *sample_path;
    const char *xml_path;
    int type;
};

class ANNChGrayTrainer : public Trainer
{
public:
    explicit ANNChGrayTrainer(const char *sample_path,const char *xml_path);
    virtual ~ANNChGrayTrainer();
    virtual void train();
    virtual void test();
private:
    int getBorderColor(const Mat &in);
    Mat getSyntheticMat(const Mat &in);
    int identityChinese(const Mat &in);
    virtual Ptr<TrainData> preprocessTrainData(int min_input_num);
    Ptr<ANN_MLP> ann_ptr;
    const char *sample_path;
    const char *xml_path;
};

#endif 