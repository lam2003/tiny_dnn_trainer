#ifndef LOCATER_H
#define LOCATER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "cplate.h"
#include "mser2.h"
#include "global.h"
#include "identifier.h"

using namespace cv;
using namespace std;

class Locater
{
public:
    Locater(const Mat &in,bool display_process);
    ~Locater();
    void mserCharLocated();
    inline vector<CPlate> getCPlateVec(){return cplate_vec;}
private:
    vector<CPlate> cplate_vec;
    Mat rgb_mat;
    Mat gray_mat;
    bool display_process;
};

#endif 