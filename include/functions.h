#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

Rect getCenterRect(const Mat &in);
Mat getRectMat(const Mat &in, const Rect &rect);

//flag == 0,计算每列的均衡化
//flag == 1,计算每行的均衡化
Mat projectedHistogram(const Mat &in, int flag, int thresh = 20);
Mat getTranslatedMat(const Mat &in, float x_offset, float y_offset,int bg_color = 0);
Mat getRotatedMat(const Mat &in, float angle,int bg_color = 0);
Mat getCropMat(const Mat &in,int x,int y,int width,int height);


#endif