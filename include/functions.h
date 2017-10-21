#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;


Rect getCenterRect(const Mat &in);
Mat getRectMat(const Mat &in, const Rect &rect);
Rect getSafeRect(const Mat &in,const Point2f &center_point,float width,float height);
Mat getTranslatedMat(const Mat &in, float x_offset, float y_offset,int bg_color = 0);
Mat getRotatedMat(const Mat &in, float angle,int bg_color = 0);
Mat getCropMat(const Mat &in,int x,int y,int width,int height);
Mat preprocessChar(const Mat &in, int char_size);



#endif