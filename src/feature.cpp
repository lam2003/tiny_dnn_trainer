#include "feature.h"

void charFeatureForCNN(const Mat &in,int data_size,double min_v,double max_v,vector<vec_t> &out_vec)
{
    
    Mat feature_mat = preprocessChar(in,20);
    Rect center_rect = getCenterRect(feature_mat);
    feature_mat = getRectMat(feature_mat, center_rect);

    Mat_<uint8_t> mat;
    resize(feature_mat,mat,Size(data_size,data_size));

    vec_t out;
    transform(mat.begin(),mat.end(),back_inserter(out),[=](uint8_t c) { return (255 - c) * (max_v - min_v) / 255.0 + min_v; });
    out_vec.push_back(out);
}