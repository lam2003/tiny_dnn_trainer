#include "feature.h"

Mat charFeaturesForANNTrain(const Mat &in, int data_size)
{
    const int VERTICAL = 0;
    const int HORIZONTAL = 1;

    Rect center_rect = getCenterRect(in);
    Mat center_mat = getRectMat(in, center_rect);

    Mat low_data_mat;
    resize(in, low_data_mat, Size(data_size, data_size));

    Mat v_hist = projectedHistogram(low_data_mat, VERTICAL);
    Mat h_hist = projectedHistogram(low_data_mat, HORIZONTAL);

    int total_cols = v_hist.cols + h_hist.cols + data_size * data_size;

    Mat out = Mat::zeros(1, total_cols, CV_32F);

    int j = 0;
    for (int i = 0; i < v_hist.cols; i++)
    {
        out.at<float>(j) = v_hist.at<float>(i);
        j++;
    }
    for (int i = 0; i < h_hist.cols; i++)
    {
        out.at<float>(j) = h_hist.at<float>(i);
        j++;
    }
    for (int c = 0; c < low_data_mat.cols; c++)
    {
        for (int r = 0; r < low_data_mat.rows; r++)
        {
            out.at<float>(j) += float(low_data_mat.at<unsigned char>(c, r));
            j++;
        }
    }

    return out;
}

Mat getProjectedMat(const Mat &in,int data_size)
{
    const int VERTICAL = 0;
    const int HORIZONTAL = 1;
    Mat low_data_mat;

    resize(in,low_data_mat,Size(data_size,data_size));
    Mat v_hist = projectedHistogram(low_data_mat,VERTICAL);
    Mat h_hist = projectedHistogram(low_data_mat,HORIZONTAL);

    int total_cols = v_hist.cols + h_hist.cols;
   
    Mat out = Mat::zeros(1,total_cols,CV_32FC1);
    int j = 0;
    for (int i = 0; i < v_hist.cols; i++)
    {
        out.at<float>(j) = v_hist.at<float>(i);
        j++;
    }
    for (int i = 0; i < h_hist.cols; i++)
    {
        out.at<float>(j) = h_hist.at<float>(i);
        j++;
    }
    return out;
    
}

Mat charFeaturesForANNChGrayTrain(const Mat &in)
{
   
    Mat char_mat = in.clone();
   
    float scale = 1.f / 255.f;
    char_mat.convertTo(char_mat,CV_32FC1,scale,0);
    char_mat -= mean(char_mat);
    char_mat = char_mat.reshape(1,1);

    Mat binary_mat;
    threshold(in,binary_mat,0,255,CV_THRESH_OTSU + CV_THRESH_BINARY);
  
    Mat projected_mat = getProjectedMat(binary_mat,kANNChGrayDataSize);
    
    Mat feature_mat;
    hconcat(char_mat,projected_mat.reshape(1,1),feature_mat); 
    return feature_mat;
}