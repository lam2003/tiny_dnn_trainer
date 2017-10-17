#include "functions.h"

Rect getCenterRect(const Mat &in)
{
    int top = 0;
    int bottom = in.rows - 1;
    for (int i = 0; i < in.rows; i++)
    {
        bool is_found = false;
        for (int j = 0; j < in.cols; j++)
        {
            if (in.data[i * in.step[0] + j] > 20)
            {
                top = i;
                is_found = true;
                break;
            }
        }

        if (is_found)
        {
            break;
        }
    }
    for (int i = in.rows - 1; i >= 0; i--)
    {
        bool is_found = false;
        for (int j = 0; j < in.cols; j++)
        {
            if (in.data[i * in.step[0] + j] > 20)
            {
                bottom = i;
                is_found = true;
                break;
            }
        }

        if (is_found)
        {
            break;
        }
    }

    int left = 0;
    int right = in.cols - 1;
    for (int j = 0; j < in.cols; j++)
    {
        bool is_found = false;
        for (int i = 0; i < in.rows; i++)
        {
            if (in.data[i * in.step[0] + j] > 20)
            {
                left = j;
                is_found = true;
                break;
            }
        }

        if (is_found)
        {
            break;
        }
    }
    for (int j = in.cols - 1; j >= 0; j--)
    {
        bool is_found = false;
        for (int i = 0; i < in.rows; i++)
        {
            if (in.data[i * in.step[0] + j] > 20)
            {
                right = j;
                is_found = true;
                break;
            }
        }
        if (is_found)
        {
            break;
        }
    }

    Rect out(Point(left, top), Point(right, bottom));
    return out;
}

Mat getRectMat(const Mat &in, const Rect &rect)
{
    Mat out = Mat::zeros(in.rows, in.cols, CV_8UC1);

    int x_offset = int(floor(float(in.cols - rect.width) / 2.f));
    int y_offset = int(floor(float(in.rows - rect.height) / 2.f));

    for (int i = 0; i < rect.height; i++)
    {
        for (int j = 0; j < rect.width; j++)
        {
            out.data[out.step[0] * (i + y_offset) + j + x_offset] =
                in.data[out.step[0] * (i + rect.tl().y) + j + rect.tl().x];
        }
    }
    return out;
}

float getOverThreshNum(const Mat &in, int thresh)
{
    float over_thresh_num = 0.f;
    if (in.rows > 1)
    {
        for (int i = 0; i < in.rows; i++)
        {
            if (in.data[in.step[0] * i] > thresh)
            {
                over_thresh_num++;
            }
        }
    }
    else if (in.cols > 1)
    {
        for (int i = 0; i < in.cols; i++)
        {
            if (in.data[i] > thresh)
            {
                over_thresh_num++;
            }
        }
    }
    return over_thresh_num;
}

Mat projectedHistogram(const Mat &in, int flag, int thresh)
{
    int cols = (flag) ? in.rows : in.cols;
    Mat out = Mat::zeros(1, cols, CV_32F);

    for (int i = 0; i < cols; i++)
    {
        Mat temp_mat = (flag) ? in.row(i) : in.col(i);
        out.at<float>(i) = getOverThreshNum(temp_mat, thresh);
    }

    double min_val, max_val;
    minMaxLoc(out, &min_val, &max_val);

    if (max_val > 0)
    {
        out.convertTo(out, -1, 1.f / max_val, 0);
    }

    return out;
}

Mat getTranslatedMat(const Mat &in, float x_offset, float y_offset,int bg_color)
{
    Mat translate_mat = Mat::eye(2, 3, CV_32F);
    translate_mat.at<float>(0, 2) = x_offset;
    translate_mat.at<float>(1, 2) = y_offset;

    Mat out;
    warpAffine(in, out, translate_mat, in.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(bg_color));
    return out;
}

Mat getRotatedMat(const Mat &in, float angle,int bg_color)
{
    Point2f center_point(in.cols / 2.f, in.rows / 2.f);
    Mat rotate_mat = getRotationMatrix2D(center_point, angle, 1.0);
    Mat out;
    warpAffine(in, out, rotate_mat, in.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(bg_color));
    return out;
}

Mat getCropMat(const Mat &in,int x,int y,int width,int height)
{
    Rect rect(x,y,width,height);
    Mat out = in(rect);
    resize(out,out,Size(width,height));
    return out;
}

