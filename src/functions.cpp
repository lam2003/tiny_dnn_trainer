#include "functions.h"



Rect getSafeRect(const Mat &in, const Point2f &center_point, float width, float height)
{
    float tlx = center_point.x - width / 2.f;
    float tly = center_point.y - height / 2.f;
    float brx = center_point.x + width / 2.f;
    float bry = center_point.y + height / 2.f;

    tlx = tlx > 0 ? tlx : 0;
    tly = tly > 0 ? tly : 0;
    brx = brx < float(in.cols) ? brx : float(in.cols);
    bry = bry < float(in.rows) ? bry : float(in.rows);

    Rect out;
    out.x = tlx;
    out.y = tly;
    out.width = brx - tlx;
    out.height = bry - tly;
    return out;
}



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


Mat getTranslatedMat(const Mat &in, float x_offset, float y_offset, int bg_color)
{
    Mat translate_mat = Mat::eye(2, 3, CV_32F);
    translate_mat.at<float>(0, 2) = x_offset;
    translate_mat.at<float>(1, 2) = y_offset;

    Mat out;
    warpAffine(in, out, translate_mat, in.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(bg_color));
    return out;
}

Mat getRotatedMat(const Mat &in, float angle, int bg_color)
{
    Point2f center_point(in.cols / 2.f, in.rows / 2.f);
    Mat rotate_mat = getRotationMatrix2D(center_point, angle, 1.0);
    Mat out;
    warpAffine(in, out, rotate_mat, in.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(bg_color));
    return out;
}

Mat getCropMat(const Mat &in, int x, int y, int width, int height)
{
    x = x > 0 ? x : 0;
    y = y > 0 ? y : 0;
    width = width < in.cols ? width : in.cols - x;
    height = height < in.rows ? height : in.rows - y;

    Rect rect(x, y, width, height);
    
    Mat out = in(rect);
    resize(out, out, Size(width, height));
    return out;
}





Mat preprocessChar(const Mat &in, int char_size)
{
    int m = max(in.cols, in.rows);
    float x_offset = float(m / 2 - in.cols / 2);
    int y_offset = float(m / 2 - in.rows / 2);

    Mat out = getTranslatedMat(in, x_offset, y_offset);
    resize(out, out, Size(char_size, char_size));

    return out;
}




