#include "functions.h"

bool compareCCharByScore(const CChar &cchar1, const CChar &cchar2)
{
    return cchar1.getScore() > cchar2.getScore();
}

bool compareCCharByCenterX(const CChar &cchar1,const CChar &cchar2)
{
    return cchar1.getCenterPoint().x < cchar2.getCenterPoint().x;
}

bool compareCCharByRectTlX(const CChar &cchar1,const CChar &cchar2)
{
    return cchar1.getRect().tl().x < cchar2.getRect().tl().x;
}

bool compareVec2iByX(const Vec2i &vec2i1,const Vec2i &vec2i2)
{
    return vec2i1[0] < vec2i2[0];
}

bool compareCCharByRectTlXGD(const CChar &cchar1,const CChar &cchar2)
{
    return cchar1.getRect().tl().x > cchar2.getRect().tl().x;
}

Rect getSafeRect(const Mat &in,const Point2f &center_point,float width,float height)
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
    Rect rect(x, y, width, height);
    Mat out = in(rect);
    resize(out, out, Size(width, height));
    return out;
}

Mat scaleImage(const Mat &in, const Size &size)
{
    Mat out;
    if (in.cols > size.width || in.rows > size.height)
    {
        float width_ratio = float(in.cols / size.width);
        float height_ratio = float(in.rows / size.height);

        float scale_ratio = width_ratio > height_ratio ? width_ratio : height_ratio;
        int new_width = in.cols / scale_ratio;
        int new_height = in.rows / scale_ratio;
        resize(in, out, Size(new_width, new_height), 0, 0);
    }
    else
        out = in;

    return out;
}

bool verifyCharSizes(const Rect &rect)
{
    float error = 0.35f;
    float aspect = 45.f / 90.f;
    float min_aspect = 0.05f;
    float max_aspect = aspect + error * aspect;
    float ratio = (float)rect.width / (float)rect.height;

    if (ratio > min_aspect && ratio < max_aspect)
        return true;
    return false;
}

Mat paintImageByPoints(const vector<Point> &point_vec, const Rect &rect)
{
    int width_expend = 0;
    int height_expend = 0;

    if (rect.width > rect.height)
        height_expend = (rect.width - rect.height) / 2;
    else
        width_expend = (rect.height - rect.width) / 2;

    Mat out(rect.height + height_expend * 2, rect.width + width_expend * 2, CV_8UC1, Scalar(0, 0, 0));
    for (int i = 0; i < point_vec.size(); i++)
    {
        Point point = point_vec[i];
        int x = point.x - rect.tl().x + width_expend;
        int y = point.y - rect.tl().y + height_expend;
        if (x >= 0 && x < out.cols && y >= 0 && y < out.rows)
            out.data[out.step[0] * y + x] = Scalar(255,255,255).val[0];
    }
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

Rect adaptiveCharRect(const Rect &rect, int max_width)
{
    int width_expend = 0;
    if (rect.height > 3 * rect.width)
    {
        // rect.height/2等于应得的宽度(长宽比90/45)
        width_expend = int(int(rect.height * 0.5f) - rect.width) * 0.5f;
    }

    int tlx = rect.tl().x - width_expend > 0 ? rect.tl().x - width_expend : 0;
    int brx = rect.br().x + width_expend < max_width ? rect.br().x + width_expend : max_width;

    Rect out(tlx, rect.tl().y, brx - tlx, rect.height);
    return out;
}

bool judgeMserCharDiffRatio(const Mat &in, const Rect &rect, float thresh)
{
    Mat binary_mat;
    threshold(in(rect), binary_mat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    Rect adaptive_rect = adaptiveCharRect(rect, in.cols);
    Mat adaptive_mat = in(adaptive_rect);

    Mat adaptive_binary_mat;
    threshold(adaptive_mat, adaptive_binary_mat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    int diff = abs(countNonZero(adaptive_binary_mat) - countNonZero(binary_mat));
    float diff_ratio = float(diff) / float(rect.area());

    if (diff_ratio > thresh)
        return false;

    return true;
}

void notMaxSuppression(vector<CChar> &cchar_vec, float overlap)
{
    sort(cchar_vec.begin(), cchar_vec.end(), compareCCharByScore);

    vector<CChar>::iterator it = cchar_vec.begin();
    for (; it != cchar_vec.end(); it++)
    {
        CChar cchar1 = *it;
        Rect cchar_rect1 = cchar1.getRect();
        vector<CChar>::iterator itn = it + 1;

        for (; itn != cchar_vec.end();)
        {
            CChar cchar2 = *itn;
            Rect cchar_rect2 = cchar2.getRect();
            float iou = calcInsertOverUnion(cchar_rect1, cchar_rect2);
            if (iou > overlap)
                itn = cchar_vec.erase(itn);
            else
                itn++;
        }
    }
}

Rect interRect(const Rect &rect1, const Rect &rect2)
{
    int x = rect1.x > rect2.x ? rect1.x : rect2.x;
    int y = rect1.y > rect2.y ? rect1.y : rect2.y;
    int width = (rect1.x + rect1.width < rect2.x + rect2.width ? rect1.x + rect1.width : rect2.x + rect2.width) - x;
    int height = (rect1.y + rect1.height < rect2.y + rect2.height ? rect1.y + rect1.height : rect2.y + rect2.height) - y;

    Rect out(x, y, width, height);
    if (out.width <= 0 || out.height <= 0)
        out = Rect();
    return out;
}

Rect mergeRect(const Rect &rect1, const Rect &rect2)
{
    int x = rect1.x < rect2.x ? rect1.x : rect2.x;
    int y = rect1.y < rect2.y ? rect1.y : rect2.y;
    int width = (rect1.x + rect1.width > rect2.x + rect2.width ? rect1.x + rect1.width : rect2.x + rect2.width) - x;
    int height = (rect1.y + rect1.height > rect2.y + rect2.height ? rect1.y + rect1.height : rect2.y + rect2.height) - y;

    Rect out(x, y, width, height);
    return out;
}

float calcInsertOverUnion(const Rect &rect1, const Rect &rect2)
{
    Rect inter_rect = interRect(rect1, rect2);
    Rect merge_rect = mergeRect(rect1, rect2);

    float iou = float(inter_rect.area()) / float(merge_rect.area());

    return iou;
}

bool isCharsBelongToOneGroup(const CChar &cchar1, const CChar &cchar2)
{
    Rect rect1 = cchar1.getRect();
    Rect rect2 = cchar2.getRect();

    //候选字符具有相似的矩型高度
    float height1 = float(rect1.height);
    float height2 = float(rect2.height);

    float height_diff = abs(height1 - height2);
    float height_diff_ratio = height_diff / min(height1, height2);

    if (height_diff_ratio > 0.25f)
        return false;

    //候选字符在同一水平线(y轴相似)
    float tly1 = float(rect1.tl().y);
    float tly2 = float(rect2.tl().y);

    float tly_diff = abs(tly1 - tly2);
    float tly_diff_ratio = tly_diff / min(height1, height2);

    if (tly_diff_ratio > 0.5f)
        return false;

    //候选字符在同一水平线上,但是中心点不那么靠近
    float center_x1 = float(rect1.tl().x + rect1.width / 2);
    float center_x2 = float(rect2.tl().x + rect2.width / 2);

    float center_x_diff = abs(center_x1 - center_x2);
    float center_x_diff_ratio = center_x_diff / min(height1, height2);

    if (center_x_diff_ratio < 0.25f)
        return false;

    //候选字符在同一水平线上,距离不能太远
    float min_brx = float(min(rect1.br().x, rect2.br().x));
    float max_tlx = float(max(rect1.tl().x, rect2.tl().x));

    float x_diff = abs(max_tlx - min_brx);
    float x_diff_ratio = x_diff / min(height1, height2);

    if (x_diff_ratio > 1.f)
        return false;

    return true;
}

void mergeCharToGroup(const vector<CChar> &cchar_vec, vector<vector<CChar> > &cchar_group_vec)
{
    vector<int> group_label_vec;

    int group_num;

    if (cchar_vec.size() < 0)
        return;

    group_num = partition(cchar_vec, group_label_vec, &isCharsBelongToOneGroup);

    for (int i = 0; i < group_num; i++)
    {
        vector<CChar> cchar_group;

        for (int j = 0; j < cchar_vec.size(); j++)
        {
            int group_label = group_label_vec[j];

            if (group_label == i)
                cchar_group.push_back(cchar_vec[j]);
        }

        if (cchar_group.size() < 2)
            continue;

        cchar_group_vec.push_back(cchar_group);
    }
}

void removeRightOutliers(vector<CChar> &cchar_vec,vector<CChar> &out_cchar_vec,float min_thresh,float max_thresh)
{
    sort(cchar_vec.begin(),cchar_vec.end(),compareCCharByCenterX);
    
    vector<float> slope_vec;

    for(int i = 0; i + 1 < cchar_vec.size(); i++)
    {
        Vec4f line_vec4f;
        CChar left_cchar = cchar_vec.at(i);
        CChar right_cchar = cchar_vec.at(i + 1);

        vector<Point> point_vec;
        point_vec.push_back(left_cchar.getCenterPoint());
        point_vec.push_back(right_cchar.getCenterPoint());

        fitLine(Mat(point_vec),line_vec4f,CV_DIST_L2,0,0.01,0.01);
        
        float slope = line_vec4f[1] / line_vec4f[0];

        slope_vec.push_back(slope);
    }

    int inlier_num = 0;
    int outlier_index = -1;

    for(int i = 0; i + 1 < slope_vec.size(); i++)
    {
        float slope1 = slope_vec.at(i);
        float slope2 = slope_vec.at(i + 1);
        float slope_diff = abs(slope1 - slope2);

        
        if(slope_diff <= min_thresh)
            inlier_num++;
           
        if(inlier_num >= 2 && slope_diff >= max_thresh)
        {
            
            outlier_index = i + 2;
            break;
        }
     
    }
   
    for(int i = 0; i < cchar_vec.size(); i++)
    {
       
        if(i != outlier_index)
        {
            CChar cchar = cchar_vec.at(i);
            out_cchar_vec.push_back(cchar);
        }
    }
   
   
}

void removeContainChar(vector<CChar> &cchar_vec,float thresh)
{
    sort(cchar_vec.begin(),cchar_vec.end(),compareCCharByRectTlX);
    
    vector<CChar>::iterator it = cchar_vec.begin();

    for(;it != cchar_vec.end();it++)
    {
        CChar cchar1 = *it;
        Rect cchar_rect1 = cchar1.getRect();
        vector<CChar>::iterator itn = it + 1;

        for(;itn != cchar_vec.end();)
        {
            CChar cchar2 = *itn;
            Rect cchar_rect2 = cchar2.getRect();

            Rect and_rect = cchar_rect1 & cchar_rect2;

            float thresh1 = 1.f - float(and_rect.area()) / float(cchar_rect1.area());
            float thresh2 = 1.f - float(and_rect.area()) / float(cchar_rect2.area());

            if(thresh1 < thresh || thresh2 < thresh)
                itn = cchar_vec.erase(itn);
            else 
                itn++;
        }
    }
    sort(cchar_vec.begin(),cchar_vec.end(),compareCCharByRectTlX);
}

void axesSearch(const Vec4f &line_vec4f,
                Point &left_point,
                Point &right_point,
                Rect &max_cchar_rect,
                Rect &cplate_rect,
                vector<CChar> &cchar_vec,
                vector<CChar> &out_cchar_vec,
                float thresh,
                int flag)
{

    vector<CChar> axes_cchar_vec;
    axes_cchar_vec.reserve(128);

    float k = line_vec4f[1] / line_vec4f[0];
    float x = line_vec4f[2];
    float y = line_vec4f[3];

    for (int i = 0; i < cchar_vec.size(); i++)
    {
        CChar cchar = cchar_vec.at(i);
        Rect cchar_rect = cchar.getRect();

        if (flag == 0)
        {
            if (cchar_rect.tl().x + cchar_rect.width > left_point.x)
                continue;
        }
        else if (flag == 1)
        {
            if (cchar_rect.tl().x < right_point.x)
                continue;
        }

        Point cchar_center_point = cchar.getCenterPoint();
        float cchar_x = float(cchar_center_point.x);
        float cchar_y = float(cchar_center_point.y);
        float guess_cchar_y = k * (cchar_x - x) + y;
        float y_diff_ratio = abs(guess_cchar_y - cchar_y) / max_cchar_rect.height;

        if (y_diff_ratio < thresh)
        {
            float width1 = float(max_cchar_rect.width);
            float height1 = float(max_cchar_rect.height);

            float width2 = float(cchar_rect.width);
            float height2 = float(cchar_rect.height);

            float height_diff_ratio = abs(height1 - height2) / height1;
            float width_diff_ratio = abs(width1 - width2) / width1;

            if ((height_diff_ratio < thresh && width_diff_ratio < 0.5f) || (cchar_rect.area() < max_cchar_rect.area() && cchar_rect.area() > max_cchar_rect.area() * 0.5))
                axes_cchar_vec.push_back(cchar);
        }
    }

    if (axes_cchar_vec.size() != 0)
    {
        if (flag == 0)
            sort(axes_cchar_vec.begin(), axes_cchar_vec.end(), compareCCharByRectTlXGD);

        else if (flag == 1)
            sort(axes_cchar_vec.begin(), axes_cchar_vec.end(), compareCCharByRectTlX);

        CChar first_axes_cchar = axes_cchar_vec.at(0);
        Point first_axes_cchar_center_point = first_axes_cchar.getCenterPoint();

        float ratio;
        if (flag == 0)
            ratio = float(abs(first_axes_cchar_center_point.x - left_point.x)) / float(max_cchar_rect.height);
        else if (flag == 1)
            ratio = float(abs(first_axes_cchar_center_point.x - right_point.x)) / float(max_cchar_rect.height);
        if (ratio > 0.9f)
            return;

        out_cchar_vec.push_back(first_axes_cchar);
        cplate_rect |= first_axes_cchar.getRect();

        if (flag == 0)
            left_point = first_axes_cchar_center_point;
        else if (flag == 1)
            right_point = first_axes_cchar_center_point;

        for (int i = 0; i + 1 < axes_cchar_vec.size(); i++)
        {
            CChar axes_cchar1 = axes_cchar_vec.at(i);
            CChar axes_cchar2 = axes_cchar_vec.at(i + 1);
            Rect axes_cchar_rect1 = axes_cchar1.getRect();
            Rect axes_cchar_rect2 = axes_cchar2.getRect();

            float height1 = axes_cchar_rect1.height;
            float height2 = axes_cchar_rect2.height;

            float min_brx = min(axes_cchar_rect1.br().x, axes_cchar_rect2.br().x);
            float max_tlx = max(axes_cchar_rect1.tl().x, axes_cchar_rect2.tl().x);
            float diff_ratio = abs(min_brx - max_tlx) / min(height1, height2);

            if (diff_ratio > 1.f)
                break;
            else
            {
                out_cchar_vec.push_back(axes_cchar2);
                cplate_rect |= axes_cchar_rect2;
                if (flag == 0)
                {
                    if (axes_cchar2.getCenterPoint().x < left_point.x)
                        left_point = axes_cchar2.getCenterPoint();
                }
                else if (flag == 1)
                {
                    if (axes_cchar2.getCenterPoint().x > right_point.x)
                        right_point = axes_cchar2.getCenterPoint();
                }
            }
        }
    }
}

void slideWindowSearch(const Mat &in,
                       const Vec4f &line_vec4f,
                       Point &left_point,
                       Point &right_point,
                       const Rect &max_cchar_rect,
                       Rect &cplate_rect,
                       vector<CChar> &out_cchar_vec,
                       float otsu_level,
                       float window_ratio,
                       float thresh,
                       bool is_chinese,
                       int flag)
{

    float k = line_vec4f[1] / line_vec4f[0];
    float x = line_vec4f[2];
    float y = line_vec4f[3];
    int slide_length = window_ratio * max_cchar_rect.width;
    int slide_step  = 1;
    int from_x = 0;

    Point from_point;
   
    if(flag == 0)
    {
        from_point = left_point;
        from_x = from_point.x - max_cchar_rect.width;
    }
    else if(flag == 1)
    {
        from_point = right_point;
        from_x = from_point.x + max_cchar_rect.width;
    }

    vector<CChar> slide_cchar_vec;

    for(int slide_x = -slide_length; slide_x < slide_length; slide_x += slide_step)
    {
        float temp_x = 0;
        if(flag == 0)
        {
            temp_x = float(from_x - slide_x);

        }
        else if(flag == 1)
        {
            temp_x = float(from_x + slide_x);
        }
        
        float temp_y = y + k * (temp_x - x); 

        Point slide_point(temp_x,temp_y);

        
        int chinese_height = 1.05 * max_cchar_rect.height;
        int chinese_width = 1.05 * max_cchar_rect.width;

        Rect rect(Point2f(temp_x - chinese_width / 2,temp_y - chinese_height / 2),
                  Point2f(temp_x + chinese_width / 2,temp_y + chinese_height / 2));
        
        if(rect.tl().x < 0 || rect.tl().y < 0 || rect.br().x > in.cols || rect.br().y > in.rows)
            continue;
         

        Mat cchar_mat = in(rect);
        Mat cchar_binary_mat;
        threshold(cchar_mat,cchar_binary_mat,otsu_level,255,CV_THRESH_BINARY);
        cchar_binary_mat = preprocessChar(cchar_binary_mat,kCharSize);

        CChar cchar;
        cchar.setMat(cchar_binary_mat);
        cchar.setRect(rect);
        slide_cchar_vec.push_back(cchar);
    }

    if(is_chinese)
        CharIdentifier::getInstance()->classifyChinese(slide_cchar_vec);
    else
        CharIdentifier::getInstance()->classify(slide_cchar_vec);
     
    float overlap = 0.1;
    notMaxSuppression(slide_cchar_vec,overlap);

    for(int i = 0; i < slide_cchar_vec.size(); i++)
    {
        CChar cchar = slide_cchar_vec.at(i);
        Rect cchar_rect = cchar.getRect();
        Point cchar_center_point(cchar_rect.tl().x + cchar_rect.width / 2,cchar_rect.tl().y + cchar_rect.height / 2);
        if(cchar.getScore() > thresh && strcmp(cchar.getLabel().c_str(),"1") != 0)
        {
            cplate_rect |= cchar_rect;
            out_cchar_vec.push_back(cchar);
            from_point = cchar_center_point;
        }
    }
    if(flag == 0)
        left_point = from_point;
    else if(flag == 1)
        right_point = from_point;
}

void combineRect(const Mat &in, vector<CChar> &cchar_vec,vector<CChar> &out_cchar_vec,const Vec2i &dest_vec2i,const Rect &max_cchar_rect, float min_thresh, float max_thresh)
{
    if (cchar_vec.size() == 0)
        return;

    sort(cchar_vec.begin(), cchar_vec.end(), compareCCharByCenterX);
    int avg_dist = dest_vec2i[0] * dest_vec2i[0] + dest_vec2i[1] * dest_vec2i[1];

    vector<CChar> combine_cchar_vec;
    int i = 0;
    for (;i + 1 < cchar_vec.size(); i++)
    {
        CChar cchar1 = cchar_vec.at(i);
        CChar cchar2 = cchar_vec.at(i + 1);

        Point cchar_center_point1 = cchar1.getCenterPoint();
        Point cchar_center_point2 = cchar2.getCenterPoint();

        int x_diff = cchar_center_point1.x - cchar_center_point2.x;
        int y_diff = cchar_center_point1.y - cchar_center_point2.y;

        int dist = x_diff * x_diff + y_diff * y_diff;

        float ratio = float(dist) / float(avg_dist);
        if (ratio > max_thresh)
        {
            float x_avg = float(cchar_center_point1.x + cchar_center_point2.x) / 2.f;
            float y_avg = float(cchar_center_point1.y + cchar_center_point2.y) / 2.f;

            float width = float(max_cchar_rect.width);
            float height = float(max_cchar_rect.height);

            Rect cchar_rect = getSafeRect(in, Point2f(x_avg, y_avg), width, height);

            combine_cchar_vec.push_back(cchar1);

            CChar cchar;
            cchar.setCenterPoint(Point((int)x_avg, (int)y_avg));
            cchar.setRect(cchar_rect);
            combine_cchar_vec.push_back(cchar);
        }
        else if (ratio < min_thresh)
        {
            Rect or_rect = cchar1.getRect() | cchar2.getRect();

            int x_avg = or_rect.tl().x + or_rect.width / 2;
            int y_avg = or_rect.tl().y + or_rect.height / 2;

            CChar cchar;
            cchar.setCenterPoint(Point(x_avg, y_avg));
            cchar.setRect(or_rect);
            combine_cchar_vec.push_back(cchar);

            i++;
        }
        else
            combine_cchar_vec.push_back(cchar1);
    }

    if (i + 1 == cchar_vec.size())
    {
        combine_cchar_vec.push_back(cchar_vec.at(i));
    }

    out_cchar_vec = combine_cchar_vec;
}