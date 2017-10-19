#include "locater.h"

Locater::Locater(const Mat &in, bool display_process)
{
    rgb_mat = in.clone();
    cvtColor(in, gray_mat, CV_BGR2GRAY);
    cplate_vec.reserve(16);
    this->display_process = display_process;
}

Locater::~Locater() {}

void Locater::mserCharLocated()
{
    Mat scale_gray_mat = scaleImage(gray_mat, Size(1000, 1000));
    Mat scale_rgb_mat = scaleImage(rgb_mat, Size(1000, 1000));

    vector<vector<Point> > mser_blue_contour_vec;
    mser_blue_contour_vec.reserve(1024);
    vector<vector<Point> > mser_yellow_contour_vec;
    mser_yellow_contour_vec.reserve(1024);
    vector<Rect> mser_blue_rect_vec;
    mser_blue_rect_vec.reserve(1024);
    vector<Rect> mser_yellow_rect_vec;
    mser_yellow_rect_vec.reserve(1024);

    vector<Color> color_vec;
    color_vec.push_back(BLUE);
    color_vec.push_back(YELLOW);

    Mat formser_mat = scale_gray_mat.clone();

    const int mser_delta = 1;
    const int mser_area = formser_mat.rows * formser_mat.cols;
    const int mser_min_area = 30;
    const int mser_max_area = int(mser_area * 0.05);

    Ptr<MSER2> mser;
    mser = MSER2::create(mser_delta, mser_min_area, mser_max_area);
    mser->detectRegions(formser_mat, mser_blue_contour_vec, mser_blue_rect_vec, mser_yellow_contour_vec, mser_yellow_rect_vec);

    vector<vector<vector<Point> > > contour_vec_vec;
    contour_vec_vec.push_back(mser_blue_contour_vec);
    contour_vec_vec.push_back(mser_yellow_contour_vec);
    vector<vector<Rect> > rect_vec_vec;
    rect_vec_vec.push_back(mser_blue_rect_vec);
    rect_vec_vec.push_back(mser_yellow_rect_vec);

    for (int color_index = 0; color_index < color_vec.size(); color_index++)
    {
        vector<CChar> cchar_vec;
        cchar_vec.reserve(128);

        for (int i = 0; i < contour_vec_vec.at(color_index).size(); i++)
        {
            Rect mser_rect = rect_vec_vec.at(color_index)[i];
            vector<Point> &mser_contour = contour_vec_vec.at(color_index)[i];

            if (verifyCharSizes(mser_rect))
            {
                Mat mser_mat = paintImageByPoints(mser_contour, mser_rect);
                Mat cchar_mat = preprocessChar(mser_mat, kCharSize);
                Rect cchar_rect = mser_rect;
                Point cchar_center_point(cchar_rect.tl().x + cchar_rect.width / 2, cchar_rect.tl().y + cchar_rect.height / 2);

                Mat cchar_binary_mat;
                float cchar_otsu_level = threshold(cchar_mat, cchar_binary_mat, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);

                if (judgeMserCharDiffRatio(scale_gray_mat, cchar_rect))
                {
                    CChar cchar;
                    cchar.setRect(cchar_rect);
                    cchar.setMat(cchar_mat);
                    cchar.setOtsuLevel(cchar_otsu_level);
                    cchar.setCenterPoint(cchar_center_point);
                    cchar_vec.push_back(cchar);
                }
            }
        }

        CharIdentifier::getInstance()->classify(cchar_vec);

        notMaxSuppression(cchar_vec, 0.6); //sort by score

        vector<CChar> strong_seed_vec;
        strong_seed_vec.reserve(128);

        for (int i = 0; i < cchar_vec.size(); i++)
        {
            CChar cchar = cchar_vec.at(i);
            if (cchar.getIsStrong())
                strong_seed_vec.push_back(cchar);
        }

        notMaxSuppression(strong_seed_vec, 0.3); //sort by score
    
        if (display_process)
        {
            for (int i = 0; i < cchar_vec.size(); i++)
            {
                CChar cchar = cchar_vec.at(i);
                rectangle(scale_rgb_mat, cchar.getRect(), Scalar(0, 255, 0), 1); // green
            }
            for (int i = 0; i < strong_seed_vec.size(); i++)
            {
                CChar cchar = strong_seed_vec.at(i);
                rectangle(scale_rgb_mat, cchar.getRect(), Scalar(0, 255, 255), 1); //yellow
            }
        }

        vector<vector<CChar> > cchar_group_vec;
        cchar_group_vec.reserve(128);

        mergeCharToGroup(strong_seed_vec,cchar_group_vec);
        
        for(int i = 0; i < cchar_group_vec.size(); i++)
        {
            vector<CChar> rmo_cchar_group; //removed outliers cchar group
            rmo_cchar_group.reserve(128);
           
            removeRightOutliers(cchar_group_vec.at(i),rmo_cchar_group,0.2f,0.5f); //sort by centerx
           
            removeContainChar(rmo_cchar_group,0.1f); //sort by tlx
           
            
            vector<Point> cchar_center_point_vec;
            cchar_center_point_vec.reserve(128);
            vector<CChar> mser_cchar_vec;
            mser_cchar_vec.reserve(128);
            float cchar_otsu_level_sum = 0.f;
            int max_area = 0;
            Rect cplate_rect = cchar_group_vec.at(i)[0].getRect();
            Point cplate_left_point(scale_gray_mat.cols,0);
            Point cplate_right_point(0,0);
            Vec4f cplate_line_vec4f;
            Rect cplate_max_cchar_rect;

          
            for(int j = 0; j < rmo_cchar_group.size(); j++)
            {
                CChar cchar = rmo_cchar_group[j];
                Rect cchar_rect = cchar.getRect();

                cplate_rect |= cchar_rect;
                cchar_center_point_vec.push_back(cchar.getCenterPoint());

                mser_cchar_vec.push_back(cchar);
                cchar_otsu_level_sum += cchar.getOtsuLevel();

                if(cchar_rect.area() > max_area)
                {
                    cplate_max_cchar_rect = cchar_rect;
                    max_area = cchar_rect.area();
                }

                if(cchar.getCenterPoint().x < cplate_left_point.x)
                    cplate_left_point = cchar.getCenterPoint();
                if(cchar.getCenterPoint().x > cplate_right_point.x)
                    cplate_right_point = cchar.getCenterPoint();
            }

            float cplate_otsu_level = cchar_otsu_level_sum / rmo_cchar_group.size();
            float cplate_max_cchar_rect_ratio = float(cplate_max_cchar_rect.width) / float(cplate_max_cchar_rect.height);

        
            if(cchar_center_point_vec.size() >= 2 && cplate_max_cchar_rect_ratio >= 0.3)
            {
                fitLine(Mat(cchar_center_point_vec),cplate_line_vec4f,CV_DIST_L2,0,0.01,0.01);

                Vec2i cplate_dest_vec2i;
                vector<Vec2i> dist_vec2i_vec;
                dist_vec2i_vec.reserve(128);

                for(int j =0; j + 1 < mser_cchar_vec.size(); j++)
                {
                    Rect cchar_rect1 = mser_cchar_vec.at(j).getRect();
                    Rect cchar_rect2 = mser_cchar_vec.at(j + 1).getRect();
                    
                 
                    Vec2i dist_vec2i(cchar_rect2.x - cchar_rect1.x,cchar_rect2.y - cchar_rect2.y);
                    dist_vec2i_vec.push_back(dist_vec2i);
                }

                sort(dist_vec2i_vec.begin(),dist_vec2i_vec.end(),compareVec2iByX);
                cplate_dest_vec2i = dist_vec2i_vec.at(int(float(dist_vec2i_vec.size() - 1) / 2.f));
                
                CPlate cplate;
                cplate.setLeftPoint(cplate_left_point);
                cplate.setRightPoint(cplate_right_point);
                cplate.setLineVec4f(cplate_line_vec4f);
                cplate.setDistVec2i(cplate_dest_vec2i);
                cplate.setOtsuLevel(cplate_otsu_level);
                cplate.setRect(cplate_rect);
                cplate.setMaxCCharRect(cplate_max_cchar_rect);
                cplate.setMserCCharVec(mser_cchar_vec);
                cplate_vec.push_back(cplate);
            }
        }

        
        for(int i = 0; i < cplate_vec.size(); i++)
        {
            CPlate &cplate = cplate_vec.at(i);
            Vec2i dist_vec2i = cplate.getDistVec2i();
            vector<CChar> mser_cchar_vec = cplate.getCopyOfMserCCharVec();
            Vec4f line_vec4f = cplate.getLineVec4f();
            Point left_point = cplate.getLeftPoint();
            Point right_point = cplate.getRightPoint();
            Rect max_cchar_rect = cplate.getMaxCCharRect();
            Rect cplate_rect = cplate.getRect();
            float otsu_level = cplate.getOtsuLevel();
        
            const int LEFT = 0;
            const int RIGHT = 1;

            vector<CChar> left_axes_cchar_vec;
            left_axes_cchar_vec.reserve(128);
            vector<CChar> right_axes_cchar_vec;
            right_axes_cchar_vec.reserve(128);

            if(mser_cchar_vec.size() < kPlateMaxCharNum)
            {
                axesSearch(line_vec4f,left_point,right_point,max_cchar_rect,cplate_rect,cchar_vec,left_axes_cchar_vec,0.2,LEFT);
                for(int j = 0;j < left_axes_cchar_vec.size();j++)
                    mser_cchar_vec.push_back(left_axes_cchar_vec[j]);
                axesSearch(line_vec4f,left_point,right_point,max_cchar_rect,cplate_rect,cchar_vec,right_axes_cchar_vec,0.2,RIGHT);
                for(int j = 0;j < right_axes_cchar_vec.size();j++)
                    mser_cchar_vec.push_back(right_axes_cchar_vec[j]);
            }

            
            combineRect(scale_gray_mat,mser_cchar_vec,mser_cchar_vec,dist_vec2i,max_cchar_rect,0.3f,2.5f);
           
            vector<CChar> slide_cchar_vec;
            slide_cchar_vec.reserve(128);
           
            if(mser_cchar_vec.size() < kPlateMaxCharNum)
            {
                sort(mser_cchar_vec.begin(),mser_cchar_vec.end(),compareCCharByRectTlX);

                CChar &cchar = mser_cchar_vec[0];
                Rect cchar_rect = cchar.getRect();

                Mat cchar_mat = scale_gray_mat(cchar_rect);
                Mat cchar_binary_mat;
                threshold(cchar_mat,cchar_binary_mat,0,255,CV_THRESH_BINARY | CV_THRESH_OTSU);
                cchar_binary_mat = preprocessChar(cchar_binary_mat,kCharSize);
                cchar.setMat(cchar_binary_mat);
     
                bool is_chinese = CharIdentifier::getInstance()->isChinese(cchar,0.9f);
               
                if(!is_chinese)
                {
                   
                    slideWindowSearch(scale_gray_mat, line_vec4f, left_point, right_point,max_cchar_rect, cplate_rect,slide_cchar_vec,otsu_level, 0.4, 0.8,true,LEFT);
                    for(int j = 0;j < slide_cchar_vec.size();j++)
                        mser_cchar_vec.push_back(slide_cchar_vec[j]);
                }
          
                if(mser_cchar_vec.size() < kPlateMaxCharNum)
                {
                    slideWindowSearch(scale_gray_mat, line_vec4f, left_point, right_point,max_cchar_rect, cplate_rect,slide_cchar_vec,otsu_level, 0.4, 0.8,false,RIGHT);
                    for(int j = 0;j < slide_cchar_vec.size();j++)
                        mser_cchar_vec.push_back(slide_cchar_vec[j]);
                }
              
                
            }
          
  

            cplate.setRect(cplate_rect);
            cplate.setLeftPoint(left_point);
            cplate.setRightPoint(right_point);

            if (display_process)
                rectangle(scale_rgb_mat,cplate_rect, Scalar(255, 0, 255), 2); // white
            
            for(int j = 0; j < left_axes_cchar_vec.size(); j++)
            {
                cplate.addMserCChar(left_axes_cchar_vec[j]);
                if (display_process)
                    rectangle(scale_rgb_mat, left_axes_cchar_vec[j].getRect(), Scalar(255, 255, 255), 1);
            }
            for(int j = 0; j < right_axes_cchar_vec.size(); j++)
            {
                cplate.addMserCChar(right_axes_cchar_vec[j]);
                if (display_process)
                    rectangle(scale_rgb_mat, right_axes_cchar_vec[j].getRect(), Scalar(0, 0, 0), 1);
            }
            for(int j = 0;j < slide_cchar_vec.size();j++)
            {
                cplate.addMserCChar(slide_cchar_vec[j]);
                if (display_process)
                    rectangle(scale_rgb_mat, slide_cchar_vec[j].getRect(), Scalar(0, 0, 255), 1);
                
            }
           
            
        }
        if(display_process)
        {
            char img_name[512] = {0};
            sprintf(img_name,"mserCharLocated_%d.jpg",color_index);
            imwrite(img_name,scale_rgb_mat);
            imshow(img_name,scale_rgb_mat);
            waitKey(0);
        }
    }
}
