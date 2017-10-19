#include "identifier.h"

CharIdentifier *CharIdentifier::instance = NULL;

CharIdentifier *CharIdentifier::getInstance()
{
    if (instance == NULL)
        instance = new CharIdentifier();
    return instance;
}

CharIdentifier::CharIdentifier()
{
    ann_ptr = ANN_MLP::load(kANNModelPath);
    ann_chinese_ptr = ANN_MLP::load(kANNModelChinesePath);
}

CharIdentifier::~CharIdentifier()
{
    delete instance;
}

void CharIdentifier::classify(vector<CChar> &cchar_vec)
{
    if (cchar_vec.size() == 0)
        return;

    Mat sample_mat;

    for (int i = 0; i < cchar_vec.size(); i++)
    {
        Mat cchar_mat = cchar_vec[i].getMat();
        Mat feature_mat = charFeaturesForANNTrain(cchar_mat, kANNCharDataSize);
        sample_mat.push_back(feature_mat);
    }

    Mat result_mat(cchar_vec.size(), kCharsTotalNumber, CV_32FC1);
    ann_ptr->predict(sample_mat, result_mat);

    for (int i = 0; i < cchar_vec.size(); i++)
    {
        CChar &cchar = cchar_vec[i];
        int result = 0;
        float max_val = -2.f;
        string label = "";

        for (int j = 0; j < kCharsTotalNumber; j++)
        {
            float val = result_mat.row(i).at<float>(j);
            if (val > max_val)
            {
                max_val = val;
                result = j;
            }
        }
    
        label.append(kChars[result]);
        cchar.setScore(max_val);
        cchar.setLabel(label);
    }
}

void CharIdentifier::classifyChinese(vector<CChar> &cchar_vec)
{
    if (cchar_vec.size() == 0)
        return;

    Mat sample_mat;

    for (int i = 0; i < cchar_vec.size(); i++)
    {
        Mat cchar_mat = cchar_vec[i].getMat();
        Mat feature_mat = charFeaturesForANNTrain(cchar_mat, kANNChineseDataSize);
        sample_mat.push_back(feature_mat);
    }

    Mat result_mat(cchar_vec.size(), kChineseNumber, CV_32FC1);
    ann_chinese_ptr->predict(sample_mat, result_mat);

    for (int i = 0; i < cchar_vec.size(); i++)
    {
        CChar &cchar = cchar_vec[i];
        bool is_chinese = true;
        float max_val = -2.f;
        int result = 0;
        string label = "";

        for (int j = 0; j < kChineseNumber; j++)
        {
            float val = result_mat.row(i).at<float>(j);
            if (val > max_val)
            {
                max_val = val;
                result = j;
            }
        }

        if (max_val < 0.9f)
            is_chinese = false;

        label.append(kChars[kCharsTotalNumber - kChineseNumber + result]);

        cchar.setScore(max_val);
        cchar.setIsChinese(is_chinese);
        cchar.setLabel(label);
    }
}

bool CharIdentifier::isChinese(CChar &cchar,float thresh)
{
    Mat feature_mat = charFeaturesForANNTrain(cchar.getMat(),kANNChineseDataSize);

    Mat result_mat(1,kChineseNumber,CV_32FC1);
    ann_chinese_ptr->predict(feature_mat,result_mat);

    float max_val = -2.f;
    int result = 0;

    for(int i = 0; i < kChineseNumber; i++)
    {
        float val = result_mat.at<float>(i);
        if(val > max_val)
        {
            max_val = val;
            result = i;
        }
    }
    cchar.setScore(max_val);
    cchar.setLabel(string(kChars[result + kCharsTotalNumber - kChineseNumber]));
   
    if(max_val > thresh)
    {
        cchar.setIsChinese(true);
        return true;
    }
    cchar.setIsChinese(false);
    return false;
}

