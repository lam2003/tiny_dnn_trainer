#include "trainer.h"

Trainer::Trainer() {}
Trainer::~Trainer(){};

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*ANNTrainer*/
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

ANNTrainer::ANNTrainer(const char *sample_path, const char *xml_path, int type)
{
    this->sample_path = sample_path;
    this->xml_path = xml_path;
    ann_ptr = ANN_MLP::create();
    this->type = type;
}

ANNTrainer::~ANNTrainer() {}

Mat ANNTrainer::getSyntheticMat(const Mat &in)
{
    srand(unsigned(time(NULL)));
    int rand_type = rand();
    Mat out;
    float rand_num = -3.5f;
    float rand_array[70];
    for (int i = 0; i < 70; i++)
    {
        rand_array[i] = rand_num;
        rand_num += 0.1f;
    }

    if (rand_type % 2 == 0)
    {
        float rand_x = rand_array[rand() % 20 + 25];
        float rand_y = rand_array[rand() % 20 + 25];
        out = getTranslatedMat(in, rand_x, rand_y);
    }
    else if (rand_type % 2 != 0)
    {
        float angle = rand_array[rand() % 70];
        out = getRotatedMat(in, angle);
    }
    return out;
}

Ptr<TrainData> ANNTrainer::preprocessTrainData(int min_input_num)
{

    Mat sample_mat;
    vector<int> label_vec;

    int class_num;
    srand(unsigned(time(NULL)));

    if (type == 0)
    {
        class_num = kCharsTotalNumber;
    }
    else if (type == 1)
    {
        class_num = kChineseNumber;
    }

    for (int i = 0; i < class_num; i++)
    {
        const char *dir_name = kChars[kCharsTotalNumber - class_num + i];
        string dir_path;
        dir_path.append(sample_path);
        dir_path.push_back('/');
        dir_path.append(dir_name);
        vector<string> file_path_vec;
        lsDir(dir_path, file_path_vec);

        int input_num = file_path_vec.size();

        vector<Mat> input_mat_vec;
        input_mat_vec.reserve(min_input_num);

        fprintf(stdout, "----------------------------------------\n");
        fprintf(stdout, "loading char:%s\n", dir_path.c_str());

        for (int j = 0; j < file_path_vec.size(); j++)
        {
            Mat input_mat = imread(file_path_vec[j].c_str(), CV_8UC1);
            
            input_mat_vec.push_back(input_mat);
        }

        for (int j = 0; j < min_input_num - input_num; j++)
        {
            int rand_num = rand() % (input_num + j);
            Mat input_mat = input_mat_vec.at(rand_num);
            Mat synthetic_mat = getSyntheticMat(input_mat);
            char output_image_name[512] = {0};
            sprintf(output_image_name, "./res/ANNTrainer/synthetic_mat/%s-%d-synthetic_mat.jpg", dir_name, j);
            imwrite(output_image_name, synthetic_mat);

            input_mat_vec.push_back(synthetic_mat);
        }

        fprintf(stdout, "image count:%d\n", input_mat_vec.size());

        for (int j = 0; j < input_mat_vec.size(); j++)
        {
            Mat input_mat = input_mat_vec.at(j);
            Mat feature_mat;
            if (type == 0)
            {
                feature_mat = charFeaturesForANNTrain(input_mat, kANNCharDataSize);
            }
            else if (type == 1)
            {
                feature_mat = charFeaturesForANNTrain(input_mat, kANNChineseDataSize);
            }
            sample_mat.push_back(feature_mat);
            label_vec.push_back(i);
        }
    }

    sample_mat.convertTo(sample_mat, CV_32F);
    Mat class_mat = Mat::zeros(label_vec.size(), class_num, CV_32F);
    for (int i = 0; i < class_mat.rows; i++)
    {
        class_mat.at<float>(i, label_vec[i]) = 1.f;
    }
    fprintf(stdout, "----------------------------------------\n");
    return TrainData::create(sample_mat, ROW_SAMPLE, class_mat);
}

void ANNTrainer::train()
{
    int input_num = 0;
    int hidden_num = 0;
    int output_num = 0;

    if (type == 0)
    {
        output_num = kCharsTotalNumber;
        input_num = kANNCharInputNum;
    }
    else if (type == 1)
    {
        output_num = kChineseNumber;
        input_num = kANNChineseInputNum;
    }

    hidden_num = kANNHiddenNum;

    Mat layers(1, 3, CV_32SC1);
    layers.at<int>(0) = input_num;
    layers.at<int>(1) = hidden_num;
    layers.at<int>(2) = output_num;

    ann_ptr->setLayerSizes(layers);
    ann_ptr->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    ann_ptr->setTrainMethod(ANN_MLP::BACKPROP);
    if(type == 1)
    {
        ann_ptr->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 30000, 0.0001));
    }
    else if(type == 0)
    { 
        ann_ptr->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000,0));
    }
    ann_ptr->setBackpropWeightScale(0.1);
    ann_ptr->setBackpropMomentumScale(0.1);

    Ptr<TrainData> train_data_ptr = preprocessTrainData(kANNMinTrainDataNum);

    fprintf(stdout, "++++++++++++++++++++++++++++++++++++++++\n");
    fprintf(stdout, "ANNTrainer begin to train,please wait...\n");
    ann_ptr->train(train_data_ptr);
    fprintf(stdout, "saving train result\n");
    fprintf(stdout, "++++++++++++++++++++++++++++++++++++++++\n");
    ann_ptr->save(xml_path);
}

int ANNTrainer::identity(const Mat &in)
{
    Mat feature_mat = charFeaturesForANNTrain(in, kANNCharDataSize);
    float max_val = -2.f;
    int result = 0;

    Mat result_mat(1, kCharsTotalNumber, CV_32FC1);
    ann_ptr->predict(feature_mat, result_mat);

    for (int i = 0; i < kCharsTotalNumber; i++)
    {
        float val = result_mat.at<float>(i);
        if (val > max_val)
        {
            max_val = val;
            result = i;
        }
    }
    return result;
}

int ANNTrainer::identityChinese(const Mat &in)
{
    Mat feature_mat = charFeaturesForANNTrain(in, kANNChineseDataSize);
    float max_val = -2.f;
    int result = 0;

    Mat result_mat(1, kChineseNumber, CV_32FC1);
    ann_ptr->predict(feature_mat, result_mat);

    for (int i = 0; i < kChineseNumber; i++)
    {
        float val = result_mat.at<float>(i);
        if (val > max_val)
        {
            max_val = val;
            result = i;
        }
    }
    return result;
}

void ANNTrainer::test()
{
    int class_num;
    srand(unsigned(time(NULL)));
    int correct_sum = 0;
    int total_sum = 0;

    if (type == 0)
    {
        class_num = kCharsTotalNumber;
    }
    else if (type == 1)
    {
        class_num = kChineseNumber;
    }

    printf("-------------------------------------------------------\n");
    printf("ANNTrainer testing.....\n");
    for (int i = 0; i < class_num; i++)
    {
        const char *dir_name = kChars[kCharsTotalNumber - class_num + i];
        string dir_path;
        dir_path.append(sample_path);
        dir_path.push_back('/');
        dir_path.append(dir_name);
        vector<string> file_path_vec;
        lsDir(dir_path, file_path_vec);

        int correct = 0, total = 0;

        for (int j = 0; j < file_path_vec.size(); j++)
        {
            Mat input_mat = imread(file_path_vec[j], CV_8UC1);
            int result = -1;

            if (type == 0)
            {
                result = identity(input_mat);
            }
            else if (type == 1)
            {
                result = identityChinese(input_mat);
            }

            if (result == i)
            {
                correct++;
                correct_sum++;
            }
            else
            {
                printf("[%d][%d]",i,result);
                if (type == 1)
                {
                    result += kCharsNumber;
                }
                printf("[error]:correct result:%s\t result:%s\n", dir_name, kChars[result]);
            }
            total++;
            total_sum++;
        }
        total == 0 ? 1 : total;
        float rate = float(correct) / total;
        printf("[%s]total:%d\tcorrect:%d\trate:%.2f\n", dir_name, total, correct, rate);
        printf("-------------------------------------------------------\n");
    }

    total_sum == 0 ? 1 : total_sum;
    float rate_sum = float(correct_sum) / total_sum;
    printf("total_sum:%d\tcorrect_sum:%d\trate_sum:%.2f\n", total_sum, correct_sum, rate_sum);
    printf("-------------------------------------------------------\n");
}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*ANNChGrayTrainer*/
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
ANNChGrayTrainer::ANNChGrayTrainer(const char *sample_path, const char *xml_path)
{
    this->sample_path = sample_path;
    this->xml_path = xml_path;
    ann_ptr = ANN_MLP::create();
}

ANNChGrayTrainer::~ANNChGrayTrainer() {}

void ANNChGrayTrainer::train()
{
    int input_num = KANNChGrayInputNum;
    int hidden_num = KANNChGrayHiddenNum;
    int output_num = kChineseNumber;

    Mat layers(1, 3, CV_32SC1);
    layers.at<int>(0) = input_num;
    layers.at<int>(1) = hidden_num;
    layers.at<int>(2) = output_num;

    ann_ptr->setLayerSizes(layers);
    ann_ptr->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    ann_ptr->setTrainMethod(ANN_MLP::BACKPROP);
    ann_ptr->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 30000, 0.0001));
    ann_ptr->setBackpropWeightScale(0.1);
    ann_ptr->setBackpropMomentumScale(0.1);

    Ptr<TrainData> train_data_ptr = preprocessTrainData(kANNChGrayMinTrainDataNum);
    fprintf(stdout, "++++++++++++++++++++++++++++++++++++++++\n");
    fprintf(stdout, "ANNChGrayTrainer begin to train,please wait...\n");
    ann_ptr->train(train_data_ptr);
    fprintf(stdout, "saving train result\n");
    fprintf(stdout, "++++++++++++++++++++++++++++++++++++++++\n");
    ann_ptr->save(xml_path);
}

int ANNChGrayTrainer::getBorderColor(const Mat &in)
{
    float sum = 0;
    for (int i = 0; i < in.rows; i++)
    {
        sum += in.at<uchar>(i, 0);
        sum += in.at<uchar>(i, in.cols - 1);
    }
    for (int i = 0; i < in.cols; i++)
    {
        sum += in.at<uchar>(0, i);
        sum += in.at<uchar>(in.rows - 1, i);
    }

    float avg = sum / (in.cols + in.cols + in.rows + in.rows);
    return avg;
}

Mat ANNChGrayTrainer::getSyntheticMat(const Mat &in)
{
    srand(unsigned(time(NULL)));
    int rand_type = rand() % 3;
    int border_color = getBorderColor(in);

    Mat out = in.clone();
    float rand_num = -5.f;
    float rand_array[100];
    for (int i = 0; i < 100; i++)
    {
        rand_array[i] = rand_num;
        rand_num += 0.1f;
    }

    if (rand_type == 0)
    {
        int shift = 2;
        int x = rand() % shift;
        int y = rand() % shift;
        int width = in.cols - rand() % shift;
        int height = in.rows - rand() % shift;
        out = getCropMat(in, x, y, width, height);
    }
    else if (rand_type == 1)
    {
        int rand_x = rand_array[rand() % 20 + 40];
        int rand_y = rand_array[rand() % 20 + 40];
        out = getTranslatedMat(in, rand_x, rand_y, border_color);
    }
    else if (rand_type == 2)
    {
        int angle = rand_array[rand() % 100];
        out = getRotatedMat(in, angle, border_color);
    }

    return out;
}

Ptr<TrainData> ANNChGrayTrainer::preprocessTrainData(int min_input_num)
{
    Mat sample_mat;
    vector<int> label_vec;
    srand(unsigned(time(NULL)));

    for (int i = 0; i < kChineseNumber; i++)
    {
        const char *dir_name = kChars[kCharsTotalNumber - kChineseNumber + i];
        string dir_path;
        dir_path.append(sample_path);
        dir_path.push_back('/');
        dir_path.append(dir_name);
        vector<string> file_path_vec;
        lsDir(dir_path, file_path_vec);

        int input_num = file_path_vec.size();
        vector<Mat> input_mat_vec;
        input_mat_vec.reserve(min_input_num);

        fprintf(stdout, "----------------------------------------\n");
        fprintf(stdout, "loading char:%s\n", dir_path.c_str());
  
        for (int j = 0; j < file_path_vec.size(); j++)
        {
            Mat input_mat = imread(file_path_vec[j].c_str(), CV_8UC1);
      
            input_mat_vec.push_back(input_mat);
        }
  
        for (int j = 0; j < min_input_num - input_num; j++)
        {
            int rand_num = rand() % (j + input_num);
            Mat input_mat = input_mat_vec.at(rand_num); 
       
            Mat synthetic_mat = getSyntheticMat(input_mat);
           
            char output_image_name[512] = {0};
            sprintf(output_image_name, "./res/ANNChGrayTrainer/synthetic_mat/%s-%d-synthetic_mat.jpg", dir_name, j);
            imwrite(output_image_name, synthetic_mat);
           
            input_mat_vec.push_back(synthetic_mat);
        }
       
        fprintf(stdout, "image count:%d\n", input_mat_vec.size());
        for (int j = 0; j < input_mat_vec.size(); j++)
        { 
            Mat feature_mat = charFeaturesForANNChGrayTrain(input_mat_vec.at(j));
            label_vec.push_back(i);
            sample_mat.push_back(feature_mat);
        }
    }
    sample_mat.convertTo(sample_mat, CV_32F);
    Mat class_mat = Mat::zeros(label_vec.size(), kChineseNumber, CV_32F);
    for (int i = 0; i < class_mat.rows; i++)
    {
        class_mat.at<float>(i, label_vec[i]) = 1.f;
    }
    fprintf(stdout, "----------------------------------------\n");
    return TrainData::create(sample_mat, ROW_SAMPLE, class_mat);
}

int ANNChGrayTrainer::identityChinese(const Mat &in)
{
    Mat feature_mat = charFeaturesForANNChGrayTrain(in);
    float max_val = -2.f;
    int result = 0;

    Mat result_mat(1, kChineseNumber, CV_32FC1);
    ann_ptr->predict(feature_mat, result_mat);

    for (int i = 0; i < kChineseNumber; i++)
    {
        float val = result_mat.at<float>(i);
        if (val > max_val)
        {
            max_val = val;
            result = i;
        }
    }
    return result;
}

void ANNChGrayTrainer::test()
{
    srand(unsigned(time(NULL)));
    int correct_sum = 0;
    int total_sum = 0;

    printf("-------------------------------------------------------\n");
    printf("ANNChGrayTrainer testing.....\n");
    for (int i = 0; i < kChineseNumber; i++)
    {
        const char *dir_name = kChars[kCharsTotalNumber - kChineseNumber + i];
        string dir_path;
        dir_path.append(sample_path);
        dir_path.push_back('/');
        dir_path.append(dir_name);
        vector<string> file_path_vec;
        lsDir(dir_path, file_path_vec);

        int correct = 0, total = 0;

        for (int j = 0; j < file_path_vec.size(); j++)
        {
            Mat input_mat = imread(file_path_vec[j], CV_8UC1);
            int result = identityChinese(input_mat);

            if (result == i)
            {
                correct++;
                correct_sum++;
            }
            else
            {
                printf("[%d][%d]",i,result);
                result += kCharsNumber;
                printf("[error]:correct result:%s\t result:%s\n", dir_name, kChars[result]);
            }
            total++;
            total_sum++;
        }
        total == 0 ? 1 : total;
        float rate = float(correct) / total;
        printf("[%s]total:%d\tcorrect:%d\trate:%.2f\n", dir_name, total, correct, rate);
        printf("-------------------------------------------------------\n");
    }

    total_sum == 0 ? 1 : total_sum;
    float rate_sum = float(correct_sum) / total_sum;
    printf("total_sum:%d\tcorrect_sum:%d\trate_sum:%.2f\n", total_sum, correct_sum, rate_sum);
    printf("-------------------------------------------------------\n");
}
