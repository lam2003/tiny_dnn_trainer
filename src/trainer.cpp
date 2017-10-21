

 #include "trainer.h"

 CNNTrainer::CNNTrainer()
 {
    
#define O true
#define X false
    static const bool tbl[] = {
         O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
         O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
         O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
         X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
         X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
         X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X
    using conv     = tiny_dnn::layers::conv;
    using ave_pool = tiny_dnn::layers::ave_pool;
    using fc       = tiny_dnn::layers::fc;
    using tanh     = tiny_dnn::activation::tanh;
    using tiny_dnn::core::connection_table;
    
     net << conv(32, 32, 5, 1, 6) // C1,32x32 in,1 input,6 output
         << tanh(28, 28, 6)
         << ave_pool(28, 28, 6, 2) // S2,24x24 in,2x2 kernel,6 input
         << tanh(14, 14, 6)
         << conv(14, 14, 5, 6, 16,connection_table(tbl, 6, 16)) //C3,12x12 in,5x5 kernel,6 input,6 output
         << tanh(10, 10, 16)
         << ave_pool(10, 10, 16, 2) //S4,8x8 in,2x2 kernel,16 input
         << tanh(5, 5, 16)
         << conv(5, 5, 5, 16, 120) //C5 4x4 in,4x4 kernel, 16 input,120 output;
         << tanh(1, 1, 120)
         << fc(120, 65)
         << tanh(65);

 }

 Mat CNNTrainer::getSyntheticMat(const Mat &in)
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

void CNNTrainer::preprocessTrainData()
{

    for(int i = 0; i < kCharsTotalNumber; i++)
    {
        const char *sample_name = kChars[i];
        char char_folder_path[512];
        sprintf(char_folder_path,"%s/%s",kCNNSamplePath,sample_name);
        fprintf(stdout,"%s/%s\n",kCNNSamplePath,sample_name);

        vector<string> file_paths;
        lsDir(string(char_folder_path),file_paths);

        vector<Mat> sample_mat_vec;

        for(int j = 0; j < file_paths.size(); j++)
        {
            Mat sample_mat = imread(file_paths[j].c_str(),CV_8UC1);
            if(sample_mat.empty())
                continue;
            sample_mat_vec.push_back(sample_mat);
        }
     
        int image_num = sample_mat_vec.size();
        srand(unsigned(time(NULL)));

        for(int j = 0; j < kCNNMinTrainDataNum - image_num; j++)
        {
            
            int rand_num = rand() % (j + image_num);

            Mat synthetic_mat = getSyntheticMat(sample_mat_vec[rand_num]);
            sample_mat_vec.push_back(synthetic_mat);

        }

        for (int j = 0; j < sample_mat_vec.size(); j++)
        {
          
            charFeatureForCNN(sample_mat_vec[j],32,-1.0,1.0,train_images);
            train_labels.push_back(i);
            if((j + 1) % 3 == 0)
            {
                charFeatureForCNN(sample_mat_vec[j],32,-1.0,1.0,test_images);
                test_labels.push_back(i);
            }
           
        }
    }

}

 void CNNTrainer::train()
 {
    
    cout << "start training" << endl;
    progress_display disp(static_cast<unsigned long>(train_images.size()));
    timer t;
    int minibatch_size = 100;
    int num_epochs = 50;

    //optimizer.alpha *= static_cast<float_t>(sqrt(minibatch_size));

    auto on_enumerate_epoch = [&](){
        cout << t.elapsed() << "s elapsed." << endl;
        result res = net.test(test_images, test_labels);
        cout << res.num_success << "/" << res.num_total << endl;

        disp.restart(static_cast<unsigned long>(train_images.size()));
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    net.train<mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
        on_enumerate_minibatch, on_enumerate_epoch);

    cout << "end training." << endl;

    net.test(test_images, test_labels).print_detail(cout);
    net.save(kCNNModelPath);
 }

 void CNNTrainer::recognize(const Mat &in) {
   
    net.load(kCNNModelPath);

    vector<vec_t> data_vec;
    
    charFeatureForCNN(in,32,-1,1,data_vec);
    
    for(int j = 0; j < data_vec.size(); j++)
    {
        vec_t data = data_vec[j];
        auto res = net.predict(data);
        vector<pair<double, int> > scores;

    
        for (int i = 0; i < 65; i++)
            scores.emplace_back(res[i], i);

        sort(scores.begin(), scores.end(), greater<pair<double, int>>());

        for (int i = 0; i < 1; i++)
            cout << kChars[scores[i].second] << "," << scores[i].first << endl;
    }
}
