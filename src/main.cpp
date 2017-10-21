#include "pr.h"

int main(int argc, char *argv[])
{


    CNNTrainer cnn_trainer;
    cnn_trainer.preprocessTrainData();
    cnn_trainer.train();
/*
    vector<string> file_paths;
    lsDir(string(argv[1]),file_paths);
    for(int i = 0; i < file_paths.size(); i++)
    {
        Mat in = imread(file_paths[i],CV_8UC1);
        cnn_trainer.recognize(in);
    }
*/
    return 0;
}