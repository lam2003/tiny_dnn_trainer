#include "trainer.h"

int main(int argc,char *argv[])
{
    ANNTrainer ann_trainer(argv[1],argv[2],atoi(argv[3]));
    ann_trainer.train();
    ann_trainer.test();
  //  ANNChGrayTrainer ann_ch_gray_trainer(argv[1],argv[2]);
  //  ann_ch_gray_trainer.train();
  //  ann_ch_gray_trainer.test();
    return 0;
}