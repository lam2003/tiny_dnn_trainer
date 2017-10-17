#ifndef GLOBAL_H
#define GLOBAL_H

static const char *kChars[] = 
{
    "0", "1", "2",
    "3", "4", "5",
    "6", "7", "8",
    "9",
    /*  10  */
    "A", "B", "C",
    "D", "E", "F",
    "G", "H", 
    "J", "K", "L",
    "M", "N", 
    "P", "Q", "R",
    "S", "T", "U",
    "V", "W", "X",
    "Y", "Z",
    /*  24  */
    "zh_cuan", "zh_e", "zh_gan",
    "zh_gan1", "zh_gui", "zh_gui1",
    "zh_hei", "zh_hu", "zh_ji",
    "zh_jin", "zh_jing", "zh_jl",
    "zh_liao", "zh_lu", "zh_meng",
    "zh_min", "zh_ning", "zh_qing",
    "zh_qiong", "zh_shan", "zh_su",
    "zh_sx", "zh_wan", "zh_xiang",
    "zh_xin", "zh_yu", "zh_yu1",
    "zh_yue", "zh_yun", "zh_zang",
    "zh_zhe"
};

const int kCharsNumber = 34;
const int kChineseNumber = 31;
const int kCharsTotalNumber = 65;



const int kANNCharDataSize = 10;
const int kANNChineseDataSize = 20;
const int kANNHiddenNum = 40;
const int kANNCharInputNum = 120;
const int kANNChineseInputNum = 440;

const int kANNMinTrainDataNum = 1000;
const int kANNChGrayMinTrainDataNum = 1000;

const int kANNChGrayHeight = 20;
const int KANNChGrayWidth = 32;
const int KANNChGrayInputNum = KANNChGrayWidth * kANNChGrayHeight + 64;
const int KANNChGrayHiddenNum = 64;
const int kANNChGrayDataSize = 32;
#endif 