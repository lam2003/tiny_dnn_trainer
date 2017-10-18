#include "pr.h"

int main(int argc,char *argv[])
{
    Mat in = imread(argv[1]);
    Locater locater(in,true);
    locater.mserCharLocated();
    return 0;
}