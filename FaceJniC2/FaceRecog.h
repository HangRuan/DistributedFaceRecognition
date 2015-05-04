#ifndef Face
#define Face
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <stdio.h>
#include "cv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "string.h"
#define SRCDATA 1
#define LABEL 2
#define EIGENVECTOR 3
#define EIGENVALUE 4
#define SIMPLEVECTOR 5
#define PROJECT 6
#endif
//思路是2DPCA：按照同一个人人脸库建立摄影空间，然后待识别图片与每一类人脸摄影空间进行投影，计算欧式距离，找出最接近的人脸类
void getCols(CvMat *&data,CvMat *dest,int start,int end);
inline void printCvMat(const CvMat* data,int height,int width )
{   if(width>data->cols)width=data->cols;
    if(height>data->rows)height=data->rows;
    if(!width||!height)return ;
      std::cout<<data->rows<<" * "<<data->cols<<" step "<<data->step<<std::endl;

    for(int i=0;i<height;i++)
     {
          for(int j=0;j<width;j++)
        {
            std:: cout<< cvmGet(data,i,j)<<" ";

        }
       std:: cout<<std::endl;
     }

    std:: cout<<std::endl;
}
class FaceRecog
{
    public :
    FaceRecog () ;

    FaceRecog(const FaceRecog& tempFaceRecog);

    virtual  ~ FaceRecog();
    bool FaceDataReadFS(std::string InputPath);//从本地读取原始人脸数据

    bool store2FS(std::vector<CvMat *> project_y,std::vector<int> label_temp,std::string InputPath="./ProjectY/");//将ProjectY投影向量存储在指定位置

    bool ProjectYReadFS();//从本地读取训练好的2DPCA的ProjectY投影向量




    bool addData(CvMat *image,int tag);

    bool calEigensAndAvg(int numberOfProjectEigen);

    std::string dataPredict(CvMat *img);

    bool echoData(int tag,unsigned n,int width,int height);

    bool project2DPCA();

    bool calFisherFace(int numberOfProjectEigen);

    bool saveTrain(const char *path);

    bool loadTrain(const char *path);

   private :
   std:: vector<CvMat *> srcdata;

   std:: vector<int> label;
   std:: vector<int > SubCountS;
   std:: vector<CvMat *> EigenVector;//特征向量按行存储

   std:: vector<CvMat *> EigenValue;
   std:: vector<CvMat *> ProjectY;
   std:: vector<CvMat *> SimpleEigenVectors;
   std:: vector<CvMat *> SubMean;
   cv::Ptr<cv::FaceRecognizer> model;
   std::  vector<cv::Mat> FisherPro;
   int classes;

};
