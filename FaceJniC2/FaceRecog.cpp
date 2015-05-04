#include "FaceRecog.h"
using namespace std;
using namespace cv;
void getCols(CvMat *&data,CvMat *dest,int start,int end)
{   if(dest->rows!=data->rows|| dest->cols<(end-start))return;
    if(end<=start)return;
    int i=0,count=0, j=0;
    for(;i<data->rows;i++)
    {   j=start;
        count=0;
        for(; j<=end ; )
        {
//          dest->data.db[i*dest->cols+count]=data->data.db[i*data->cols+j];
                cvmSet(dest,i,count,cvmGet(data,i,j));
          j++;count++;
        }

    }
//    dest->cols=end-start+1;


}
FaceRecog::FaceRecog()
{
    classes=0;
}

FaceRecog::FaceRecog(const FaceRecog& tempFaceRecog) //深拷贝一下
{
    srcdata.assign(tempFaceRecog.srcdata.begin(),tempFaceRecog.srcdata.end());
    label.assign(tempFaceRecog.label.begin(),tempFaceRecog.label.end());
    //分配长度
    EigenValue.assign(tempFaceRecog.EigenValue.begin(),tempFaceRecog.EigenValue.end());
    EigenVector.assign(tempFaceRecog.EigenVector.begin(),tempFaceRecog.EigenVector.end());

    ProjectY.assign(tempFaceRecog.ProjectY.begin(),tempFaceRecog.ProjectY.end());
    SimpleEigenVectors.assign(tempFaceRecog.SimpleEigenVectors.begin(),tempFaceRecog.SimpleEigenVectors.end());

    vector<CvMat*>::iterator iterdest=EigenValue.begin();
    vector<CvMat*>::const_iterator itersrc=tempFaceRecog.EigenValue.begin();

    vector<CvMat*>::iterator imagedest=srcdata.begin();
    vector<CvMat*>::const_iterator imagesrc=tempFaceRecog.srcdata.begin();
    //分配长度

    //复制内容
    for(    ;iterdest!=EigenValue.end();
            iterdest++,itersrc++)
    {
           *iterdest=cvCloneMat(*itersrc);
    }

    iterdest=EigenVector.begin();
    itersrc=tempFaceRecog.EigenVector.begin();

    for(   ;iterdest!=EigenVector.end();
            iterdest ++,itersrc++)
    {
           *iterdest=cvCloneMat(*itersrc);
    }

    imagedest=srcdata.begin();
    imagesrc=tempFaceRecog.srcdata.begin();
    for(   ;imagedest!=srcdata.end();
            imagedest ++,imagesrc++)
    {
           *imagedest=cvCloneMat(*imagesrc);
    }

    imagedest=ProjectY.begin();
    imagesrc=tempFaceRecog.ProjectY.begin();
    for(   ;imagedest!=ProjectY.end();
            imagedest ++,imagesrc++)
    {
           *imagedest=cvCloneMat(*imagesrc);
    }

    imagedest=SimpleEigenVectors.begin();
    imagesrc=tempFaceRecog.SimpleEigenVectors.begin();
    for(   ;imagedest!=SimpleEigenVectors.end();
            imagedest ++,imagesrc++)
    {
           *imagedest=cvCloneMat(*imagesrc);
    }

}
FaceRecog:: ~ FaceRecog()
{

    for(vector<CvMat *>::iterator iter=EigenVector.begin(); iter!=EigenVector.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
     for(vector<CvMat *>::iterator iter=EigenValue.begin(); iter!=EigenValue.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
     for(vector<CvMat *>::iterator iter=srcdata.begin(); iter!=srcdata.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
    for(vector<CvMat *>::iterator iter=SimpleEigenVectors.begin(); iter!=SimpleEigenVectors.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
     for(vector<CvMat *>::iterator iter=ProjectY.begin(); iter!=ProjectY.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
      for(vector<CvMat *>::iterator iter=SubMean.begin(); iter!=SubMean.end();iter ++)
    {
            cvReleaseMat(&(*iter));
    }
    SubMean.clear();
    EigenValue.clear();
    EigenVector.clear();
    srcdata.clear();
    label.clear();
    SimpleEigenVectors.clear();
    ProjectY.clear();
}

bool FaceRecog::addData(CvMat *img , int tag)
{
    if(!srcdata.empty() && (srcdata[0]->rows!=img->rows || srcdata[0]->cols!= img->cols)){//重新缩放img大小
     cout<<"image size is not standard and has been modified"<<endl;
     CvMat *f=cvCreateMat(srcdata[0]->rows,srcdata[0]->cols,CV_64FC1);
     cvResize(img,f,CV_INTER_LINEAR);
     srcdata.push_back(f);
     label.push_back(tag);
    }
    else{
    CvMat *f=cvCreateMat(img->rows,img->cols,CV_64FC1);
    cvConvertScale(img,f,1,0);
    srcdata.push_back(f);
    label.push_back(tag);

    }
   return true;
}

bool FaceRecog::calEigensAndAvg(int numberOfProjectEigen)//为每个C类人脸集算出一个投影方式，然后用待识别头像一次与每个人脸空间摄像
{       numberOfProjectEigen=srcdata.size();
        if(!srcdata.empty() && srcdata.size()==label.size()){

        CvMat *MatSum=cvCreateMat(srcdata[0]->rows,srcdata[0]->cols,CV_64FC1);
        CvMat *MatMean=cvCreateMat(srcdata[0]->rows,srcdata[0]->cols,CV_64FC1);
        CvMat *MatSubResult=cvCreateMat(srcdata[0]->rows,srcdata[0]->cols,CV_64FC1);
        CvMat *MatMutiResult=cvCreateMat(srcdata[0]->cols,srcdata[0]->cols,CV_64FC1);
        CvMat *MatG=cvCreateMat(srcdata[0]->cols,srcdata[0]->cols,CV_64FC1);
        CvMat *MatEigenVec=cvCreateMat(srcdata[0]->cols,srcdata[0]->cols,CV_64FC1);
        CvMat *MatEigenVau=cvCreateMat(1,srcdata[0]->cols,CV_64FC1);
        cvConvertScale(MatSum,MatSum,0,0);
        cvConvertScale(MatMean,MatMean,0,0);
        cvConvertScale(MatSubResult,MatSubResult,0,0);
        cvConvertScale(MatMutiResult,MatMutiResult,0,0);
        cvConvertScale(MatG,MatG,0,0);
        cvConvertScale(MatEigenVau,MatEigenVau,0,0);
        cvConvertScale(MatEigenVec,MatEigenVec,0,0);

        vector<CvMat *>::iterator iterSrc=srcdata.begin();

        sort(label.begin(),label.end());
        int count=0;
        while(iterSrc!=srcdata.end()){ cvAdd(*iterSrc,MatSum,MatSum);count++;iterSrc++;}
        cvConvertScale(MatSum,MatMean,1.0/count,0);
        for(int i=0;i<count;i++)
        {
                 cvSub(srcdata[i],MatMean,MatSubResult);//X-pingjun=MatSub
                 cvMulTransposed(MatSubResult,MatMutiResult,1,NULL,1.0/count);// MatSubT* MatSub *1/ N
                 cvAdd(MatMutiResult,MatG,MatG);//accmulate MatG协防差矩阵
        }

      cvEigenVV(MatG,MatEigenVec,MatEigenVau);
      EigenValue.push_back(cvCloneMat(MatEigenVau));//MatG的特征值
      EigenVector.push_back(cvCloneMat(MatEigenVec));//MatG的特征向量

      if(numberOfProjectEigen>MatEigenVec->cols)numberOfProjectEigen=MatEigenVec->cols;//取前numberOfProjectEigen个特征向量
      CvMat *reverse=cvCreateMat(MatEigenVec->rows,MatEigenVec->cols,CV_64FC1);
      CvMat *projectEigen=cvCreateMat(MatEigenVec->rows,numberOfProjectEigen,CV_64FC1);
      cvTranspose(MatEigenVec,reverse);
      getCols(reverse,projectEigen,0,numberOfProjectEigen-1);

      SimpleEigenVectors.push_back(cvCloneMat(projectEigen));//投影空间

     cvReleaseMat(&reverse);
     cvReleaseMat(&projectEigen);
     cvReleaseMat(&MatSum);
     cvReleaseMat(&MatMean);
     cvReleaseMat(&MatSubResult);
     cvReleaseMat(&MatMutiResult);
     cvReleaseMat(&MatG);
     cvReleaseMat(&MatEigenVau);
     cvReleaseMat(&MatEigenVec);

    }

    return true;
}
bool FaceRecog:: project2DPCA()
{
       if(SimpleEigenVectors.empty())return false;
        vector<CvMat *>::iterator iterSrc=srcdata.begin();
        while(iterSrc!=srcdata.end())
        {

                    CvMat *dest=cvCreateMat(srcdata[0]->rows,SimpleEigenVectors[0]->cols,CV_64FC1);

                    cvmMul(*iterSrc++,SimpleEigenVectors[0],dest);
                    ProjectY.push_back(dest);

        }
        return true;
  }
bool FaceRecog::calFisherFace(int numberOfProjectEigen)//为每个C类人脸集算出一个投影方式，然后用待识别头像一次与每个人脸空间摄像
{
     model=createFisherFaceRecognizer();
     vector<CvMat *>::iterator iter=ProjectY.begin();
     while(iter!=ProjectY.end())
     {      cv::Mat a=Mat(*iter,true);
            FisherPro.push_back(a);
            iter++;
     }
    model->train(FisherPro, label);
    return true;
}

string FaceRecog::dataPredict(CvMat *img)
{
        if(SimpleEigenVectors.size()==0)return 0;

        CvMat *newimage=cvCreateMat(img->rows,img->cols,CV_64FC1);
        cvConvertScale(img,newimage,1,0);

        CvMat *dest=cvCreateMat(newimage->rows,SimpleEigenVectors[0]->cols,CV_64FC1);
        cvmMul(newimage,SimpleEigenVectors[0],dest);


        cv::Mat a=Mat(dest,true);

        int predicted=0;
        double possibility=0;

        std::cout<<"Start recognizing Sample: "<<endl;

        model->predict(a,predicted,possibility);


        char temp[64];
        sprintf(temp,"%d : %f",predicted+1,possibility);
        string result=temp;


        return result;
}

bool FaceRecog::saveTrain(const char *path)
{
  if(path==NULL)return false;

  if(model!=NULL)model->save(path);

  string path2=path;
  path2.insert(0,"_");


  cvSave(path2.data(),SimpleEigenVectors[0]); //保存SimpleVector

  return true;
}

bool FaceRecog::loadTrain(const char *path)
{
   model=createFisherFaceRecognizer();
   try{
       model->load(path);
   }catch(exception e){
        cout<<"cannot find the "<<path<< "in local FS"<<endl;
        return false;
   }


   string path2=path;
   path2.insert(0,"_");


   CvMat *simplevector=(CvMat*)cvLoad(path2.data());
   if(simplevector==NULL)return false;

   SimpleEigenVectors.push_back(simplevector); //读取到SimpleVector

   return true;
}
 bool FaceRecog::echoData(int tag,unsigned int n,int width,int height)
 {
     switch (tag){

        case SRCDATA: if(n<srcdata.size())printCvMat(srcdata[n],width,height);
                      else cout<<"srcdata size is "<<srcdata.size()<<" not find the"<<n<<" elment"<<endl;break;
        case LABEL:cout<<"lable the "<<n<<"th number is "<<label[n]<<endl;break;
        case EIGENVECTOR:if(n<EigenVector.size())printCvMat(EigenVector[n],width,height);
                      else cout<<"EigenVector size is "<<EigenVector.size()<<" not find the"<<n<<" elment"<<endl;break;
        case EIGENVALUE:if(n<EigenValue.size())printCvMat(EigenValue[n],width,height);
                      else cout<<"EigenValue size is "<<EigenValue.size()<<" not find the"<<n<<" elment"<<endl;break;
        case SIMPLEVECTOR:if(n<SimpleEigenVectors.size())printCvMat(SimpleEigenVectors[n],width,height);
                      else cout<<"SimpleEigenVectors size is "<<SimpleEigenVectors.size()<<" not find the"<<n<<" elment"<<endl;break;
        case PROJECT:if(n<ProjectY.size())printCvMat(ProjectY[n],width,height);
                      else cout<<"ProjectY size is "<<ProjectY.size()<<" not find the"<<n<<" elment"<<endl;break;
        }
     return true;

 }
 bool FaceRecog::FaceDataReadFS(std::string InputPath)//从本地读取原始人脸数据
 {
    DIR    *dir,*sub_dir;
    struct    dirent    *ptr, *sub_ptr;
    dir = opendir((const char *)InputPath.data()); ///open the dir
    int count=0;
    while((ptr = readdir(dir)) != NULL) ///read the list of this dir
    {
        std::string subpath=InputPath+"/"+ptr->d_name;
        sub_dir=opendir((const char *)subpath.data());
        while((sub_ptr=readdir(sub_dir))!=NULL)
        {

            cout<<sub_ptr->d_name<<endl;
            count++;
        }

    }
    closedir(dir);
    return 0;




 }

bool FaceRecog::store2FS(std::vector<CvMat *> project_y,std::vector<int> label_temp,std::string InputPath)//将ProjectY投影向量存储在指定位置
{
    return true;

}

bool FaceRecog::ProjectYReadFS()//从本地读取训练好的2DPCA的ProjectY投影向量
{
    return true;
}
