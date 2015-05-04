#include "FaceJniC2.h"
#include "FaceRecog.h"
#include "string.h"
#include "hdfs.h"
#include <fstream>
using namespace std;
string   jstringTostring(JNIEnv*   env,   jstring   jstr)
{
   char*   rtn   =   NULL;
   jclass   clsstring   =   env->FindClass("java/lang/String");
   jstring   strencode   =   env->NewStringUTF("GB2312");
   jmethodID   mid   =   env->GetMethodID(clsstring,   "getBytes",   "(Ljava/lang/String;)[B");
   jbyteArray   barr=   (jbyteArray)env->CallObjectMethod(jstr,mid,strencode);
   jsize   alen   =   env->GetArrayLength(barr);
   jbyte*   ba   =   env->GetByteArrayElements(barr,JNI_FALSE);
   if(alen>0)
   {
    rtn   =   (char*)malloc(alen+1);         //new   char[alen+1];
    memcpy(rtn,ba,alen);
    rtn[alen]=0;
   }
   env->ReleaseByteArrayElements(barr,ba,0);
   string stemp(rtn);
   free(rtn);
   return   stemp;
 }

jstring stringtoJstring( JNIEnv* env, const char* pat )
{
 //定义java String类 strClass
 jclass strClass = (env)->FindClass("Ljava/lang/String;");
 //获取java String类方法String(byte[],String)的构造器,用于将本地byte[]数组转换为一个新String
 jmethodID ctorID = (env)->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
 jbyteArray bytes = (env)->NewByteArray(strlen(pat));//建立byte数组
 (env)->SetByteArrayRegion(bytes, 0, strlen(pat), (jbyte*)pat);//将char* 转换为byte数组
 jstring encoding = (env)->NewStringUTF("GB2312"); // 设置String, 保存语言类型,用于byte数组转换至String时的参数
 return (jstring)(env)->NewObject(strClass, ctorID, bytes, encoding);//将byte数组转换为java String,并输出
}
 bool StoreTempFilterXML(hdfsFS fs,const char * path,const char *src_path)
 {
    if(fs==NULL)return false;

    char *buffer=new char[1024*1024*16];

    ifstream in(src_path);

    in.read(buffer,1024*1024*16);

    hdfsFile writeFile = hdfsOpenFile(fs, path, O_WRONLY|O_CREAT, 0, 0, 0);
    if(!writeFile) {
              fprintf(stderr, "Failed to open %s for writing!\n", path);
              exit(-1);
        }

    hdfsWrite(fs, writeFile, (void*)buffer, strlen(buffer)+1);
    if (hdfsFlush(fs, writeFile)) {
               fprintf(stderr, "Failed to 'flush' %s\n", path);
              exit(-1);
    }
    hdfsCloseFile(fs, writeFile);

   delete buffer;
    return true;
 }
int findTag(string tag)
{
   int pos=tag.rfind("/");
   string newtag(tag,pos+2,tag.size());
   return std::atoi( newtag.c_str() );
}
 bool LoadTempFilterXML(hdfsFS fs,const char * path,const char *src_path)
 {
    if(fs==NULL)return false;
    char *buffer=new char[1024*1024*6];
    if(!buffer)
    {
        cout<<"malloc buffer fails"<<endl;
        return false;
    }
    memset(buffer,32,(1024*1024*6)*sizeof(char));
    hdfsFile File = hdfsOpenFile(fs, src_path, O_RDONLY, 0, 0, 0);// 打开文件
    if(!File) {
              fprintf(stderr, "Failed to open %s for writing!\n",src_path);
              exit(-1);
        }

    int len=hdfsRead(fs, File, (void*)buffer, strlen(buffer));// 读入到buffer

    cout<<"size :"<<len<<endl;

    ofstream out(path);
    out.write(buffer,len-2);

    hdfsCloseFile(fs, File);
    delete buffer;
    return true;
 }
 CvMat* loadImageFromHadoop(hdfsFS fs ,const char * path,char *buffer)//fs 是hadoop的句柄，path是hadfs路径， buffer是读取内容缓存区,buffer前面14个字节是pgm图像的标志
{
    hdfsFile File = hdfsOpenFile(fs, path, O_RDONLY, 0, 0, 0);// 打开文件
    if(!File) {
       return NULL;
    }
    memset(buffer,32,(112*92+14)*sizeof(char));

    hdfsRead(fs, File, (void*)buffer, strlen(buffer)+1);// 读入到buffer

    CvMat *m1=cvCreateMat(112,92,CV_64FC1); //利用buffer初始化CvMat
    for(int i=0;i<112;i++)
    {
        for(int j=0;j<92;j++)
        {   int temp1=(unsigned char)(*(buffer+14+i*92+j));
            cvmSet(m1,i,j,(double)temp1);
        }
    }

    std::cout<<"hdfs read Tesing Sample: "<<path<<" Completed "<<endl;

    hdfsCloseFile(fs, File);
    return m1;


}
JNIEXPORT jint JNICALL Java_MainEntry_TrainFaceData
  (JNIEnv *env, jclass, jstring srcpath ,jstring storePath,jstring start,jstring end,jstring index)
{
    int start_step=atoi(jstringTostring(env,start).c_str());
    int end_step=atoi(jstringTostring(env,end).c_str()); //把分类器分为times个分片
    string pathOfFilter=jstringTostring(env,storePath); //分类器的存储路径


    hdfsFS fs = hdfsConnect("192.168.164.200", 9000);//连接hadoop
    hdfsSetWorkingDirectory(fs, jstringTostring(env,srcpath).data());//设置当前工作路径

    char *buffer=new char[112*92+14];//分配缓存区域

    int numberOfDir=0; //人脸文件夹的个数
    hdfsFileInfo *listOfDir=hdfsListDirectory(fs,jstringTostring(env,srcpath).data(),&numberOfDir); //获取指定目录的所有子文件夹listOfDir

/*    if(splits>numberOfDir)
    {   cout<<"supply the 3th illegal parameter "<<endl;
        return 0;
    }

    splits=numberOfDir/times; //splits*times+extra =numberOfDir

    int count_times=0;  //第times个分片
  */
    int PersonDirCount=start_step;  //遍历所有子文件夹数量，临时变量PersonDirCount


    string file_template="MyFaceProgram";
    string file_template2="_MyFaceProgram";
    string file_end=".xml";

    FaceRecog f; //声明一个新的人脸识别对象
    while(PersonDirCount<=end_step&&PersonDirCount<numberOfDir)//遍历子文件夹s[PersonDirCount]的内容
    {
            int numberOfFaceFile=0;
            hdfsFileInfo *listOfFace=hdfsListDirectory(fs,listOfDir[PersonDirCount].mName,&numberOfFaceFile);//获取子文件夹s[PersonDirCount]的内容ListofFace


            string tag=listOfDir[PersonDirCount].mName;
            int int_tag=findTag(tag);
            int facefileCount=0; //FaceData/s[PersonDirCount]/*中的所有内容
            while(facefileCount<numberOfFaceFile-1)f.addData(loadImageFromHadoop(fs,listOfFace[facefileCount++].mName,buffer), int_tag);
            PersonDirCount++;
    }

    f.calEigensAndAvg(4);
    f.project2DPCA();
    f.calFisherFace(1);//1 暂时无意义


    file_template+=jstringTostring(env,index)+file_end;
    file_template2+=jstringTostring(env,index)+file_end;


    f.saveTrain(file_template.c_str());
    StoreTempFilterXML(fs,(pathOfFilter+"/"+file_template).c_str(),file_template.c_str());//从本地保存到HDFS的LDA的训练集
    StoreTempFilterXML(fs,(pathOfFilter+"/"+file_template2).c_str(),file_template2.c_str());//从本地保存到HDFS的2DPCA到LDA中间的映射空间


    delete buffer;
    hdfsDisconnect(fs);

    return 1;

}

JNIEXPORT jstring JNICALL Java_MainEntry_RecognizeFace
  (JNIEnv *env, jclass, jstring srcpath,jstring PathOfFilter,jstring IndexOfFilter)
{
    hdfsFS fs = hdfsConnect("192.168.164.200", 9000);//连接hadoop
    if(!fs){cout<<"fail to connect hdfs"<<endl;return 0;}

    string index=jstringTostring(env,IndexOfFilter);
    string file_template="MyFaceProgram";
    string file_template2="_MyFaceProgram";
    string file_end=".xml";
    string pathOfFilter=jstringTostring(env,PathOfFilter);

    file_template+=index+file_end;
    file_template2+=index+file_end;

    FaceRecog f;
    if(!f.loadTrain(file_template.c_str())){
        cout<<"try to import the "<<file_template<< "from hdfs to local fs"<<endl;                            //将Simpevector和LDA的训练集都导入
        LoadTempFilterXML(fs,file_template.c_str(),(pathOfFilter+"/"+file_template).c_str()); //从HDFS到本地目录
        LoadTempFilterXML(fs,file_template2.c_str(),(pathOfFilter+"/"+file_template2).c_str());
        }

    if(!f.loadTrain(file_template.c_str())){                              //将Simpevector和LDA的训练集都导入
            fprintf(stderr, "Failed to open MyFaceProgram.xml for writing!\n");
            exit(-1);
        }

    char *buffer=new char[112*92+14];//分配缓存区域
    string result=f.dataPredict(loadImageFromHadoop(fs,jstringTostring(env,srcpath).data(),buffer));

    cout<<"RecognizeFace is "<<result<<endl;

    hdfsDisconnect(fs);

    delete buffer;
    return stringtoJstring(env,result.c_str());
}

