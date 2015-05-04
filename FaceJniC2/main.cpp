#include "FaceJniC2.cpp"
int main()
{
    FaceRecog f;
    hdfsFS fs = hdfsConnect("192.168.164.200", 9000);//连接hadoop
    char *buffer=new char[8*92*112];
    f.addData(loadImageFromHadoop(fs,"/s1/1.pgm",buffer),0);
    f.addData(loadImageFromHadoop(fs,"/s1/2.pgm",buffer),0);



    f.addData(loadImageFromHadoop(fs,"/s2/1.pgm",buffer),1);
    f.addData(loadImageFromHadoop(fs,"/s2/2.pgm",buffer),1);


    f.addData(loadImageFromHadoop(fs,"/s3/1.pgm",buffer),2);
    f.addData(loadImageFromHadoop(fs,"/s3/2.pgm",buffer),2);


    f.calEigensAndAvg(6);
    f.project2DPCA();
    f.calFisherFace(1);//1 暂时无意义

    f.dataPredict(loadImageFromHadoop(fs,"/s1/1.pgm",buffer));
}

