import java.io.IOException;
import java.util.StringTokenizer;
import java.lang.String.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class MainEntry{

	
	static {  
       System.loadLibrary("FaceJniC2");     
  }  
  /* this function read the "srcinput" directory ,and insides it, trains the face classifier from the "start" position to "end" position, 
   * the face classifier stored as .xml in the "StorePathOfFilter" diretory ,the tag marks the face classifier .xml file。
   
     srcinput : the diretctory containing the source face data
     StorePathOfFilter: the diretory of the the classifier created by training face data in HDFS
     start: the start position of the source face data of srcinput
     end: the start position of the source face data of srcinput
     tag: the tag of stored classifier,with which is the file name of classifier  
     //	TrainFaceData("/FaceData","/OuputSet", "0","5","0");
	//	TrainFaceData("/FaceData","/OuputSet", "3","5","1");
	//	TrainFaceData("/FaceData","/OuputSet", "4","5","2");
   *
   */
  public native static  int TrainFaceData(String srcinput,String StorePathOfFilter,String start,String end,String tag);
  
  /* this function read the "srcpath" directory ,it's the absolute path of .pgm file stored in HDFS or other file system  
   * the face classifier stored as .xml in the "PathofFilter" diretory ,the tag marks the face classifier .xml file。
   
     srcpath : the diretctory containing the testing face data
     PathofFilter: the diretory of the the classifier created by training face data
     tag: the tag of stored classifier,with which is the file name of classifier  
   *
   */
  public native static  String RecognizeFace(String srcpath,String PathofFilter,String tag);
  public static class TokenizerMapper extends Mapper<Object, Text, IntWritable, Text>{

    private  static int count = 0;
    private Text word = new Text();  
    
    public void map(Object key, Text value, Context context)throws IOException, InterruptedException {
//training task;
//    	String scope=value.toString();
//    	 String[] pos=scope.split(",");
//    	 TrainFaceData("/orl_faces","/OuputSet", pos[0],pos[1],""+count);
//    	 count++;
    	
    	
//Recognize task;   	 
    	 if(value.toString().isEmpty())return ;
    	 
    	 
    	 String path="/"+value.toString()+"/10.pgm";   
    	 
    	 int a=0;
    	 float b=10000;
    	 
    	 String temp0=RecognizeFace(path,"/OuputSet","0");
    	 String []temp0_a=temp0.split(" \\: ");
    	 if(b>Integer.parseInt(temp0_a[1].trim()))
		 {
			 a=Integer.parseInt(temp0_a[0].trim());
		 }
    	 
    	 String temp1=RecognizeFace(path,"/OuputSet","1");
    	 String []temp1_a=temp1.split(" \\: ");
    	 if(b>Integer.parseInt(temp1_a[1].trim()))
		 {
			 a=Integer.parseInt(temp1_a[0].trim());
		 }
    	 
    	 String temp2=RecognizeFace(path,"/OuputSet","2");
    	 String []temp2_a=temp2.split(" \\: ");
    	 if(b>Integer.parseInt(temp2_a[1].trim()))
		 {
			 a=Integer.parseInt(temp2_a[0].trim());
		 }
    	 
    	 String temp3=RecognizeFace(path,"/OuputSet","3");
    	 String []temp3_a=temp0.split(" \\: ");
    	 if(b>Integer.parseInt(temp3_a[1].trim()))
		 {
			 a=Integer.parseInt(temp3_a[0].trim());
		 }
    	 
    	 String temp4=RecognizeFace(path,"/OuputSet","4");
    	 String []temp4_a=temp1.split(" \\: ");
    	 if(b>Integer.parseInt(temp4_a[1].trim()))
		 {
			 a=Integer.parseInt(temp4_a[0].trim());
		 }
    	 
    	 String temp5=RecognizeFace(path,"/OuputSet","5");
    	 String []temp5_a=temp2.split(" \\: ");
    	 
    	
    		 if(b>Integer.parseInt(temp5_a[1].trim()))
    		 {
    			 a=Integer.parseInt(temp5_a[0].trim());
    		 }
    	
    	 
    	 RecognizeFace(path,"/OuputSet","0");
    	 RecognizeFace(path,"/OuputSet","1");
    	 RecognizeFace(path,"/OuputSet","2");
    	 context.write(new IntWritable(count),new Text(value.toString()+": "+"is recongnized as "+RecognizeFace(path,"/OuputSet","0")
    			 ));
    	 count++;   	
    	  
    }
  } 
  public static class IntSumReducer extends Reducer<IntWritable,Text,IntWritable,Text> {
    private static int result = 0;
	
    public void reduce(IntWritable key, Iterable<Text> values, Context context)  throws IOException, InterruptedException {
//      int sum = 0;    
//     context.write(key,values.iterator().next());
    }
  }
//
  public static void main(String[] args) throws Exception {
	  


    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    //这里需要配置参数即输入和输出的HDFS的文件路径
    if (otherArgs.length != 2) {
      System.err.println("Usage: wordcount <in> <out>");
      System.exit(2);
    }
   // JobConf conf1 = new JobConf(WordCount.class);
    Job job = new Job(conf, "NewProject");//Job(Configuration conf, String jobName) 设置job名称和
    job.setJarByClass(MainEntry.class);
    job.setMapperClass(TokenizerMapper.class); //为job设置Mapper类 
//    job.setCombinerClass(IntSumReducer.class); //为job设置Combiner类  
    job.setReducerClass(IntSumReducer.class); //为job设置Reduce类   
    job.setOutputKeyClass(IntWritable.class);        //设置输出key的类型
    job.setOutputValueClass(Text.class);//  设置输出value的类型
    FileInputFormat.addInputPath(job, new Path(otherArgs[0])); //为map-reduce任务设置InputFormat实现类   设置输入路径   
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));//为map-reduce任务设置OutputFormat实现类  设置输出路径
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}



