#include "stdafx.h"
#include "HU.h"
#include<io.h>
using namespace cv;
using namespace std;

HU::HU(void)
{
}


HU::~HU(void)
{
}
int & HU:: ExcuteHu(Mat img, void * param)
{
	//变量
 Mat g_grayImage;
int g_nThresh = 100;
int g_nMaxThresh = 255;
RNG g_rng(12345);
Mat g_cannyMat_output;
vector<vector<Point> > g_vContours;
	//定义矩变量
CvMoments moment;
//double m00,m10,m01;
 Mat src ;
vector<Vec4i> g_vHierarchy;

	cvtColor( img, g_grayImage, CV_BGR2GRAY);
	blur( g_grayImage, g_grayImage, Size(3,3) );
	


int bmpWidth =g_grayImage.cols;
int bmpHeight =g_grayImage.rows;
int bmpStep = g_grayImage.step;
int bmpChannels =g_grayImage.channels();
uchar*pBmpBuf = (uchar*)g_grayImage.data;
 
 double m00=0,m11=0,m20=0,m02=0,m30=0,m03=0,m12=0,m21=0; //中心矩
 double x0=0,y0=0; //计算中心距时所使用的临时变量（x-x'）
 double u20=0,u02=0,u11=0,u30=0,u03=0,u12=0,u21=0;//规范化后的中心矩
 //double M[7]; //HU不变矩
 double t1=0,t2=0,t3=0,t4=0,t5=0;//临时变量，
 //double Center_x=0,Center_y=0;//重心
 int Center_x=0,Center_y=0;//重心
 int i,j; //循环变量
 
// 获得图像的区域重心
double s10=0,s01=0,s00=0; //0阶矩和1阶矩 //注：二值图像的0阶矩表示面积
 for(j=0;j<bmpHeight;j++)//y
 {
 for(i=0;i<bmpWidth;i++)//x
 {
 s10+=i*pBmpBuf[j*bmpStep+i];
 s01+=j*pBmpBuf[j*bmpStep+i];
 s00+=pBmpBuf[j*bmpStep+i];
 }
 }
 Center_x=(int)(s10/s00+0.5);
 Center_y=(int)(s01/s00+0.5);
 
// 计算二阶、三阶矩
 m00=s00;
 for(j=0;j<bmpHeight;j++)
 {
 for(i=0;i<bmpWidth;i++)//x
 {
 x0=(i-Center_x);
y0=(j-Center_y);
 m11+=x0*y0*pBmpBuf[j*bmpStep+i];
 m20+=x0*x0*pBmpBuf[j*bmpStep+i];
 m02+=y0*y0*pBmpBuf[j*bmpStep+i];
 m03+=y0*y0*y0*pBmpBuf[j*bmpStep+i];
 m30+=x0*x0*x0*pBmpBuf[j*bmpStep+i];
 m12+=x0*y0*y0*pBmpBuf[j*bmpStep+i];
 m21+=x0*x0*y0*pBmpBuf[j*bmpStep+i];
 }
 } 

 // 计算规范化后的中心矩
 u20=m20/pow(m00,2);
 u02=m02/pow(m00,2);
 u11=m11/pow(m00,2);
 u30=m30/pow(m00,2.5);
 u03=m03/pow(m00,2.5);
 u12=m12/pow(m00,2.5);
 u21=m21/pow(m00,2.5);

 // 计算中间变量。
 t1=(u20-u02);
 t2=(u30-3*u12);
 t3=(3*u21-u03);
 t4=(u30+u12);
 t5=(u21+u03);

 // 计算不变矩
 double M[7];
 M[0]=u20+u02;
 M[1]=t1*t1+4*u11*u11;
 M[2]=t2*t2+t3*t3;
 M[3]=t4*t4+t5*t5;
 M[4]=t2*t4*(t4*t4-3*t5*t5)+t3*t5*(3*t4*t4-t5*t5);
 M[5]=t1*(t4*t4-t5*t5)+4*u11*t4*t5;
 M[6]=t3*t4*(t4*t4-3*t5*t5)-t2*t5*(3*t4*t4-t5*t5);

 	//param =  (double*)malloc(sizeof(M));如果报错则去掉注释
	  memcpy(param,M,sizeof(M));
	 
  
 
	int length_Hu = sizeof(M);

	return length_Hu;
}
double HU::MatchingHu(float *data1, float *data2)
{
	


	  double dbR2 =0; //相似度
  double temp2 =0;
  double temp3 =0;  
      {for(int i=0;i<7;i++)
      {
        temp2 += fabs(data1[i]-data2[i]);
          temp3 += fabs(data1[i]+data2[i]);
     }}
    dbR2 =1- (temp2*1.0)/(temp3);
	int length_match = 0;
	return    dbR2;
}
int & HU::ExcuteFFT(Mat img,void * param)
{

 
    

    cvtColor(img,img,CV_BGR2GRAY);
	GaussianBlur(img,img,Size(3,3),0,0);
	Canny(img,img,3,9,3);
    //imshow("src",img);
	vector<vector<Point>> contours_dst;
	vector<Vec4i> hierarchy;
	findContours(img,contours_dst, hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	
   img.convertTo(img,CV_32FC1);
    for(int i=0; i<img.rows; i++)        //中心化
    {
        float *p = img.ptr<float>(i);
        for(int j=0; j<img.cols; j++)
        {
            p[j] = p[j] * pow(-1, i+j);
        }
    }
    
        

vector<Point>wer;

 for(vector<vector<Point>>::iterator it=contours_dst.begin();it!=contours_dst.end();it++)  //判断外层循环  
{  
	for(vector<Point>::iterator it1=it->begin();it1!=it->end();it1++)  //判断内层循环  
	{
		
		wer.push_back(*it1); 
		
		
		 
	}
   
}  

int length = EllipticFourierDescriptor(wer,param) ; 


	waitKey(0);  
	
	return length;
}
int & HU::EllipticFourierDescriptor(vector<Point> &contour,void *param)  
{  
    vector<float> ax,bx,ay,by;  
    int m=contour.size();  
    int n=20;  
    float t=(2*3.1415926535)/m;  
    for(int k=0;k<n;k++)  
    {  
        ax.push_back(0.0);  
        bx.push_back(0.0);  
        ay.push_back(0.0);  
        by.push_back(0.0);  
        for(int i=0;i<m;i++)  
        {  
            ax[k]=ax[k]+contour[i].x*cos((k+1)*t*i);  
            bx[k]=bx[k]+contour[i].x*sin((k+1)*t*i);  
            ay[k]=ay[k]+contour[i].y*cos((k+1)*t*i);  
            by[k]=by[k]+contour[i].y*sin((k+1)*t*i);  
        }  
        ax[k]=ax[k]/m;  
        bx[k]=bx[k]/m;  
        ay[k]=ay[k]/m;  
        by[k]=by[k]/m;        
    }  
  vector<float> CE;
    for(int k=0;k<n;k++)  
    {  
        float value=(float)sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(ax[0]*ax[0]+ay[0]*ay[0]))+sqrt((bx[k]*bx[k]+by[k]*by[k])/(bx[0]*bx[0]+by[0]*by[0]));  
        CE.push_back(value);  
  
    }  
	
   
	float result [20];
	for(int i = 0;i<20;i++)
	{
		result[i] = CE[i];
	}
	// param = (float*)malloc(sizeof(result));如果有问题就把它前面的注释去掉
	  memcpy(param,result,sizeof(result));
 
	int length_CE = sizeof(CE);
	return length_CE;
} 
float HU::MatchingFFT(void *data1, void *data2)
{
	float *M1 = new  float [20];
	float *M2 = new  float [20];
   float *p1 = new  float [20];
    float *p2 = new  float [20];
   p1 =   static_cast<float *>(data1);
   p2 = static_cast<float *>(data2);

 for(int i=0;i<20;i++)
 {
	 M1[i] = p1[i];
	 M2[i] = p2[i];
 }
	float FFTresult;
	for(int i=0;i<20;i++)
	{
		FFTresult = M1[i] + M2[i];

	}
	FFTresult = sqrt(FFTresult);
	return FFTresult;
}

int &  HU::ExcuteHog(Mat img, void * param, vector<float>&descriptors)
{
	 Mat trainImg;	 
	 cvtColor(img,trainImg,CV_BGR2GRAY);

      HOGDescriptor *hog = new HOGDescriptor(Size(3,3),Size(3,3),Size(5,10),Size(3,3),9); 
	  hog->gammaCorrection = true;//gamma校正
	 
  hog->compute(trainImg,descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算
  int desc_length = descriptors.size();
 
	  memcpy(param,&descriptors,sizeof(descriptors));

	int length = sizeof(descriptors);
	 delete hog;
	hog = NULL;
	return length;

	
}
void HU::ExcuteHog2(Mat img, vector<float>&descriptors)
{
	 Mat trainImg; 
	 cvtColor(img,trainImg,CV_BGR2GRAY);
        HOGDescriptor *hog = new HOGDescriptor(Size(3,3),Size(3,3),Size(5,10),Size(3,3),9); 
	  hog->gammaCorrection = true;//gamma校正 
         hog->compute(trainImg,descriptors,Size(1,1), Size(0,0)); //调用计算函数开始计算
 
      delete hog;
	hog = NULL;
	
}
float HU::CalDiffHog(void *data1, void *data2)
{

  vector< float >*p1 = NULL ;
    vector< float > * p2 = NULL ;
	
	double Mutisum = 0,Muti_x = 0,Muti_y = 0;
	
   p1 =  static_cast<vector< float >*>(data1);
   p2 = static_cast<vector< float >*>(data2);
   
  
   for(int i =0;i<p1->size();i++)
   {
	 
	   Mutisum += (*p1)[i]*(*p2)[i];
	   Muti_x  += (*p1)[i]*(*p1)[i];
	   Muti_y  += (*p2)[i]*(*p2)[i];
   }
   double distance = Mutisum/(sqrt(Muti_x)*sqrt(Muti_y));
	int length =0;
	return distance;
}
void HU::getFiles( string path, string exd, vector<string>& files )
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}
	
	if((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
			}
			else
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

