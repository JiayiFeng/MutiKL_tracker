#include "Haar.h"
#include "Config.h"
#include "lbp.h"

#ifdef _DEBUG 
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_calib3d249d.lib")
#pragma comment(lib, "opencv_contrib249d.lib")
#pragma comment(lib, "opencv_features2d249d.lib")
#pragma comment(lib, "opencv_flann249d.lib")
#pragma comment(lib, "opencv_gpu249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#pragma comment(lib, "opencv_legacy249d.lib")
#pragma comment(lib, "opencv_ml249d.lib")
#pragma comment(lib, "opencv_objdetect249d.lib")
#pragma comment(lib, "opencv_ts249d.lib")
#pragma comment(lib, "opencv_video249d.lib")
#pragma comment(lib, "gurobi_c++mdd2012.lib")
#pragma comment(lib, "gurobi56.lib")
#else
#pragma comment(lib, "opencv_calib3d249.lib")
#pragma comment(lib, "opencv_contrib249.lib")
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_features2d249.lib")
#pragma comment(lib, "opencv_flann249.lib")
#pragma comment(lib, "opencv_gpu249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_legacy249.lib")
#pragma comment(lib, "opencv_ml249.lib")
#pragma comment(lib, "opencv_objdetect249.lib")
#pragma comment(lib, "opencv_ts249.lib")
#pragma comment(lib, "opencv_video249.lib")
#pragma comment(lib, "gurobi_c++md2012.lib")
#pragma comment(lib, "gurobi56.lib")
#endif



int main()
{
	// read config file
	string configPath = "config.txt";
	Config conf(configPath);
	cout << conf << endl;
	VideoCapture cap;
	vector<Haar> feat;
	feat.resize(100);
	// ��LBP�����Ļ������������100��haar����
	for (int i=0; i<100; i++)
	{
		// LBP������ÿ��ͼ��鱻��һ��Ϊ32*32����������patchwidth��patchheight����32
		feat[i] = HaarGenerate(patchwidth,patchheight);
	}
	int startFrame = -1;
	int endFrame = -1;
	string imgFormat;

	string framesFilePath = conf.sequenceBasePath+"\\"+conf.sequenceName+"\\"+conf.sequenceName+"_frames.txt";
	ifstream framesFile(framesFilePath.c_str(), ios::in);
	string framesLine;
	getline(framesFile, framesLine);
	sscanf(framesLine.c_str(), "%d,%d", &startFrame, &endFrame);
	imgFormat = conf.sequenceBasePath+"\\"+conf.sequenceName+"\\imgs\\%04d.jpg";
	string gtFilePath = conf.sequenceBasePath+"\\"+conf.sequenceName+"\\"+conf.sequenceName+"_gt.txt";
	ifstream gtFile(gtFilePath.c_str(), ios::in);
	string gtLine;
	getline(gtFile, gtLine);
	float xmin = -1.f;
	float ymin = -1.f;
	float width = -1.f;
	float height = -1.f;
	sscanf(gtLine.c_str(), "%f,%f,%f,%f", &xmin, &ymin, &width, &height);
	// initRect��ʼ����
	Rect initRect(xmin,ymin,width,height);
	oriRect.x = initRect.x;
	oriRect.y = initRect.y;
	oriRect.width = initRect.width;
	oriRect.height = initRect.height;
	// rect1��ʾ����
	Rect rect1(initRect);
	bool paused = false;
	bool doInitialise = false;
	vector<Rect> res;
	// res������¼��ʷ����
	res.push_back(rect1);
	
	for (int frameInd = startFrame; frameInd <= endFrame-1; ++frameInd)
	{
		long t1 = GetTickCount();
		//Mat frame;		
		char imgPath[256];
		sprintf(imgPath, imgFormat.c_str(), frameInd);
		cout<<imgPath;
		string path(imgPath);
		// frameOrig��ʾ��ǰ֡
		Mat frameOrig = imread(path, 0);
		vectorf w = LearnApp(frameOrig, rect1, feat);
		sprintf(imgPath, imgFormat.c_str(), frameInd+1); 
		string path2(imgPath);
		frameOrig = imread(path2, 0);
		rect1 = Locate(frameOrig,rect1,feat,w);
		namedWindow("im",1);
		cvtColor(frameOrig,frameOrig,CV_GRAY2BGR);
		putText(frameOrig,int2str(frameInd,4),Point(20,20),1,2,Scalar(0,255,0));
		rectangle(frameOrig,rect1,Scalar(0,255,0),2);
		imshow("im",frameOrig);
		waitKey(1);
		res.push_back(rect1);
		long t2 = GetTickCount();
		cout<<t2-t1<<endl;
	}
	string name = conf.sequenceName + int2str(conf.seed,2) + int2str(6,1) +".txt";
	ofstream outFile(name.c_str());
	for (int i=0; i<res.size(); i++)
	{
		outFile<<res[i].x<<' '<<res[i].y<<' '<<res[i].width<<' '<<res[i].height<<endl;
	}
	system("pause");
	return EXIT_SUCCESS;
}
