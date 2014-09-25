#ifndef H_HAAR
#define H_HAAR

#include "Public.h"
#include "lbp.h"
#include "trainsvm.h"


class Haar
{
public:
	vector<float>			_weights;
	vector<Rect>			_rects;
	vector<float>			_rsums;
	double					_maxSum;
	int                     _numrect;
};
Haar HaarGenerate(int width, int height)
{
	int _width =  width;
	int _height =  height;
	// _width和_height都是32
	int numrects;
	static int ss=0;
	// numrect是haar特征中矩形框的数量，2-6随机产生
	numrects = randint(2,6);
	Haar feat;
	feat._rects.resize(numrects);
	feat._weights.resize(numrects);
	feat._rsums.resize(numrects);
	feat._maxSum = 0.0f;
	feat._numrect = numrects;
	for( int k=0; k<numrects; k++ )
	{
		// 完全用随机的方式确定haar特征中框的位置、大小以及权重
		// 所有的框都被限定在LBP特征的32*32的范围中
		feat._weights[k] = 2*randfloat() - 1;
		feat._rects[k].x = randint(0,(uint)(_width-4));
		feat._rects[k].y = randint(0,(uint)(_height-4));
		feat._rects[k].width = randint(1,(_width-feat._rects[k].x-2));
		feat._rects[k].height = randint(1 ,(_height-feat._rects[k].y-2));
		// 框的权重乘以框的面积再乘以255，这个是什么意义呢
		feat._rsums[k] = abs(feat._weights[k]*(feat._rects[k].width+1)*(feat._rects[k].height+1)*255);
	}
	return feat;
}
void HaarCal(const Mat &im, const vector<Haar> &feat, vectorf &ftval)
{
	Mat iim;
	integral(im,iim,-1);
	int numfeat = feat.size();
	#pragma omp parallel for
	for (int i=0; i<numfeat; i++)
	{
		Haar ft = feat[i];
		int numrect = ft._numrect;
		float sum = 0;
		#pragma omp parallel for
		for( int j=0; j<numrect; j++ )
		{
			int x = ft._rects[j].x;
			int y = ft._rects[j].y;
			int xw = x + ft._rects[j].width;
			int yh = y + ft._rects[j].height;
			sum += ft._weights[j]*(iim.at<int>(y,x)+iim.at<int>(yh,xw)-iim.at<int>(yh,x)-iim.at<int>(y,xw))/ft._rsums[j];
		}
		ftval[i] = sum;
	}
}
vector<Rect> sampleImage(const Mat &im, Rect rect1,float inrad, float outrad, int maxnum)
{
	// sampleImage函数用于在当前帧下的检测结果周围选取样本，构成训练的正样本和负样本
	// im:当前帧    rect1:检测结果框
	// inrad:选取的样本与检测框偏移量的下界    outrad:选取的样本与检测框偏移量的上界
	// “样本”指的是一个个和检测框大小相同的框
	int rowsz = im.rows - rect1.height - 1;
	int colsz = im.cols - rect1.width - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;
	// 下面的四个参数初步限定了可以进行采样的范围。以样本框的左上角点表示
	uint minrow = max(0,(int)rect1.y-(int)inrad);
	uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
	uint mincol = max(0,(int)rect1.x-(int)inrad);
	uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
	// samples：样本的集合
	vector<Rect> samples( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	// prob用来控制样本加入sample的频率，从而控制sample的大小
	/* 如果sample的一开始初始的size越大，也就是说可能可以加入sample的样本越多，prob就会越小，
	   后面的判断中randfloat() < prob的概率也会越低，于是样本就比较难加入sample 
	*/
	float prob = ((float)(maxnum))/samples.size();
	for( int r=minrow; r<=(int)maxrow; r++ )
	{
		for( int c=mincol; c<=(int)maxcol; c++ )
		{
			// 真正控制样本是否可以加入sample大家庭的条件在这里
			dist = (rect1.y-r)*(rect1.y-r) + (rect1.x-c)*(rect1.x-c);
			if( randfloat() < prob && dist < inradsq && dist >= outradsq )
			{
				Rect re;
				re.x = c;
				re.y = r;
				re.width = rect1.width;
				re.height = rect1.height;
				samples[i] = re;
				i++;
			}
		}
	}
	// 调整sample的大小，把后面没有用到的位置清理掉
	samples.resize(min(i,maxnum));
	return samples;
}
// ==================================== 从未被调用 =================================================== //
vector<Rect>	samplePosImage(const Mat &im, Rect rect1,float inrad, float outrad, int maxnum)
{
	int rowsz = im.rows - rect1.height - 1;
	int colsz = im.cols - rect1.width - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;
	uint minrow = max(0,(int)rect1.y-(int)inrad);
	uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
	uint mincol = max(0,(int)rect1.x-(int)inrad);
	uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
	vector<Rect> samples( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	float prob = ((float)(maxnum))/samples.size();
	for( int r=minrow; r<=(int)maxrow; r+=2 )
	{
		for( int c=mincol; c<=(int)maxcol; c+=2 )
		{
			dist = (rect1.y-r)*(rect1.y-r) + (rect1.x-c)*(rect1.x-c);
			if( randfloat()<prob && dist < inradsq && dist >= outradsq )
			{
				Rect re;
				re.x = c;
				re.y = r;
				re.width = rect1.width;
				re.height = rect1.height;
				samples[i] = re;
				i++;
			}
		}
	}
	samples.resize(min(i,maxnum));
	return samples;

}

vector<Rect>	sampleImage2(const Mat &im, Rect rect1,float inrad, float outrad, int maxnum, int step)
{
	int rowsz = im.rows - rect1.height - 1;
	int colsz = im.cols - rect1.width - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;
	uint minrow = max(0,(int)rect1.y-(int)inrad);
	uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
	uint mincol = max(0,(int)rect1.x-(int)inrad);
	uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
	vector<Rect> samples( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	float prob = ((float)(maxnum))/samples.size();
	for( int r=minrow; r<=(int)maxrow; r+=step )
	{
		for( int c=mincol; c<=(int)maxcol; c+=step )
		{
			dist = (rect1.y-r)*(rect1.y-r) + (rect1.x-c)*(rect1.x-c);
			if( randfloat()<prob && dist < inradsq && dist >= outradsq )
			{
				Rect re;
				re.x = c;
				re.y = r;
				re.width = rect1.width;
				re.height = rect1.height;
				samples[i] = re;
				i++;
			}
		}
	}
	samples.resize(min(i,maxnum));
	return samples;

}

void	sampleMotionImage2(vector<Rect> &motrect, vectorf &scale, const Mat &im, Rect rect1,float inrad, float outrad)
{
	static int numFrame = 0;
	int rowsz = im.rows - rect1.height - 1;
	int colsz = im.cols - rect1.width - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;
	uint minrow = max(0,(int)rect1.y-(int)inrad);
	uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
	uint mincol = max(0,(int)rect1.x-(int)inrad);
	uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
	vector<Rect> samples;
	for( int r=minrow; r<=(int)maxrow; r+=2 )
	{
		for( int c=mincol; c<=(int)maxcol; c+=2 )
		{
			float fx = 0, fcur=0;
			fcur = 0.5*(rect1.width/oriRect.width + rect1.height/oriRect.height);
			if (numFrame<30)
			{
				fx = 1.0;
			}
			else
			{
				if (fcur<0.6)
				{
					fx = (randfloat())*0.1+1.0;//randint(0,2)*0.05+1;
				}
				else if (fcur>1.5)
				{
					fx = (randfloat()-1.0)*0.1+1.0;//randint(-2,0)*0.05+1;
				}
				else
				{
					fx =(randfloat()-0.5)*0.2+1.0;// randint(-2,2)*0.05+1;
				}

			}
			float delwidth = abs(1-fx)*rectscale*oriRect.width/2.0f;
			float delheight = abs(1-fx)*rectscale*oriRect.height/2.0f; 
			if (fx>1)
			{
				Rect re(c-delwidth,r-delheight,rectscale*oriRect.width+2*delwidth,rectscale*oriRect.height+2*delheight);
				if (re.x<0||re.y<0||re.x+re.width>im.cols||re.y+re.height>im.rows)
				{
					;
				}
				else
				{
					motrect.push_back(re);
					scale.push_back(fx);
				}
			}
			else
			{
				Rect re(c+delwidth,r+delheight,rect1.width-2*delwidth,rect1.height-2*delheight);
				if (re.x<0||re.y<0||re.x+re.width>im.cols||re.y+re.height>im.rows)
				{
					;
				}
				else
				{
					motrect.push_back(re);
					scale.push_back(fx);
				}

			}
		}
	}
	numFrame++;
}
// ================================================================================================ //

void	sampleMotionImage(vector<Rect> &motrect, vectorf &scale, const Mat &im, Rect rect1,float inrad, float outrad)
{
	// motrect:存储候选框的容器    scale：各个候选框的尺度    im:下一帧的原始图像    rect1:上一帧的检测结果框    inrad:候选框与rect1最大距离    outrad：候选框与rect1最小距离
	static int numFrame = 0;
	int num = motrect.size();
	for (int i=0; i<num; i++)
	{
		// fcur用来记录上一帧的检测框和初始框的大小比例
		// fx随机生成，用来表示相对于上一帧的尺度变换
		float fx = 0, fcur=0;
		fcur = 0.5*(rect1.width/oriRect.width + rect1.height/oriRect.height);
		if (numFrame<30)
		{
			fx = 1.0;
		}
		else
		{
			if (fcur<0.6)
			{
				// 如果fcur很小，则fx只会生成大于1的数
				fx = (randfloat())*0.1+1.0;//randint(0,1)*0.1+1;
			}
			else if (fcur>1.5)
			{
				// 如果fcur很大，则fx只会生成小于1的数
				fx = (randfloat()-1.0)*0.1+1.0;//randint(-1,0)*0.1+1;
			}
			else
			{
				fx = (randfloat()-0.5)*0.2+1.0;//randint(-1,1)*0.1+1;
			}
			
		}
		// 候选框的位置和尺度大小都是随机生成的，每一个候选框都有不同的位置和尺度
		scale[i] = fx;
		Rect rect2(rect1);
		rect2.width = cvRound(fx*rectscale*rect1.width);
		rect2.height = cvRound(fx*rectscale*rect1.height); 
		int rowsz = im.rows - rect1.height - 1;
		int colsz = im.cols - rect1.width - 1;
		uint minrow = max(0,(int)rect1.y-(int)inrad);
		uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
		uint mincol = max(0,(int)rect1.x-(int)inrad);
		uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
		rect2.x = randint(mincol, maxcol);
		rect2.y = randint(minrow, maxrow);
		if (rect2.x<0||rect2.y<0||rect2.x+rect2.width>im.cols||rect2.y+rect2.height>im.rows)
		{
			i--;
		}
		else
		{
			motrect[i] = rect2;
		}
		
	
	}
	numFrame++;
}

void HaarFeaCal(const Mat &im, const vector<Haar> &feat, const vector<Rect> &vecrect, vector<vectorf> &feaval)
{
	Size patchsize(patchwidth,patchheight);
	int num = vecrect.size();
	feaval.resize(num);
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		resize(im2,imr,patchsize);
		feaval[i].resize(feat.size());
		HaarCal(imr,feat,feaval[i]);
	}
}
void PixFeaCal(const Mat &im, const vector<Haar> &feat, const vector<Rect> &vecrect, vector<vectorf> &feaval)
{
	// im:当前帧图像    feat:100种随机生成的haar特征    vectrect:需要计算灰度特征的样本的集合  feaval:用于储存各个样本特征值的容器
	// feat似乎未被使用
	// patch大小还是32*32，和文章里写的不太一样
	Size patchsize(patchwidth,patchheight);
	// num:需要处理的样本数量
	int num = vecrect.size();
	// 先把feaval调整到需要的大小，避免运算中调整，加快计算效率
	feaval.resize(num);
	long t1 = GetTickCount();
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		// imr是一个大小已经被归一化到patchsize（32*32）的样本
		resize(im2,imr,patchsize);
		feaval[i].resize(patchwidth/2*patchheight/2);
		int ind = 0;
		for (int j=0; j<patchheight/2; j++)
		{
			for (int k=0; k<patchwidth/2; k++)
			{
				float v1 = (float)imr.at<unsigned char>(2*j,2*k);
				float v2 = (float)imr.at<unsigned char>(2*j+1,2*k+1);
				float v3 = (float)imr.at<unsigned char>(2*j,2*k+1);
				float v4 = (float)imr.at<unsigned char>(2*j+1,2*k);
				// 所谓的灰度特征就是大小归一化后的样本上每相邻四个像素点的灰度之和排列组成的向量
				feaval[i][ind] = (v1+v2+v3+v4)/(255.0f*4.0f);
				ind++;
			}
		}
	}
	long t2 = GetTickCount();
	//cout<<t2-t1<<endl;
}
void LBPFeaCal(const Mat &im, const vector<Haar> &feat, const vector<Rect> &vecrect, vector<vectorf> &feaval)
{
	Size patchsize(patchwidth,patchheight);
	int num = vecrect.size();
	feaval.resize(num);
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		resize(im2,imr,patchsize);
		int cellsize = 16;
		feaval[i] = CalLBP(imr, cellsize);
	}
}
void HistFeaCal(const Mat &im, const vector<Haar> &feat, const vector<Rect> &vecrect, vector<vectorf> &feaval)
{
	Size patchsize(patchwidth,patchheight);
	int num = vecrect.size();
	int histSize = 64;
	float ranges[] = { 0, 255 };
	const float* histRange = { ranges };
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		resize(im2,imr,patchsize);
		Mat hist;
		calcHist( &imr, 1, 0, Mat(),hist, 1, &histSize, &histRange,true,false );
		normalize(hist, hist, 1, 0, NORM_L1);
		feaval[i].resize(hist.rows);
		for (int j=0; j<hist.rows; j++)
		{
			feaval[i][j] = hist.at<float>(j);
		}
	}
}
void HogFeaCal(const Mat &im, const vector<Haar> &feat, const vector<Rect> &vecrect, vector<vectorf> &feaval)
{
	// 计算样本们的HOG特征
	// im:当前帧图像    feat:100种随机生成的haar特征    vectrect:需要计算灰度特征的样本的集合  feaval:用于储存各个样本特征值的容器
	// feat似乎依然未被使用
	Size patchsize(patchwidth,patchheight);
	int num = vecrect.size();
	feaval.resize(num);
	HOGDescriptor hog( patchsize, Size(16,16), Size(8,8),
		Size(8,8), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, true, 
		HOGDescriptor::DEFAULT_NLEVELS);
	vector <Point> loc;
	loc.push_back(Point(0,0));
	long t1 = GetTickCount();
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		resize(im2,imr,patchsize);
		hog.compute(imr,feaval[i],hog.blockStride,patchsize,loc);
	}
	long t2 = GetTickCount();
	//cout<<t2-t1<<endl;
}
vectorf LearnApp(const Mat &im, const Rect &rect1, const vector<Haar> &feat)
{
	// im：当前的被检测帧    rect1：检测框    feat：包含了100种随机生成的haar特征的vector
	static int numFrame = 0;
	int cellsize = 16;
	// pos正样本，neg负样本
	vector<Rect> pos, neg;
	// patchsize==32*32
	Size patchsize(patchwidth,patchheight);
	// 产生正负样本
	pos = sampleImage(im, rect1, 2, 0, 10000);
	neg = sampleImage(im, rect1, 37.5, 16, 80);
	// feapos正样本特征，feaneg负样本特征
	// 最外层vector：不同的特征
	// 第二层vector：某一特征下的各个样本
	// 最内层vectorf：一个样本某一特征的具体值
	vector<vector<vectorf>> feapos(numtype), feaneg(numtype);
	// 计算灰度特征（每相邻四个像素的灰度值之和）
	PixFeaCal(im,feat,pos,feapos[0]);
	PixFeaCal(im,feat,neg,feaneg[0]);
	// 计算Hog特征
	HogFeaCal(im,feat,pos,feapos[1]);
	HogFeaCal(im,feat,neg,feaneg[1]);
	//LBPFeaCal(im,feat,pos,feapos[1]);
	//LBPFeaCal(im,feat,neg,feaneg[1]);
	if (curscore > 0.0)
	{
		long t1 = GetTickCount();
		// 利用上面得到的正负样本和正负样本的两种特征值训练SVM分类器
		// 训练结果被存入sv_coef和supfea中
		svmtrain(feapos,feaneg,pos.size(),neg.size());
		long t2 = GetTickCount();
		//cout<<t2-t1<<endl;
	}
	
	numFrame++;
	vectorf w(3);
	return w;
}
Rect Locate(const Mat &im, const Rect &rect1, const vector<Haar> &feat, const vectorf &w)
{
	// im：刚刚读入的下一帧图像    rect1：上一帧图像中的检测框    feat：100个随机产生的haar特征    w：三个元素都是0的vector
	int nummot = 400;
	// 候选的检测框数量为400个，存放在mot中，从这些候选的结果中选出最好的
	vector<Rect> mot(nummot);
	vectorf scale(nummot);
	// 随机生成这400个候选框，每个候选框的位置和大小都是随机生成的
	sampleMotionImage(mot, scale, im, rect1, 25, 0);
	// feamot记录每个候选框的特征
	vector<vector<vectorf>> feamot(numtype);
	// int numfeat = w.size();
	PixFeaCal(im,feat,mot,feamot[0]);
	HogFeaCal(im,feat,mot,feamot[1]);
	//LBPFeaCal(im,feat,mot,feamot[1]);
	long t1 = GetTickCount();
	vectorf score = svmpredict(feamot,scale);
	long t2 = GetTickCount();
	// 找出了最优的位置
	int bestIdx = max_idx(score);
	curscore = score[bestIdx];
	cout<<"The best score is "<<curscore<<endl;
	if (curscore>0.0)
	{
		Rect newrect = mot[bestIdx];
		rectscale *= scale[bestIdx];
		return newrect;
	}
	else
	{
		Rect newRect = mot[bestIdx];
		rectscale *= scale[bestIdx];
		return newRect;
	}
		
	
}
#endif 