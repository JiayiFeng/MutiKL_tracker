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
	// _width��_height����32
	int numrects;
	static int ss=0;
	// numrect��haar�����о��ο��������2-6�������
	numrects = randint(2,6);
	Haar feat;
	feat._rects.resize(numrects);
	feat._weights.resize(numrects);
	feat._rsums.resize(numrects);
	feat._maxSum = 0.0f;
	feat._numrect = numrects;
	for( int k=0; k<numrects; k++ )
	{
		// ��ȫ������ķ�ʽȷ��haar�����п��λ�á���С�Լ�Ȩ��
		// ���еĿ򶼱��޶���LBP������32*32�ķ�Χ��
		feat._weights[k] = 2*randfloat() - 1;
		feat._rects[k].x = randint(0,(uint)(_width-4));
		feat._rects[k].y = randint(0,(uint)(_height-4));
		feat._rects[k].width = randint(1,(_width-feat._rects[k].x-2));
		feat._rects[k].height = randint(1 ,(_height-feat._rects[k].y-2));
		// ���Ȩ�س��Կ������ٳ���255�������ʲô������
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
	// sampleImage���������ڵ�ǰ֡�µļ������Χѡȡ����������ѵ�����������͸�����
	// im:��ǰ֡    rect1:�������
	// inrad:ѡȡ�����������ƫ�������½�    outrad:ѡȡ�����������ƫ�������Ͻ�
	// ��������ָ����һ�����ͼ����С��ͬ�Ŀ�
	int rowsz = im.rows - rect1.height - 1;
	int colsz = im.cols - rect1.width - 1;
	float inradsq = inrad*inrad;
	float outradsq = outrad*outrad;
	int dist;
	// ������ĸ����������޶��˿��Խ��в����ķ�Χ��������������Ͻǵ��ʾ
	uint minrow = max(0,(int)rect1.y-(int)inrad);
	uint maxrow = min((int)rowsz-1,(int)rect1.y+(int)inrad);
	uint mincol = max(0,(int)rect1.x-(int)inrad);
	uint maxcol = min((int)colsz-1,(int)rect1.x+(int)inrad);
	// samples�������ļ���
	vector<Rect> samples( (maxrow-minrow+1)*(maxcol-mincol+1) );
	int i=0;

	// prob����������������sample��Ƶ�ʣ��Ӷ�����sample�Ĵ�С
	/* ���sample��һ��ʼ��ʼ��sizeԽ��Ҳ����˵���ܿ��Լ���sample������Խ�࣬prob�ͻ�ԽС��
	   ������ж���randfloat() < prob�ĸ���Ҳ��Խ�ͣ����������ͱȽ��Ѽ���sample 
	*/
	float prob = ((float)(maxnum))/samples.size();
	for( int r=minrow; r<=(int)maxrow; r++ )
	{
		for( int c=mincol; c<=(int)maxcol; c++ )
		{
			// �������������Ƿ���Լ���sample���ͥ������������
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
	// ����sample�Ĵ�С���Ѻ���û���õ���λ�������
	samples.resize(min(i,maxnum));
	return samples;
}
// ==================================== ��δ������ =================================================== //
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
	// motrect:�洢��ѡ�������    scale��������ѡ��ĳ߶�    im:��һ֡��ԭʼͼ��    rect1:��һ֡�ļ������    inrad:��ѡ����rect1������    outrad����ѡ����rect1��С����
	static int numFrame = 0;
	int num = motrect.size();
	for (int i=0; i<num; i++)
	{
		// fcur������¼��һ֡�ļ���ͳ�ʼ��Ĵ�С����
		// fx������ɣ�������ʾ�������һ֡�ĳ߶ȱ任
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
				// ���fcur��С����fxֻ�����ɴ���1����
				fx = (randfloat())*0.1+1.0;//randint(0,1)*0.1+1;
			}
			else if (fcur>1.5)
			{
				// ���fcur�ܴ���fxֻ������С��1����
				fx = (randfloat()-1.0)*0.1+1.0;//randint(-1,0)*0.1+1;
			}
			else
			{
				fx = (randfloat()-0.5)*0.2+1.0;//randint(-1,1)*0.1+1;
			}
			
		}
		// ��ѡ���λ�úͳ߶ȴ�С����������ɵģ�ÿһ����ѡ���в�ͬ��λ�úͳ߶�
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
	// im:��ǰ֡ͼ��    feat:100��������ɵ�haar����    vectrect:��Ҫ����Ҷ������������ļ���  feaval:���ڴ��������������ֵ������
	// feat�ƺ�δ��ʹ��
	// patch��С����32*32����������д�Ĳ�̫һ��
	Size patchsize(patchwidth,patchheight);
	// num:��Ҫ�������������
	int num = vecrect.size();
	// �Ȱ�feaval��������Ҫ�Ĵ�С�����������е������ӿ����Ч��
	feaval.resize(num);
	long t1 = GetTickCount();
	#pragma omp parallel for
	for (int i=0; i<num; i++)
	{
		Mat im2(im,vecrect[i]);
		Mat imr;
		// imr��һ����С�Ѿ�����һ����patchsize��32*32��������
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
				// ��ν�ĻҶ��������Ǵ�С��һ�����������ÿ�����ĸ����ص�ĻҶ�֮��������ɵ�����
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
	// ���������ǵ�HOG����
	// im:��ǰ֡ͼ��    feat:100��������ɵ�haar����    vectrect:��Ҫ����Ҷ������������ļ���  feaval:���ڴ��������������ֵ������
	// feat�ƺ���Ȼδ��ʹ��
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
	// im����ǰ�ı����֡    rect1������    feat��������100��������ɵ�haar������vector
	static int numFrame = 0;
	int cellsize = 16;
	// pos��������neg������
	vector<Rect> pos, neg;
	// patchsize==32*32
	Size patchsize(patchwidth,patchheight);
	// ������������
	pos = sampleImage(im, rect1, 2, 0, 10000);
	neg = sampleImage(im, rect1, 37.5, 16, 80);
	// feapos������������feaneg����������
	// �����vector����ͬ������
	// �ڶ���vector��ĳһ�����µĸ�������
	// ���ڲ�vectorf��һ������ĳһ�����ľ���ֵ
	vector<vector<vectorf>> feapos(numtype), feaneg(numtype);
	// ����Ҷ�������ÿ�����ĸ����صĻҶ�ֵ֮�ͣ�
	PixFeaCal(im,feat,pos,feapos[0]);
	PixFeaCal(im,feat,neg,feaneg[0]);
	// ����Hog����
	HogFeaCal(im,feat,pos,feapos[1]);
	HogFeaCal(im,feat,neg,feaneg[1]);
	//LBPFeaCal(im,feat,pos,feapos[1]);
	//LBPFeaCal(im,feat,neg,feaneg[1]);
	if (curscore > 0.0)
	{
		long t1 = GetTickCount();
		// ��������õ�������������������������������ֵѵ��SVM������
		// ѵ�����������sv_coef��supfea��
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
	// im���ոն������һ֡ͼ��    rect1����һ֡ͼ���еļ���    feat��100�����������haar����    w������Ԫ�ض���0��vector
	int nummot = 400;
	// ��ѡ�ļ�������Ϊ400���������mot�У�����Щ��ѡ�Ľ����ѡ����õ�
	vector<Rect> mot(nummot);
	vectorf scale(nummot);
	// ���������400����ѡ��ÿ����ѡ���λ�úʹ�С����������ɵ�
	sampleMotionImage(mot, scale, im, rect1, 25, 0);
	// feamot��¼ÿ����ѡ�������
	vector<vector<vectorf>> feamot(numtype);
	// int numfeat = w.size();
	PixFeaCal(im,feat,mot,feamot[0]);
	HogFeaCal(im,feat,mot,feamot[1]);
	//LBPFeaCal(im,feat,mot,feamot[1]);
	long t1 = GetTickCount();
	vectorf score = svmpredict(feamot,scale);
	long t2 = GetTickCount();
	// �ҳ������ŵ�λ��
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