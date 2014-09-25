#include "svm.h"
#include <gurobi_c++.h>
#include <queue>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//void parse_command_line(vector<vector<vectorf>> &kermat, const vectorf &gamma);
//float verdot(const vectorf &v1, const vectorf &v2);
//vector<vectord> Ker(const vector<vectorf> &x1, const vector<vectorf> &x2, float gamma);
//void svmtrain(vector<vector<vectorf>> &pos, vector<vector<vectorf>> &neg, const int &numpos, const int & numneg);
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
vector<vector<vectorf>> supfea(numtype);
// trares记录了跟踪历史上的“优秀的有代表性的”跟踪结果，这些结果被看做正样本加入SVM的训练
// trares由trackers中的样本隔若干帧抽样，再经过svc()的聚类筛选而得到
vector<vector<vectorf>> trares(numtype);
// trackers代表追踪结果，储存了每一帧追踪结果框的特征
// 第一层vector特征，第二层vector样本（帧），第三层vectorf具体值
vector<vector<vectorf>> trackres(numtype);
vector<vectorf> sv_coef(numtype);
void parse_command_line(vector<vector<vectord>> &kermat, const vectorf &gamma)
{
	// 程序配置信息，已被配置文件替代，不会被调用
	param.svm_type = C_SVC;
	param.solver_type = SMO;
	param.d_regularizer = ENT; 
	param.d_proj = SIMPLEX; 
	param.num_kernels = 1;
	param.L_p = 2.0;
	param.kernels = NULL;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 0.5;
	param.lambda = 0.1;
	param.obj_threshold = 0.1;
	param.diff_threshold = 1e-3;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.kernels=new kernel[param.num_kernels];
	float uniform_weight = 1.0f/param.num_kernels;
	for(int i=0;i<param.num_kernels;i++)
	{
		param.kernels[i].coef = uniform_weight;
		param.kernels[i].kernel_type = PRECOMPUTED;
		param.kernels[i].degree = 3;
		param.kernels[i].gamma = gamma[i];
		param.kernels[i].coef0 = 0;
		int num = kermat[i][0].size();
		param.kernels[i].precomputed = new double *[num];
		for (int j=0; j<num; j++)
		{
			param.kernels[i].precomputed[j] = &kermat[i][j][0];
		}
		
		param.kernels[i].precomputed_filename = NULL;
	}
}

float verdot(const vectorf &v1, const vectorf &v2)
{
	// 线性Kernel，也就是直接在低维上点积
	int dims = v1.size();
	float sum = 0;
	#pragma omp parallel for
	for (int i=0; i<dims; i++)
	{
		sum += v1[i]*v2[i];
	}
	return sum;
}
float rbfdot(const vectorf &v1, const vectorf &v2)
{
	// RBF kernel的实现
	int dims = v1.size();
	float sum = 0;
	const float* pv1 = &v1[0];
	const float* pv2 = &v2[0];
	#pragma omp parallel for
	for (int i=0; i<dims; i++)
	{
		sum += pow(*(pv1)-*(pv2),2);
		pv1++;
		pv2++;
	}
	return sum;
}
vector<vectorf> Ker(const vector<vectorf> &x1, const vector<vectorf> &x2, const float gamma)
{
	int num1 = x1.size();
	int num2 = x2.size();
	vector<vectorf> Q(num1,vectorf(num2));
	#pragma omp parallel for
	for (int i=0; i<num1; i++)
	{
		// 自身与自身是0.5
		Q[i][i] = 0.5;
		#pragma omp parallel for
		for (int j=i+1; j<num2; j++)
		{
			float x = rbfdot(x1[i],x2[j]);
			float y = exp(-x*gamma);
			Q[i][j] = y;
			//Q[j][i] = y;
		}
	}
	return Q;
}
vector<vectorf> Ker2(const vector<vectorf> &x1, const vector<vectorf> &x2, const float gamma)
{
	// 这个版本被svmpredict调用
	int num1 = x1.size();
	int num2 = x2.size();
	vector<vectorf> Q(num1,vectorf(num2));
	#pragma omp parallel for
	for (int i=0; i<num1; i++)
	{
		#pragma omp parallel for
		for (int j=0; j<num2; j++)
		{
			float x = rbfdot(x1[i],x2[j]);
			Q[i][j] = exp(-x*gamma);
		}
	}
	return Q;
}

vector<vectorf> Ker2(const vectorf &x1, const vector<vectorf> &x2, const float gamma)
{
	// 这个版本被svc_predict调用
	int num1 = 1;
	int num2 = x2.size();
	vector<vectorf> Q(num1,vectorf(num2));
	#pragma omp parallel for
	for (int i=0; i<num1; i++)
	{
		#pragma omp parallel for
		for (int j=0; j<num2; j++)
		{
			float x = rbfdot(x1,x2[j]);
			Q[i][j] = exp(-x*gamma);
		}
	}
	return Q;
}

float svc_predict(const vectorf & gamma, const vector<vectorf> &csv_coef, vector<vector<vectorf>>& csupfea, const vector<vectorf>&  fea)
{
	float d = 0.0f;
	for (int i= 0; i<numtype; i++)
	{
		vector<vectorf> Q = Ker2(fea[i],csupfea[i],gamma[i]);
		for (int j=0; j<csv_coef[i].size(); j++)
		{
			d += csv_coef[i][j]*Q[0][j];
		}
	}
	return d;
}

vector<vector<bool>> FindAdjMatrix(const vectorf &gamma, double rho, const vector<vectorf> &csv_coef, vector<vector<vectorf>>& csupfea)
{
	bool adj_flag;
	vector<vectorf> diff(numtype), z(numtype);
	vectori dim(numtype);
	for (int i=0; i<numtype; i++)
	{
		dim[i] = csupfea[i][0].size();
	}
	
	for (int i=0; i<numtype; i++)
	{
		diff[i].resize(dim[i]);
		z[i].resize(dim[i]);
	}
	int ls = trackres[0].size();
	vector<vector<bool>> adjMatrix(ls);
	for (int i=0; i<ls; i++)
	{
		adjMatrix[i].resize(ls);
	}

	for(int i = 0;i < ls;i++)
	{
		for(int j = 0;j < ls;j++)
		{
			if(j < i)
			{
				if(adjMatrix[i][j])
				{
					for(int k = j+1;k < ls;k++)
					{
						adjMatrix[i][k] = (adjMatrix[i][k] || adjMatrix[j][k]);
					}
				}
			}
			else
			{
				if((i<j)&&(!adjMatrix[i][j]))
				{
					adj_flag = true; //unless a point on the path exits the shpere - the points are adj
					for (int k=0; k<numtype; k++)
					{
						for (int m=0; m<dim[k]; m++)
						{
							diff[k][m] = trackres[k][j][m] - trackres[k][i][m];
						}
					}
					
					for(double interval = 0.1;interval <1.05;interval+=0.1)
					{
						for (int k=0; k<numtype; k++)
						{
							for (int m=0; m<dim[k]; m++)
							{
								z[k][m] = trackres[k][i][m] + interval * (trackres[k][i][m] - trackres[k][j][m]);
							}
						}
						float d = svc_predict(gamma, csv_coef, csupfea, z)/numtype;
						//cout<<d<<' ';
						if(d > rho*0.975)
						{
							adj_flag = false;
							break;
						}
					}
					if(adj_flag)
					{
						adjMatrix[i][j] = true;
						adjMatrix[j][i] = true;
					}
				}
				else
				{
					adjMatrix[i][j] = true;
				}
			}
		}
	}
	return adjMatrix;
}

vector<vector<bool>> FindAdjMatrixPro(const vectorf &gamma, double rho, const vector<vectorf> &csv_coef, vector<vector<vectorf>>& csupfea, const vector<vectori>& ind)
{
	bool adj_flag;
	vector<vectorf> diff(numtype), z(numtype);
	vectori dim(numtype);
	for (int i=0; i<numtype; i++)
	{
		dim[i] = csupfea[i][0].size();
	}

	for (int i=0; i<numtype; i++)
	{
		diff[i].resize(dim[i]);
		z[i].resize(dim[i]);
	}
	int ls = trackres[0].size();
	vector<vector<bool>> adjMatrix(ls);
	for (int i=0; i<ls; i++)
	{
		adjMatrix[i].resize(ls);
	}
	int num = ind[0].size();
	for(int i = 0; i < ls; i++)
	{
		adjMatrix[i][i] = true;
		for(int k = 0; k < num; k++)
		{
			int j = ind[i][k];
			if(!adjMatrix[i][j])
			{
				adj_flag = true; 
				for(double interval = 0.1;interval <1.05;interval+=0.1)
				{
					for (int k=0; k<numtype; k++)
					{
						for (int m=0; m<dim[k]; m++)
						{
							z[k][m] = trackres[k][i][m] + interval * (trackres[k][i][m] - trackres[k][j][m]);
						}
					}
					float d = svc_predict(gamma, csv_coef, csupfea, z)/numtype;
					//cout<<d<<' ';
					if(d > rho)
					{
						adj_flag = false;
						break;
					}
				}
				if(adj_flag)
				{
					adjMatrix[i][j] = true;
					adjMatrix[j][i] = true;
				}
			}
		}
	}
	return adjMatrix;
}

vector<int> FindConnComp(const vector<vector<bool >> &adjMatrix)
{
	int ls = adjMatrix[0].size();
	vector<int> clustAssign(ls);
    vector<bool> visited(ls);
    for(int i = 0;i < ls;i++)
    {
		visited[i] = false;
    }
                                                                                                                        
    queue<int> Q;
    int currNode, currClust = 1;
    for(int i = 0;i < ls;i++)
    {
        if(!visited[i])
        {
            visited[i] = true;
            Q.push(i);
            while(!Q.empty())
            {
                currNode = Q.front();
                Q.pop();
                for(int j = 0;j < ls;j++)
                {
                        if((adjMatrix[currNode][j]) && (!visited[j]))
                        {
                                visited[j] = true;
                                Q.push(j);
                        }
                }
                clustAssign[currNode] = currClust;
            }
            currClust++;
        }
    }
	return clustAssign;
}

vector<vectori> knn(int num,const vector<vector<vectorf>>& kermat)
{
	int numtracked = trackres[0].size();
	vector<vectorf> Q(numtracked, vectorf(numtracked));
	vector<vectori> ind(numtracked,vectori(num));
	for (int i=0; i<numtracked; i++)
	{
		for (int j=i; j<numtracked; j++)
		{
			float s = 0.0f;
			for (int k=0; k<numtype; k++)
			{
				s += 1.0f-kermat[k][i][j];
			}
			Q[i][j] = s;
			Q[j][i] = s;
		}
	}
	for (int i=0; i<numtracked; i++)
	{
		vectori order;
		sort_order_des(Q[i],order);
		for (int k=0; k<num; k++ )
		{
			ind[i][k] = order[k];
		}
	}
	return ind;
}

vectori FindConnect(const vector<vector<bool >> &adjMatrix)
{
	int N = adjMatrix[0].size();
	vectori clustAss(N,0);
	int cluster_index = 0;
	bool done = false;

	while (done!=true)
	{
		int root = 0;
		while (clustAss[root] != 0)
		{
			root++;
			if (root>N-1)
			{
				done = true;
				break;
			}
			if (done)
			{
				break;
			}
		}
		if (done!=true)
		{
			cluster_index++;
			vectori sta(N,0);
			int sta_ind = -1;
			sta_ind++;
			sta[sta_ind] = root;
			
			while (sta_ind != -1)
			{
				int nod = sta[sta_ind];
				sta_ind--;
				clustAss[nod] = cluster_index;
				for (int i=0; i<N; i++)
				{
					if (adjMatrix[nod][i]&&clustAss[i]==0&&i!=nod)
					{
						sta_ind++;
						sta[sta_ind] = i;
					}
				}
			}
			
		}
	}
	return clustAss;	
	
}

void svc()
{
	int numkernel = numtype;
	int numofgroup = 5;
	int dim = trackres[0].size();
	vector<vector<vectorf>> kermat(numkernel);
	vectorf gamma(numkernel);
	gamma[0] = 0.05; gamma[1] = 0.10;
	for (int i=0; i<numkernel; i++)
	{
		kermat[i] = Ker(trackres[i],trackres[i],gamma[i]);
	}
	const float C = 0.05;
	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);
	//model.getEnv().set(GRB_IntParam_OutputFlag,0);
	vector<vector<GRBVar> > alpha(numkernel, vector<GRBVar >(dim));
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<dim; j++)
		{
			alpha[i][j] = model.addVar(0,C,0.0, GRB_CONTINUOUS);
		}		
	}
	model.update();
	GRBQuadExpr obj = 0;
	for (int k=0; k<numkernel; k++)
	{
		for (int i=0; i<dim; i++)
		{
			for (int j=i; j<dim; j++)
			{
				obj += 2*alpha[k][i]*kermat[k][i][j]*alpha[k][j];
			}
		}
	}
	model.setObjective(obj);
	GRBLinExpr c0 = 0;
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<dim; j++)
		{
			c0 += alpha[i][j];
		}
	}	
	model.addConstr(c0 == 1, "c0");
	model.optimize();
	int numsv = 0;

	vector<vectori> sel(numkernel);
	vector<vectorf> csv_coef(numkernel);
	vector<vector<vectorf>> csupfea(numkernel);
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<dim; j++)
		{
			float val = alpha[i][j].get(GRB_DoubleAttr_X);
			if (val>1e-6)
			{
				numsv++;
				sel[i].push_back(j);
				csv_coef[i].push_back(val);
				csupfea[i].push_back(trackres[i][j]);
			}
		}	
	}
	vectorf b(numsv,0);
	int num = 0;
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<sel[i].size(); j++)
		{
			for (int k=0; k<sel[i].size(); k++)
			{
				if (j==k)
				{
					b[num] += 2*csv_coef[i][k]*kermat[i][sel[i][j]][sel[i][k]];  
				}
				else if (j<k)
				{
					b[num] += csv_coef[i][k]*kermat[i][sel[i][j]][sel[i][k]]; 
				}
				else
				{
					b[num] += csv_coef[i][k]*kermat[i][sel[i][k]][sel[i][j]]; 
				}
			}
			num++;
		}
	}
	float rho = mean(b);
	vector<vectori> ind = knn(8, kermat);
	//vector<vector<bool>> adjMatrix = FindAdjMatrix(gamma,rho,csv_coef,csupfea);
	vector<vector<bool>> adjMatrix = FindAdjMatrixPro(gamma,rho,csv_coef,csupfea,ind);
	vector<int> clusterAssin = FindConnComp(adjMatrix);
	int numtracked = trackres[0].size();
	vectori sel1(numtracked,0);
	int idx = max_idx(clusterAssin);
	int numcluster = clusterAssin[idx];
	vector<vector<int> > indx(numcluster);
	cout<<"The maximum cluster center is "<<numcluster<<endl;
	for (int i=0; i<numcluster; i++)
	{
		for (int j=0; j<numtracked; j++)
		{
			if (clusterAssin[j] == i+1)
			{
				indx[i].push_back(j);
			}
		}
		
	}
	
	for (int i=0; i<numcluster; i++)
	{
		if (indx[i].size()>1)
		{
			int pernum = indx[i].size();
			int selnum = ceil(float(50*pernum/numtracked));
			vectori order = randperm(0,pernum);
			for (int j=0; j<selnum; j++)
			{
				sel1[indx[i][order[j]]] = 1;
			}
		}
		
	}
	ofstream oL;
	oL.open("L.txt");
	for (int i=0; i<sel1.size(); i++)
	{
		oL<<sel1[i]<<' ';
	}
	oL.close();
	ofstream oI;
	oI.open("I.txt");
	for (int i=0; i<numtracked; i++)
	{
		for (int j=0; j<256; j++)
		{
			oI<<trackres[0][i][j]<<' ';
		}
		oI<<endl;
	}
	oI.close();
	for (int i=0; i<numtype; i++)
	{
		trares[i].erase(trares[i].begin()+1,trares[i].end());
	}
	for (int i=0; i<numtracked; i++)
	{
		if (sel1[i] == 1)
		{
			for (int j=0; j<numtype; j++)
			{
				trares[j].push_back(trackres[j][i]);
			}

		}
	}
	for (int i=0; i<numtype; i++)
	{
		trackres[i].clear();
	}
	for (int i=0; i<numtype; i++)
	{
		trackres[i].assign(trares[i].begin(),trares[i].end());
	}

}

vector<int> svmtrain(vector<vector<vectorf>> &pos, vector<vector<vectorf>> &neg, const int &numpos, const int & numneg)
{
	static int numFrame = 0;
	if (numFrame==0)
	{
		// 对于第一帧来说，从正样本中选择中间的那个作为trackers的记录
		for (int i=0; i<numtype; i++)
		{
			trackres[i].push_back(pos[i][numpos/2]);
		}
	}
	//prob = Malloc(svm_problem, 1);
	// 如果记录的跟踪结果数量已经到达100，则启动增强核聚类算法（5.3.2内容）
	if (trackres[0].size()==100)
	{
		svc();
	}
	//prob = Malloc(svm_problem, 1);
	// 我trares是从trackers里面选出来的具有代表性的跟踪结果，选取过程中调用了svc()
	int numtracked = trares[0].size();
	// dim表示目前所有参与训练的样本数量，是传进来的正负样本数量之和加上trares里的样本数量
	int dim = numpos + numneg + numtracked;
	cout<<'\n'<<"numpos "<<numpos<<" numtracker "<<numtracked<<" numneg "<<numneg<<endl;
	cout<<dim<<endl;
	int numkernel = numtype;
	vectorf gamma(numkernel);
	//gamma[0] = 0.085; gamma[1] = 0.15; gamma[2] = 0.3;
	gamma[0] = 0.05; gamma[1] = 0.1; //gamma[2] = 0.3;
	// label是训练的标签，正样本（包括了以前的追踪结果）为1，负样本为-1
	vector<vector<int>> label(numkernel,vector<int>(dim));
	// 正样本被放在前面，因此前面的都标为1，负样本放在后面，标为-1
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<dim; j++)
		{
			if (j<numpos+numtracked)
			{
				label[i][j] = 1;
			}
			else
			{
				label[i][j] = -1;
			}
			
		}
	}
	// fea就是训练的样本了！其实就是把传进来的正负样本都放到一个vector里去，命名为fea
	vector<vector<vectorf>> fea;
	fea.assign(pos.begin(),pos.end());
	for (int i = 0; i < fea.size(); i++)
	{
		// 把以前的有代表性的追踪结果加进去
		fea[i].insert(fea[i].end(),trares[i].begin(),trares[i].end());
	}
	for (int i = 0; i < fea.size(); i++)
	{
		// 把负样本合进去
		fea[i].insert(fea[i].end(),neg[i].begin(),neg[i].end());
	}

	// Kermat为各个样本之间在经过核函数映射的高位空间上的点积结果
	// 第一层vector用来区分不同特征的结果，第二层vector开始可以看做一个矩阵
	vector<vector<vectorf>> kermat(numkernel);
	long t1 = GetTickCount();
	#pragma omp parallel for
	// 好像两种特征用的都是RBF核？
	for (int i=0; i<numkernel; i++)
	{
		kermat[i] = Ker(fea[i],fea[i],gamma[i]);
	}
	long t2 = GetTickCount();
	//cout<<t2-t1<<endl;
	const float C = 0.5;
	GRBEnv env = GRBEnv();
	GRBModel model = GRBModel(env);
	model.getEnv().set(GRB_IntParam_OutputFlag,0);
	// alpha就是优化的变量，也是SVM的核心所在！
	vector<vector<GRBVar>> alpha(numkernel, vector<GRBVar>(dim));
	//#pragma omp parallel for
	for (int i=0; i<numkernel; i++)
	{
		//#pragma omp parallel for
		for (int j=0; j<dim; j++)
		{
			alpha[i][j] = model.addVar(0,C,0.0, GRB_CONTINUOUS);
		}		
	}
	model.update();
	// obj是优化的目标函数！参见论文的公式5.8
	GRBQuadExpr obj = 0;
	//#pragma omp parallel for
	for (int k=0; k<numkernel; k++)
	{
		//#pragma omp parallel for
		for (int i=0; i<dim; i++)
		{
			//#pragma omp parallel for
			for (int j=i; j<dim; j++)
			{
				obj += label[k][i]*label[k][j]*alpha[k][i]*kermat[k][i][j]*alpha[k][j];
			}
		}
	}
	//#pragma omp parallel for
	for (int i=0; i<numkernel; i++)
	{
		//#pragma omp parallel for
		for (int j=0; j<dim; j++)
		{
			obj -= alpha[i][j]; 
		}
	}
	model.setObjective(obj);
	// 定义约束条件
	GRBLinExpr c0 = 0;
	//#pragma omp parallel for
	for (int i=0; i<numkernel; i++)
	{
		//#pragma omp parallel for
		for (int j=0; j<dim; j++)
		{
			c0 += alpha[i][j]*float(label[i][j]);
		}
	}	
	model.addConstr(c0 == 0, "c0");
	// 进行优化！！
	model.optimize();
	//model.write("abc.mps");
	int numsv = 0;
	
	ofstream oL;
	oL.open("sv.txt");
	ofstream oI;
	oI.open("supfea.txt");

	// 读出优化结果alpha中的数据，输出到文件sv.txt，同时将样本的灰度特征输出到supfea.txt
	// sv表示支持向量，support vector
	vector<int> sel(numkernel*dim,0);
	for (int i=0; i<numkernel; i++)
	{
		for (int j=0; j<dim; j++)
		{
			float val = alpha[i][j].get(GRB_DoubleAttr_X);
			oL<<val<<' ';
			if (i==0)
			{
				for (int jj=0; jj<256; jj++)
				{
					oI<<fea[0][j][jj]<<' ';
				}
				oI<<endl;
			}
			if (val>1e-6)
			{
				// val>1e-6表示这个样本在这个特征上被选为支持向量
				numsv++;
				// sv_coef是支持向量的标签，supfea是支持向量具体的值（已经乘以系数val）
				sv_coef[i].push_back(val*label[i][j]);
				supfea[i].push_back(fea[i][j]);
			}
		}	
		oL<<endl;
	}
	oL.close();
	oI.close();
	if (numFrame==250)
	{
		vectori supind(0);
		for (int i=0; i<supind.size(); i++)
		{
			cout<<supind[i];
		}
	}
	vector<int> supind(numsv);
	// 30帧之前每个tracker都加入trares，30帧之后每4帧抽取以帧加入
	// trares的大小控制在75个以下
	// 目测trares还要经过svc()的进一步筛选才能成为优秀的历史跟踪数据加入到下一帧的训练样本中去
	if (numFrame % 4 == 0 || numFrame < 30)
	{
		if (numtracked>75)
		{
			for (int i=0; i<numtype; i++)
			{
				trares[i].erase(trares[i].begin()+1);
			}
			for (int i=0; i<numtype; i++)
			{
				trares[i].push_back(trackres[i].back());
			}
		}
		else
		{
			for (int i=0; i<numtype; i++)
			{
				trares[i].push_back(trackres[i].back());
			}
		}
	}
	numFrame++;
	return supind;
}

vectorf svmpredict(const vector<vector<vectorf>> &feaval, vectorf &scale)
{
	// feaval:各个候选框的特征    scale：各个候选框相对于上一帧检测框的尺度变化
	static int numFrame = 0;
	// numfea为候选样本数目
	int numfea = feaval[0].size();
	
	vectorf score(numfea,0);
	// numkernel表示采用了几种特征。也就是kernel的种类数
	int numkernel = numtype;
	vectorf gamma(numkernel);
	//gamma[0] = 0.085; gamma[1] = 0.15; gamma[2] = 0.3;
	// gamma表示各个特征的权重系数
	gamma[0] = 0.05; gamma[1] = 0.1; //gamma[2] = 0.3;
	for (int i=0; i<numkernel; i++)
	{
		// supfea中存的是上一轮训练出来的支持向量的值（已经乘以系数alpha）
		// Q表示各个候选样本与各个支持向量之间在高维空间下的点积结果
		vector<vectorf> Q = Ker2(feaval[i],supfea[i],gamma[i]);
		int numsup = supfea[i].size();
		#pragma omp parallel for
		for (int j=0; j<numfea; j++)
		{
			#pragma omp parallel for
			for (int k=0; k<numsup; k++)
			{
				// 将同一候选样本与不同支持向量的相关结果乘上标签加在一起，得到最终的得分
				score[j] += sv_coef[i][k]*Q[j][k];
			}
		}
	}
	#pragma omp parallel for
	for (int i=0; i<numfea; i++)
	{
		//考虑一下样本的scale变化程度，scale变化越厉害的，得分越要被拉
		if (score[i]>0)
		{
			score[i] *= exp(-0.95*abs((scale[i]-1)));
		}
		else
		{
			score[i] /= exp(-0.95*abs((scale[i]-1)));
		}
		
	}
	int bestind = max_idx(score);
	curscore = score[bestind];
	if (curscore>0.0 && (numFrame % 2 == 0)||(curscore>0.0&&numFrame<100))
	{
		for (int i=0; i<numtype; i++)
		{
			// tracker本身也是隔两帧记录一次
			trackres[i].push_back(feaval[i][bestind]);
		}
	}
	if (curscore>0.0)
	{
		for (int i=0; i<numtype; i++)
		{
			sv_coef[i].clear();
			supfea[i].clear();
		}
	}
	numFrame++;
	return score;	
}