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
// trares��¼�˸�����ʷ�ϵġ�������д����Եġ����ٽ������Щ�������������������SVM��ѵ��
// trares��trackers�е�����������֡�������پ���svc()�ľ���ɸѡ���õ�
vector<vector<vectorf>> trares(numtype);
// trackers����׷�ٽ����������ÿһ֡׷�ٽ���������
// ��һ��vector�������ڶ���vector������֡����������vectorf����ֵ
vector<vector<vectorf>> trackres(numtype);
vector<vectorf> sv_coef(numtype);
void parse_command_line(vector<vector<vectord>> &kermat, const vectorf &gamma)
{
	// ����������Ϣ���ѱ������ļ���������ᱻ����
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
	// ����Kernel��Ҳ����ֱ���ڵ�ά�ϵ��
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
	// RBF kernel��ʵ��
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
		// ������������0.5
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
	// ����汾��svmpredict����
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
	// ����汾��svc_predict����
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
		// ���ڵ�һ֡��˵������������ѡ���м���Ǹ���Ϊtrackers�ļ�¼
		for (int i=0; i<numtype; i++)
		{
			trackres[i].push_back(pos[i][numpos/2]);
		}
	}
	//prob = Malloc(svm_problem, 1);
	// �����¼�ĸ��ٽ�������Ѿ�����100����������ǿ�˾����㷨��5.3.2���ݣ�
	if (trackres[0].size()==100)
	{
		svc();
	}
	//prob = Malloc(svm_problem, 1);
	// ��trares�Ǵ�trackers����ѡ�����ľ��д����Եĸ��ٽ����ѡȡ�����е�����svc()
	int numtracked = trares[0].size();
	// dim��ʾĿǰ���в���ѵ���������������Ǵ�������������������֮�ͼ���trares�����������
	int dim = numpos + numneg + numtracked;
	cout<<'\n'<<"numpos "<<numpos<<" numtracker "<<numtracked<<" numneg "<<numneg<<endl;
	cout<<dim<<endl;
	int numkernel = numtype;
	vectorf gamma(numkernel);
	//gamma[0] = 0.085; gamma[1] = 0.15; gamma[2] = 0.3;
	gamma[0] = 0.05; gamma[1] = 0.1; //gamma[2] = 0.3;
	// label��ѵ���ı�ǩ������������������ǰ��׷�ٽ����Ϊ1��������Ϊ-1
	vector<vector<int>> label(numkernel,vector<int>(dim));
	// ������������ǰ�棬���ǰ��Ķ���Ϊ1�����������ں��棬��Ϊ-1
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
	// fea����ѵ���������ˣ���ʵ���ǰѴ������������������ŵ�һ��vector��ȥ������Ϊfea
	vector<vector<vectorf>> fea;
	fea.assign(pos.begin(),pos.end());
	for (int i = 0; i < fea.size(); i++)
	{
		// ����ǰ���д����Ե�׷�ٽ���ӽ�ȥ
		fea[i].insert(fea[i].end(),trares[i].begin(),trares[i].end());
	}
	for (int i = 0; i < fea.size(); i++)
	{
		// �Ѹ������Ͻ�ȥ
		fea[i].insert(fea[i].end(),neg[i].begin(),neg[i].end());
	}

	// KermatΪ��������֮���ھ����˺���ӳ��ĸ�λ�ռ��ϵĵ�����
	// ��һ��vector�������ֲ�ͬ�����Ľ�����ڶ���vector��ʼ���Կ���һ������
	vector<vector<vectorf>> kermat(numkernel);
	long t1 = GetTickCount();
	#pragma omp parallel for
	// �������������õĶ���RBF�ˣ�
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
	// alpha�����Ż��ı�����Ҳ��SVM�ĺ������ڣ�
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
	// obj���Ż���Ŀ�꺯�����μ����ĵĹ�ʽ5.8
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
	// ����Լ������
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
	// �����Ż�����
	model.optimize();
	//model.write("abc.mps");
	int numsv = 0;
	
	ofstream oL;
	oL.open("sv.txt");
	ofstream oI;
	oI.open("supfea.txt");

	// �����Ż����alpha�е����ݣ�������ļ�sv.txt��ͬʱ�������ĻҶ����������supfea.txt
	// sv��ʾ֧��������support vector
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
				// val>1e-6��ʾ�����������������ϱ�ѡΪ֧������
				numsv++;
				// sv_coef��֧�������ı�ǩ��supfea��֧�����������ֵ���Ѿ�����ϵ��val��
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
	// 30֮֡ǰÿ��tracker������trares��30֮֡��ÿ4֡��ȡ��֡����
	// trares�Ĵ�С������75������
	// Ŀ��trares��Ҫ����svc()�Ľ�һ��ɸѡ���ܳ�Ϊ�������ʷ�������ݼ��뵽��һ֡��ѵ��������ȥ
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
	// feaval:������ѡ�������    scale��������ѡ���������һ֡����ĳ߶ȱ仯
	static int numFrame = 0;
	// numfeaΪ��ѡ������Ŀ
	int numfea = feaval[0].size();
	
	vectorf score(numfea,0);
	// numkernel��ʾ�����˼���������Ҳ����kernel��������
	int numkernel = numtype;
	vectorf gamma(numkernel);
	//gamma[0] = 0.085; gamma[1] = 0.15; gamma[2] = 0.3;
	// gamma��ʾ����������Ȩ��ϵ��
	gamma[0] = 0.05; gamma[1] = 0.1; //gamma[2] = 0.3;
	for (int i=0; i<numkernel; i++)
	{
		// supfea�д������һ��ѵ��������֧��������ֵ���Ѿ�����ϵ��alpha��
		// Q��ʾ������ѡ���������֧������֮���ڸ�ά�ռ��µĵ�����
		vector<vectorf> Q = Ker2(feaval[i],supfea[i],gamma[i]);
		int numsup = supfea[i].size();
		#pragma omp parallel for
		for (int j=0; j<numfea; j++)
		{
			#pragma omp parallel for
			for (int k=0; k<numsup; k++)
			{
				// ��ͬһ��ѡ�����벻֧ͬ����������ؽ�����ϱ�ǩ����һ�𣬵õ����յĵ÷�
				score[j] += sv_coef[i][k]*Q[j][k];
			}
		}
	}
	#pragma omp parallel for
	for (int i=0; i<numfea; i++)
	{
		//����һ��������scale�仯�̶ȣ�scale�仯Խ�����ģ��÷�ԽҪ����
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
			// tracker����Ҳ�Ǹ���֡��¼һ��
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