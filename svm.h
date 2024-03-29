#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 290

#ifdef __cplusplus
extern "C" {
#endif

extern int libsvm_version;

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

struct kernel
{
	double coef; /* initial coefficient of the kernel */
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */
	// precomputed stuff
	double **precomputed;
	int precomputed_numrows;
	int precomputed_numcols;
	char *precomputed_filename;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
enum { SMO, SMKL, MD };	/* solver_type */
enum { ENT, L1, L2, LPS };	/* d_regularizer */
enum { SIMPLEX, NN_ORTHANT };	/* d_proj */

struct svm_parameter
{
	int svm_type;
  int solver_type;   /* 1 for SMO based, 2 for reduced gradient (SimpleMKL), and 3 for Mirror Descent */
  int d_regularizer; /* 0 for entropy, 1 for L1, 2 for L2, and 3 for Lp */
  int d_proj;       /* 0 for simplex, 1 for non negative orthant */

	int num_kernels;/* number of kernels in convex combination */
	int l;          /* number of data points */
  float L_p;
  struct kernel *kernels;/* pointer to an array of the kernels */
  
	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	double lambda; /* tradeoff between regularizer and objective function */
	double obj_threshold; /* threshold of the increase in objective function in line search */
	double diff_threshold; /* threshold that affects when line search terminates */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
//
struct svm_model
{
	svm_parameter *param;	// array of parameter structures
	int nr_class;		// number of classes,=2 in regression/one class svm
	int l;			// total #SV
	svm_node **SV;		// SVs (SV[l])
	double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[k-1][l])
	double *rho;		// constants in decision functions (rho[k*(k-1)/2])
	double *probA;		// pariwise probability information
	double *probB;

	// for classification only

	int *label;		// label of each class (label[k])
	int *nSV;		// number of SVs for each class (nSV[k])
	// nSV[0] + nSV[1] + ... + nSV[k-1]=l
	// XXX
	int free_sv;		// 1 if svm_model is created by svm_load_model
	// 0 if svm_model is created by svm_train
};

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int read_kernels(FILE *fp,struct svm_parameter *param,int read_num_kernels);
void save_kernels(FILE *fp,const struct svm_parameter *param);
int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_destroy_model(struct svm_model *model);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

extern void (*svm_print_string) (const char *);


#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
