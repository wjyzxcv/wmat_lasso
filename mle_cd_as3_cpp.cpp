#if !defined(ARMA_64BIT_WORD)  
#define ARMA_64BIT_WORD  
#endif  
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppArmadillo.h>
#include <RcppEigen.h>
using namespace arma;

double det_sp(const sp_mat arma_A, const char& mode) {
  double output = 0;
  switch(mode){
  case 'a': output = det(mat(arma_A));
    break;
  case 'e':   Eigen::SparseMatrix<double> eigen_s = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::wrap(arma_A));
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(eigen_s);
    double det = solver.logAbsDeterminant();
    output = exp(det);
    break;
  }
  return output;
}

double th_num(const double& n, const double& th_tol){
  double output = 0;
  if (std::abs(n) > th_tol){
    output = n;
  }
  return output;
}

sp_mat make_w_cpp(const int& n, const vec& para){
  sp_mat wmat(n, n);
  for (int i = 0; i<n; ++i){
    uvec ind_w = regspace<uvec>(0, n-1);
    uvec ind_para = regspace<uvec>(i*(n-1), (i+1)*(n-1) - 1);
    ind_w.shed_row(i);
    vec ins_vec(n, fill::zeros);
    ins_vec(ind_w) = para.elem(ind_para);
    wmat.col(i) = ins_vec;
  }
  return wmat.t();
}


bool elem_0_check(const sp_mat& mat_a, const sp_mat& mat_b){
  const sp_mat umat_a =  abs(sign(mat_a));
  const sp_mat umat_b =  abs(sign(mat_b));
  return (umat_a - umat_b).is_zero();
}

double plogl(const double& par_j, const vec& para, const vec& cmats, const double& cmat_d, const int& i, const int& j, const int& n, const int& t, const vec& y, const mat& x, const vec& beta, const double& sigma2, const double& penalty){
  vec new_para = para;
  new_para((n-1)*i + j) = par_j;
  const sp_mat wmat = make_w_cpp(n, new_para);
  const sp_mat temp_mat = speye(n, n) - wmat;
  const vec e = kron(speye(t,t), temp_mat)*y - x*beta;
  const double temp_det = cmat_d + dot(cmats, new_para(regspace<uvec>((n-1)*i, (i+1)*(n-1) - 1)));
  const double llik = -n*t*penalty*std::abs(par_j) + t*log(temp_det) - 0.5/sigma2*dot(e,e);
  return llik;
}

sp_mat main_loop(const mat& x, const vec& y, const int& n, const int& t, const double& penalty, const double& th_tol, const char& det_mode, const bool& full_mode, vec& beta, double& sigma2, vec& temp_par, vec& new_par){
  sigma2 /= 2;
  for (int i=0; i < n; ++i){
    uvec ind_in = regspace<uvec>(i*(n-1), (i+1)*(n-1) - 1);
    if (!temp_par(ind_in).is_zero()|| full_mode){
      uvec ind_it = regspace<uvec>(i, n, n*t-1);
      sp_mat A_mat = speye(n, n) - make_w_cpp(n, new_par);
      vec cmats(n-1);
      const mat x_n = x.rows(ind_it);
      const vec xnbeta = x_n*beta;
      vec new_par_i = new_par(ind_in);
      for (int j = 0; j < n - 2; ++j){
        if (new_par_i(j) ==0){
          cmats(j) = 0;
        } else if (j >= i){
          sp_mat A_cp = A_mat;
          A_cp.shed_row(i);
          A_cp.shed_col(j+1);
          cmats(j) = det_sp(A_cp, det_mode)*pow(-1, i+j+1);
        } else {
          sp_mat A_cp = A_mat;
          A_cp.shed_row(i);
          A_cp.shed_col(j);
          cmats(j) = det_sp(A_cp, det_mode)*pow(-1, i+j);
        }
      }
      sp_mat A_cp = A_mat;
      A_cp.shed_row(i);
      A_cp.shed_col(i);
      double cmat_d = det_sp(A_cp, det_mode);
      for (int j = 0; j< n-1; ++j){
        if (temp_par((n-1)*i+j) != 0|| full_mode){
          uvec ind_nj = regspace<uvec>(0, n-2);
          ind_nj.shed_row(j);
          int k = j;
          if (j >= i){
            k = j + 1;
          }
          A_cp = A_mat;
          A_cp.shed_row(i);
          A_cp.shed_col(k);
          cmats(j) = det_sp(A_cp, det_mode)*pow(-1, i+k);
          new_par_i = new_par(ind_in);
          //bound
          const double lim = 1 - sum(abs(new_par_i(ind_nj)));
          //initialize
          double max_point = 0;
          double llik_rec = -1e9;
          //solve derivative
          //temp values
          const vec temp_yjt = y(regspace<uvec>(k, n, n*t-1));
          const double temp_yjt2 = as_scalar(sum(pow(temp_yjt, 2)));
          vec temp_wj = -new_par_i;
          vec temp_wjt(n, fill::ones);
          uvec ind_wj = regspace<uvec>(0, n-1);
          ind_wj.shed_row(i);
          temp_wj(j) = 0;
          temp_wjt(ind_wj) = temp_wj;
          const sp_mat temp_wjm(kron(speye(t,t), sp_mat(temp_wjt).t()));
          const double temp_ye = sum(dot(temp_yjt, temp_wjm*y - xnbeta));
          double temp_cwk = cmat_d;
          vec new_par_i_2 = new_par_i;
          new_par_i_2(j) = 0;
          temp_cwk += dot(new_par_i_2, cmats);
          const double temp_a = -cmats(j)*temp_yjt2/sigma2;
          const double temp_b1 = (cmats(j)*temp_ye - temp_yjt2 *temp_cwk)/sigma2;
          const double temp_bp = temp_b1 - n*t*penalty*cmats(j);
          const double temp_bn = temp_b1 + n*t*penalty*cmats(j);
          const double temp_c1 = -t*cmats(j) + temp_ye*temp_cwk/sigma2;
          const double temp_cp = temp_c1 - n*t*penalty*temp_cwk;
          const double temp_cn = temp_c1 + n*t*penalty*temp_cwk;
          if (std::abs(temp_a) < datum::eps){
            //linear
            //positive case
            double r_c = -temp_cp/temp_bp;
            if (r_c >0 && r_c < lim && r_c > -lim){
              const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
              if (new_llik > llik_rec){
                max_point = r_c;
                llik_rec = new_llik;
              }
            }
            //negative case
            r_c = -temp_cn/temp_bn;
            if (r_c < 0 && r_c < lim && r_c > -lim){
              const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
              if (new_llik > llik_rec){
                max_point = r_c;
                llik_rec = new_llik;
              }
            }
          }else{
            //quadratic
            //positive case
            double temp_bac = pow(temp_bp, 2) - 4*temp_a*temp_cp;
            if (temp_bac >= 0) {
              double r_c = (-temp_bp + sqrt(temp_bac))/(2*temp_a);
              if (r_c >0 && r_c < lim && r_c > -lim){
                const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
                if (new_llik > llik_rec){
                  max_point = r_c;
                  llik_rec = new_llik;
                }
              }
              r_c = (-temp_bp - sqrt(temp_bac))/(2*temp_a);
              if (r_c >0 && r_c < lim && r_c > -lim){
                const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
                if (new_llik > llik_rec){
                  max_point = r_c;
                  llik_rec = new_llik;
                }
              }
            }
            //negative case
            temp_bac = pow(temp_bn, 2) - 4*temp_a*temp_cn;
            if (temp_bac >= 0) {
              double r_c = (-temp_bn + sqrt(temp_bac))/(2*temp_a);
              if (r_c <0 && r_c < lim && r_c > -lim){
                const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
                if (new_llik > llik_rec){
                  max_point = r_c;
                  llik_rec = new_llik;
                }
              }
              r_c = (-temp_bn - sqrt(temp_bac))/(2*temp_a);
              if (r_c <0 && r_c < lim && r_c > -lim){
                const double new_llik = plogl(r_c, new_par, cmats, cmat_d, i, j, n, t, y, x, beta, sigma2*2, penalty);
                if (new_llik > llik_rec){
                  max_point = r_c;
                  llik_rec = new_llik;
                }
              }
            }
          }
          max_point = th_num(max_point, th_tol);
          new_par((n-1)*i+j) = max_point;
          if (max_point == 0 && j != n-2) cmats(j) = 0;
        }
      }
    }
  }
  //beta and sigma2
  sp_mat new_w = make_w_cpp(n, new_par);
  temp_par = new_par;
  sp_mat temp_mat = kron(speye(t, t), speye(n, n) - new_w);
  beta = (x.t()*x).i()*x.t()*(temp_mat*y);
  sigma2 = dot(temp_mat*y - x*beta, temp_mat*y - x*beta)/(n*t);
  return new_w;
}


// [[Rcpp::export]]
Rcpp::List mle_cd_as3_cpp(const mat& x, const vec& y, const int& n, const int& t, const double& penalty, const double& th_tol, const char& det_mode, const int& max_full_it){
  //initial beta, sigma2
  vec beta = (x.t()*x).i()*x.t()*y;
  double sigma2 = dot(y - x*beta, y - x*beta)/(n*t);
  //temp_variables
  sp_mat temp_w(speye(n,n));
  vec temp_par(n*(n-1), fill::zeros);
  vec new_par(n*(n-1), fill::zeros);
  //first full cycle
  sp_mat new_w = main_loop(x, y, n, t, penalty, th_tol, det_mode, true, beta, sigma2, temp_par, new_par);
  //main cycle
  int full_count = 1;
  int partial_count = 0;
  do{
    do{
      temp_w = new_w;
      new_w = main_loop(x, y, n, t, penalty, th_tol, det_mode, false, beta, sigma2, temp_par, new_par);
      partial_count += 1;
    } while (!elem_0_check(temp_w, new_w));
    temp_w = new_w;
    new_w = main_loop(x, y, n, t, penalty, th_tol, det_mode, true, beta, sigma2, temp_par, new_par);
    full_count += 1;
    if (full_count >= max_full_it) break;
  } while(!elem_0_check(temp_w, new_w));
    //partial cycle
  return Rcpp::List::create(Rcpp::Named("w") = new_w,
                            Rcpp::Named("beta") = beta,
                            Rcpp::Named("sigma2") = sigma2,
                            Rcpp::Named("full_cycle_count") = full_count,
                            Rcpp::Named("partial_cycle_count") = partial_count);
}