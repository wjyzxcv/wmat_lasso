#For rCMA version 1.1, you might need to add "if (!is.matrix(popR)) popR = t(popR)" after line 235 in rCMA/R/rCMA.R
library(rCMA)
library(Matrix)

#source mle_cd_nx_as3_cpp.cpp and mle_cd_as3_cpp.cpp
ml_lasso_1 = function(dataset, th_tol, max_full_it = 20){
  s_time = Sys.time()
  n = dataset$n
  t = dataset$t
  x = as.matrix(dataset$x[,-c(1,2), drop = F])
  y = dataset$y[,-c(1,2)]
  y = (y- mean(y))/sd(y)
  if (any(is.na(x))){
    BIC = function(penalty_level, y, n ,t, th_tol, det_mode, max_full_it){
      lasso_res = mle_cd_nx_as3_cpp(y = y, n = n ,t = t, penalty = penalty_level, th_tol = th_tol, det_mode = det_mode, max_full_it = max_full_it)
      if (lasso_res$full_cycle_count >= max_full_it) return(1e9)
      iw_mat = Diagonal(n) - lasso_res$w
      e = kronecker(Diagonal(t), iw_mat)%*%y
      logl = -(n*t)/2*log(lasso_res$sigma2) + t*determinant(Diagonal(n) - lasso_res$w)$modulus - .5/lasso_res$sigma2*crossprod(e)
      logl = as.numeric(logl)
      n_par = as.numeric(sum(lasso_res$w != 0) + 1)
      return(n_par*log(n*t) - 2*logl)
    }
    #grid search upper limit
    pl_grid = c(0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    pl_upper = 1
    for (pl in pl_grid){
      pl_grid_fit = mle_cd_nx_as3_cpp(y = y, n = n ,t = t, penalty = pl, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it)
      if (sum(pl_grid_fit$w) == 0) {pl_upper = pl; break}
    }
    message("upper bound: ", pl_upper)
    objfun_rcma = function(par) {BIC(penalty_level = par, y = y, n = n ,t = t, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it, ebic_par = ebic_par)}
    #modify CMAEvolutionStrategy_mod.properties to tune the optimizer
    cma_obj = cmaNew("./CMAEvolutionStrategy_mod.properties")
    cmaInit(cma_obj)
    opt_pl = cmaOptimDP(cma_obj, objfun_rcma, isFeasible = function (x) {x>0 & x< pl_upper})
    message(paste0("optimal penalty level: ", opt_pl$bestX))
    opt_lasso = mle_cd_nx_as3_cpp(y = y, n = n ,t = t, penalty = opt_pl$bestX, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it)
  }else{
    BIC = function(penalty_level, x, y, n ,t, th_tol, det_mode, max_full_it){
      lasso_res = mle_cd_as3_cpp(x = x, y = y, n = n ,t = t, penalty = penalty_level, th_tol = th_tol, det_mode = det_mode, max_full_it = max_full_it)
      if (lasso_res$full_cycle_count >= max_full_it) return(1e9)
      iw_mat = Diagonal(n) - lasso_res$w
      e = kronecker(Diagonal(t), iw_mat)%*%y - x%*%lasso_res$beta
      logl = -(n*t)/2*log(lasso_res$sigma2) + t*determinant(Diagonal(n) - lasso_res$w)$modulus - .5/lasso_res$sigma2*crossprod(e)
      logl = as.numeric(logl)
      n_par = as.numeric(sum(lasso_res$w != 0) + 1 + length(lasso_res$beta))
      return(n_par*log(n*t) - 2*logl)
    }
    #grid search upper limit
    pl_grid = c(0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    pl_upper = 1
    for (pl in pl_grid){
      pl_grid_fit = mle_cd_as3_cpp(x = x, y = y, n = n ,t = t, penalty = pl, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it)
      if (sum(pl_grid_fit$w) == 0) {pl_upper = pl; break}
    }
    message("upper bound: ", pl_upper)
    objfun_rcma = function(par) {BIC(penalty_level = par, x = x, y = y, n = n ,t = t, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it, ebic_par = ebic_par)}
    #modify CMAEvolutionStrategy_mod.properties to tune the optimizer
    cma_obj = cmaNew("./CMAEvolutionStrategy_mod.properties")
    cmaInit(cma_obj)
    opt_pl = cmaOptimDP(cma_obj, objfun_rcma, isFeasible = function (x) {x>0 & x< pl_upper})
    message(paste0("optimal penalty level: ", opt_pl$bestX))
    opt_lasso = mle_cd_as3_cpp(y = y, x = x, n = n ,t = t, penalty = opt_pl$bestX, th_tol = th_tol, det_mode = "a", max_full_it = max_full_it)
  }
  output = list(penalty_optim = opt_pl, lasso = opt_lasso)
  output$time_elapsed = difftime(Sys.time(), s_time, units = "min")
  return(output)
}
