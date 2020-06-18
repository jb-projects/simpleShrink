#' Simple shrink
#' 
#' \code{simpleShrink} takes pooled and unpooled estimates as input and returns 
#' EB estimates and corresponding shrinkage weights.
#' 
#' @param unpooled_estimates A vector of unpooled point estimates.
#' @param pooled_estimate A value of pooled point esitmates
#' @param unpooled_std_error A vector of unpooled standard errors.
#' @param pooled_std_error A value of pooled standard error.
#' 
#' @return A list of components:
#' \describe{
#'   \item{eb}{A vector of EB estimates.}
#'   \item{posterior_sd}{A vector of posterior standard deviations.}
#'   \item{pooled_weight}{A vector of shrinkage weights on the pooled estimate.}
#'   \item{unpooled_weight}{A vector of shrinkage weights on unpooled estimates.}
#' }


simpleShrink <- function(unpooled_estimates, 
                         pooled_estimate, 
                         unpooled_std_error, 
                         pooled_std_error) {
  
  num_par <- length(unpooled_estimates)
      # number of parameters
  
  theta_i_vec <- matrix(unpooled_estimates)
  tau_i_mat   <- diag(unpooled_std_error^2)
      # convert vector unpooled estimates and std errors to matrices
  
  pooled_var <- 1/sum(unpooled_std_error^(-2))
  theta_vec  <- matrix(rep(pooled_estimate, length.out = num_par))
  sigma_mat  <- diag(rep(pooled_var, length.out = num_par))
      # convert pooled estimate and standard error to matrices
  
  gamma        <- solve(solve(tau_i_mat) + solve(sigma_mat))
  posterior_sd <- sqrt(diag(gamma))
      # compute posterior variance matrix
  
  W0 <- gamma %*% solve(sigma_mat)
  W1 <- gamma %*% solve(tau_i_mat)
      # compute shrinkage weight matrices
  
  theta_i_vec_eb <- c(W0 %*% theta_vec + W1 %*% theta_i_vec)
      # compute EB estimates
  
  return(list(eb = theta_i_vec_eb, 
              posterior_sd = posterior_sd, 
              pooled_weight = diag(W0), 
              unpooled_weight = diag(W1)))
}
