# DOCUMENTATION -----------------------------------------------------------

#   Author:     Jesse Bruhn
#   Contact:    jbruhn@princeton.edu

# PREAMBLE ----------------------------------------------------------------

#Clear workspace
rm(list=ls())

#Load Packages
pckgs <- c("tidyverse")
lapply(pckgs, library, character.only=TRUE)

#Set Options
options(scipen=100)

#Clean Workspace
rm(pckgs)

#Session Info for Reproducibility
Sys.Date()
sessionInfo()

# NOTES -------------------------------------------------------------------

#The purpose of this example is to demonstrate the most basic application of
#empirical bayes. It is a varaition on a classic example due to donald rubin
#and explored in detail in Bayesian Data Analysis called the
#"eight schools" example. The package we make should be able to accomodate
#"eight schools" type problems, and we should include this example (solved
#via the package) in the vignette.



# EXAMPLE -----------------------------------------------------------------

# A treatment (academic coaching) is rolled out into 8 different schools.
# A separate analysis of the impact of coaching on SAT scores in each school
# generates the following resutls:

eight_schools <- tibble(
  school = letters[1:8],
  treatment_effect = c(28, 8, -3, 7, -1, 1, 18, 12),
  te_std_error = c(15, 10, 16, 11, 9, 11, 10, 18)
)

print(eight_schools)

# Suppose we want to create the "best" possible guess as to the treatment
# effect in each school.

# The maximum likelihood estimate would be to just use the individual
# treatment effect estimates for each school.

TE_plot <- eight_schools %>%
  ggplot(aes(x = school, y = treatment_effect)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = treatment_effect - 1.96*te_std_error,
                    ymax = treatment_effect + 1.86*te_std_error))

print(TE_plot)

# Observe that the results of each of these experiments in isolation are
# quite noisy. So while we could use the MLE, we will not have much
# confidence in our estimate for any individual school if we use
# each individual result in isolation.

# However, we also observe that most of the treatments generated at least weakly
# positive results, with some treatment effects in fact being quite large.

# This suggests that a better guess might be to pool the information
# across all of the experiments and use that as our guess for the treatment
# effect in each school.

# Under the assumption that each of these treatment effect estimates is
# independent, the maximum likelihood estimate
# for the average effect is just the average of the treatment effects in
# each school weighted by the inverse of the variance.

mean_vec <- eight_schools$treatment_effect
inv_var <- eight_schools$te_std_error^(-2)

pooled_var <- 1/sum(inv_var)

pooled_estimate <- pooled_var*sum(mean_vec*inv_var)

print(pooled_estimate)

#Compare the two predictions for school A

conf_int_unpooled <- c(mean_vec[1] - 1.96*eight_schools$te_std_error[1], mean_vec[1] + 1.96*eight_schools$te_std_error[1])
conf_int_pooled <- c(pooled_estimate - 1.96*sqrt(pooled_var), pooled_estimate + 1.96*sqrt(pooled_var))

# the range of plausible values for the unpooled esitmate is huge. That
# seems undesirable.

# But the upport bound of the range of plausible values using the pooled
# esimtate is very far from the point estimate found with the MLE.

# Which estimate should we use for policy?

# The empirical bayes approach is to use both. Let's take the the MLE
# estimate, which is unbiassed but high variance, and combine that with the
# pooled estimate, which is biassed but low variance, to generate a consensus
# estimate that (we hope) will trade off bias and variance between the
# two approaches to get us ``closer'' in a mean squared error sense to the
# ``true'' vector of treatment effects across the eight schools.

# denote the ``true'' treatment effect in each school by theta_i, and
# suppose that these schools are a random sample from a population of
# potential schools such that theta_i ~ N(theta, sigma).

# observe that the sample disgtribution of the estimated treatment effects
# is theta_i_hat ~ N(theta_i, tau). Thus if theta, sigma, and tau are known,
# we can apply bayes rule to get a posterior mode:
# theta_i_post_mode = [tau^2/(tau^2 + sigma^2)] * theta + [sigma^2/(tau^2 + sigma^2)] * theta_i_hat

# Since theta, sigma, and tau are not known, we will replace them with estimates
# using the pooled average, pooled variance, and estaimte sampling variance
# respectively

theta_i_hat <- mean_vec[1]
tau_hat <- eight_schools$te_std_error[1]

theta_hat <- pooled_estimate
sigma_hat <- sqrt(pooled_var)

theta_i_eb <- tau_hat^2/(tau_hat^2 + sigma_hat^2) * theta_hat + sigma_hat^2/(tau_hat^2 + sigma_hat^2) * theta_i_hat

print(theta_i_eb)

# This seems (intuitively) like a reasonable estimate.

# Now we will do this to the entire vector of treatment effect estiamtes
# for all schools

theta_i_vec <- matrix(mean_vec)
tau_i_mat <- diag(eight_schools$te_std_error^2)

theta_vec <- matrix(rep(pooled_estimate, length.out = 8))
sigma_mat <- diag(rep(pooled_var, length.out = 8))

gamma <- solve(solve(tau_i_mat) + solve(sigma_mat))

W0 <- gamma %*% solve(sigma_mat)
W1 <- gamma %*% solve(tau_i_mat)

theta_i_vec_eb <- W0 %*% theta_vec + W1 %*% theta_i_vec

# Here, gamma turns out to be the matrix of posterior variances
# and W0 and W1 are the empirical bayes shrinkage factors.

# lets look at the pooled, unpooled, empirical bayes estimates
# together with the standard errors and shrinkage factors

results <- tibble(
  pooled = c(theta_vec),
  unpooled = c(theta_i_vec),
  unpooled_se = sqrt(diag(tau_i_mat)),
  eb = c(theta_i_vec_eb),
  posterior_sd = sqrt(diag(gamma)),
  pooled_weight = diag(W0),
  unpooled_weight = diag(W1)
)

print(results)

# observe that more precisely estimated unpooled estiamtes
# have relatively more weight allocated to the unpooled
# estimates. the noisy estimates have more weight allocated
# to the pooled estimate.

# it turns out that these ``consensus estimates'' will joinlty
# be closer to to the ``true'' parameters of interest in a
# mean squared error sense than the MLE (see stein 1956) as
# long as we are estimating at least 3 parameters.


# SYNTAX FOR R PACKAGE ----------------------------------------------------

# For the R package, we would like the function to be able to work as follows

simpleShrink(
  unpooled_estimates, #list of unpooled estimates
  pooled_estimate,
  unpooled_std_error, #list of unpooled std_errors
  pooled_std_error
)

# And, when supplied the appropriate input to these arguments from the
# 8 schools example, simpleShrink should return the empirical bayes estimate
# the posterior standard deviation, and the vectors of shrinkage weights.



