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

#Set random number seed for reproducibility
set.seed(343408)

# NOTES -------------------------------------------------------------------

# This is inspired by another classic application of empirical bayes. In
# baseball, a pitcher throws a ball. The hitter has to hit the ball with a
# bat and run to a white bag called a base. If the hitter misses the ball
# or gets tagged with the ball after hitting it, he is out. If not, he
# is said to get a "hit".

# Thus every time a batter steps up to the plate, they either get out (0)
# or get a hit (1). The number of times a batter gets a hit, divided by the
# total number of times they attempt to hit is known as a batters "Batting
# Average". Players with a high batting average are more valuable then
# players with a low batting average.

# We want to use data on players historical batting averages to predict
# who will be the most valuable players in a subsequent year.


# EXAMPLE -----------------------------------------------------------------

#First we extract data from the ``Lahman'' package for the years 2000 - 2018.
require("Lahman")
baseball_data <-  Lahman::Batting %>%
  as_tibble() %>%
  filter(yearID>=2000)

#only keep the variables we need
baseball_data <- baseball_data %>%
  select(playerID, yearID, AB, H, X2B, X3B, HR)

#remove player-years who had 0 at bats
baseball_data <- baseball_data %>%
  filter(AB>1)

#and calculate batting averages
baseball_data <- baseball_data %>%
  mutate(batting_average = H/AB)

#Creat a player ID factor variable that will be useful later
baseball_data <- baseball_data %>%
  mutate(player_factor = as_factor(playerID))

#Now we split the data into training and testing data. We will use the
#earlier data to train the model and the later date (2018) to test our
#model fit
train <- baseball_data %>%
  filter(yearID < 2018)

test <- baseball_data %>%
  filter(yearID == 2018)

#Now we restrict the data to players who had at least 250 at bats in 2018.
#This is to ensure that we are only trying to predict batting averages
#for players for whom we can estimate a reasonably precise batting average in 2018.
#This will give us a reseasonable amount of precision for the out-of-sample
#exercise.
test <- test %>%
  filter(AB >= 250)

train <- train %>%
  filter(playerID %in% unique(test$playerID))

#now filter out players in the test data for whom we have no training
#data
test <- test %>%
  filter(playerID %in% unique(train$playerID))

#finally suppose there are only 30 players of interest that we are scouting
#and hence we are only interested in predicting the 2018 performance of these
#20 players. Restricting the data in this way it is not essential,
#but it will make the final results more dramatic because it shows that the method
#works well even when the pooled esitmate is still noisy.
test <- sample_n(test, size = 50)

train <- train %>%
  filter(playerID %in% unique(test$playerID))

#observe that we have a very unequal number of at-bats and years of data
#across players in the training data
unequal <- train %>%
  group_by(playerID) %>%
  summarise(years_of_data = n(),
            total_at_bats = sum(AB),
            average_batting_average = mean(batting_average)) %>%
  arrange(total_at_bats)

head(unequal)

tail(unequal)

hist(unequal$total_at_bats, breaks = seq(from = 1, to = 5500, by = 100))

hist(unequal$average_batting_average, breaks = seq(from = 0, to = .4, by = .025))

#The estimated batting averages for the guys with 1000-5000 at bats may be reliable
#But it is probably not a good way to judge the value of someone with only 100 at bats.

#What if we instead used a pooled esimate for all the players as our prediction for
#their future performance?

pooled_est <- mean(train$batting_average)

print(pooled_est)

#This seems like a better bet for the players without much experience,
#but is probably not a good way to evaluate the players who have a long
#history for us to judge.

#Now let's use empirical bayes to synthesize both estimates.

#First, we will get our unpooled estimates from a linear model that assumes
#homeskedastic errors. Note tht we weight by the numer of at bats to recover
#the estimates we would

unpooled.model <- lm(batting_average ~ player_factor, train, weights = AB)

test <- test %>%
  mutate(unpooled_estimate = predict(unpooled.model, test),
         unpooled_std_error = predict(unpooled.model, test, se.fit = TRUE)$se.fit)

# unpooled.model <- lm(batting_average ~ -1 + player_factor, train, weights = AB)
#
# unpooled.model.coefs <- broom::tidy(unpooled.model) %>%
#   mutate(playerID = str_replace(term, "player_factor", "")) %>%
#   select(playerID, estimate, std.error) %>%
#   rename(unpooled_estimate = estimate,
#          unpooled_std_error = std.error)
#
# test <- test %>%
#   left_join(unpooled.model.coefs)

# plot(test$unpooled_estimate, test$batting_average)

#now let's get the pooled estimate using a regression as well
pooled.model <- lm(batting_average ~ 1, train, weights = AB)

test <- test %>%
  mutate(pooled_estimate = predict(pooled.model, test))

#which estimate should we use? Why not both? Denote player i's batting
#average by theta_i. Our esimtate of the batting average is distributed
#theta_i_hat ~ N(theta_i, tau_i_hat) where tau_i_hat is the standard
#error of the sampling distribution. Now suppose that the individual
#batting averages are drawn from a normally distributed population of
#heterogenous batting averages such that theta_i ~ N(theta, sigma)

#Note that we have already calculated theta_i_hat, tau_i_hat, so
#let's collect those here in matrix form:
theta_i_hat <- matrix(test$unpooled_estimate)
tau_i_hat <- diag(test$unpooled_std_error^2)

#We don't know theta or sigma. But we can estimate them.

#Theta we can estimate via the corresponding pooled regression
pooled.model <- lm(batting_average ~ 1, train, weights = AB)

test <- test %>%
  mutate(pooled_estimate = predict(pooled.model, test))

theta_hat <- matrix(test$pooled_estimate)

#To estimate sigma, note that the difference between the residuals
#of the pooled model and the unpooled model is simply theta_i. Thus
#var(u_pooled - u_unpooled) = var(theta_i) = sigma^2
sigma_hat <- diag(rep(var(pooled.model$residuals - unpooled.model$residuals),
                      length.out = nrow(test)
                      )
                  )

#now let's calculate the empirical bayes estimates
tau_i_hat_inv <- solve(tau_i_hat)
sigma_hat_inv <- solve(sigma_hat)

Gamma <- solve(tau_i_hat_inv + sigma_hat_inv)

W0 <- Gamma %*% sigma_hat_inv
W1 <- Gamma %*% tau_i_hat_inv

theta_eb <- W0 %*% theta_hat + W1 %*% theta_i_hat

test <- test %>%
  mutate(eb_estimate = theta_eb)

#Now we will compare the predicted batting averages we estimated using the
#earlier data to the actual realizations in 2018
accuracy <- test %>%
  summarise(mse_unpooled = mean((batting_average - unpooled_estimate)^2),
            mse_pooled = mean((batting_average - pooled_estimate)^2),
            mse_eb = mean((batting_average - eb_estimate)^2)
  )

print(accuracy)

#And we see that the empirical bayes is closer to the true parameters on
#average than either hte unpooled estimator or the pooled estimator. In fact
#the improvements are substantial:

accuracy$mse_unpooled/accuracy$mse_eb
accuracy$mse_pooled/accuracy$mse_eb

#and eb just crushes the unpooled and the pooled estimators on MSE out of sample.
#Let's see why by ploting all three estimates for each player with the acompanying
#errorbars for the unpooled estimate

test %>%
  arrange(unpooled_estimate) %>%
  ggplot(aes(x = as_factor(playerID), y = unpooled_estimate, alpha = .5)) +
  geom_point(color = "blue") +
  geom_pointrange(aes(ymin = unpooled_estimate - 1.96*unpooled_std_error,
                      ymax = unpooled_estimate + 1.96*unpooled_std_error, )) +
  geom_point(aes(y = eb_estimate), color = "red") +
  geom_hline(yintercept = test$pooled_estimate[1], color = "blue")


#the empirical bayes takes the noisy estimates and
#shrinks them towards the pooled mean while leaving the more precisley
#estimated batting averages alone


# SYNTAX FOR R PACKAGE ----------------------------------------------------

#This example differs from 8 schools because we are estimating the parameters
#ourselves in linear models using microdata, rather than just taking
#point estimates + standard errors from a set of studies.

#I think we want to make it easy for the user to just pass the models they have
#estimated direclty to our package and let the work happen in the back end. This
#will also require the user to specific a formula to designate which parameters
#in the unpooled model should be shrunk, as well as which parameters in the pooled
#model describe the pooled mean.

#For the baseball example, the function call would look like:

simpleShrink(
  pooled_model,
  unpooled_model,
  shrink_formula = player_factor ~ 1
  )

#and return the weights, the unpooled estimates, the empirical bayes estimates,
#posterior variance, etc.



# MORE COMPLICATED BASEBALL EXAMPLE ---------------------------------------

#we now consider the same example as before, but we will allow the
#pooled mean to vary with additional observables. I.E. we will assume
#that theta_i ~ N(X'B, sigma) where X' is a vector of other observable
#characteristics (in this case, weight, height, and age). First, we need to
#bring in some additional info about the players

player_info <- Lahman::People %>%
  select(playerID, weight, height, birthYear)

train <- train %>%
  left_join(player_info) %>%
  mutate(age = yearID - birthYear)

test <- test %>%
  left_join(player_info) %>%
  mutate(age = yearID - birthYear)

#

alt_pooled_model <- lm(batting_average ~ weight + height + age, train, weights = AB)

test <- test %>%
  mutate(alt_pooled_estimate = predict(alt_pooled_model, test))

alt_theta_hat <- matrix(test$alt_pooled_estimate)

#note that tau_i_hat and sigma_hat are unaffected by the more complicated
#pooled model. Thus we can use the same weights as before to calculate EB.
alt_theta_eb <- W0 %*% alt_theta_hat + W1 %*% theta_i_hat

test <- test %>%
  mutate(alt_eb_estimate = alt_theta_eb)

test %>%
  select(unpooled_estimate, pooled_estimate, eb_estimate, alt_pooled_estimate, alt_eb_estimate)

#and we will compare model fit out of sample once again.
accuracy <- test %>%
  summarise(mse_unpooled = mean((batting_average - unpooled_estimate)^2),
            mse_pooled = mean((batting_average - pooled_estimate)^2),
            mse_eb = mean((batting_average - eb_estimate)^2),
            mse_alt_eb = mean((batting_average - alt_eb_estimate)^2)
  )

print(accuracy)

accuracy$mse_unpooled/accuracy$mse_alt_eb
accuracy$mse_pooled/accuracy$mse_alt_eb
accuracy$mse_eb/accuracy$mse_alt_eb

#and we see that the richer model for the pooled mean yields some small
#improvement.

# SYNTAX FOR R PACKAGE ----------------------------------------------------

#For the more complicated baseball example, the function call would look like:

simpleShrink(
  pooled_model,
  unpooled_model,
  shrink_formula = player_factor ~ height + weight + age
)

#and return the weights, the unpooled estimates, the empirical bayes estimates,
#posterior variance, etc.



