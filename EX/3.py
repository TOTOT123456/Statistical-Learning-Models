# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm


# %%
from statsmodels.stats.outliers_influence \
     import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

# %%
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)


# %% [markdown]
# (a) Use the sm.OLS() function to perform a simple linear regression
# with mpg as the response and horsepower as the predictor. Use
# the summarize() function to print the results. Comment on the
# output. For example:
# i. Is there a relationship between the predictor and the response?
# ii. How strong is the relationship between the predictor and
# the response?
# iii. Is the relationship between the predictor and the response
# positive or negative?
# iv. What is the predicted mpg associated with a horsepower of
# 98? What are the associated 95 % confidence and prediction
# intervals?

# %% [markdown]
# i. Is there a relationship between the predictor and the response?
# ii. How strong is the relationship between the predictor and
# the response?
# iii. Is the relationship between the predictor and the response
# positive or negative?
# iv. What is the predicted mpg associated with a horsepower of
# 98? What are the associated 95 % confidence and prediction
# intervals?

# %%
# Auto = load_data("Auto.csv")

# %%
Auto = load_data("Auto")
Auto.columns
# load_data?

# %%
design = MS(['horsepower'])
x = design.fit_transform(Auto)
x[:4]

# %%
y = Auto['mpg']
model = sm.OLS(y,x)
results = model.fit()
summarize(results)

# %%
results.summary()

# %%
new_df = pd.DataFrame({'horsepower':[98]})
newx = design.transform(new_df)
newx

# %%
new_predictions = results.get_prediction(newx);
new_predictions.predicted_mean

# %% [markdown]
# confidence intervals:

# %%
new_predictions.conf_int(alpha=0.05)

# %% [markdown]
# prediction intervals:

# %%
new_predictions.conf_int(obs=True,alpha=0.05)

# %% [markdown]
# (b) Plot the response and the predictor in a new set of axes ax. Use
# the ax.axline() method or the abline() function defined in the
# lab to display the least squares regression line.

# %%
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)


# %%
# ax = Boston.plot.scatter('lstat', 'medv')
ax = Auto.plot.scatter('horsepower','mpg')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)

# %% [markdown]
# (c) Produce some of diagnostic plots of the least squares regression
# fit as described in the lab. Comment on any problems you see
# with the fit.

# %%
ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');

# %% [markdown]
# 9. This question involves the use of multiple linear regression on the
# Auto data set.

# %% [markdown]
# (a) Produce a scatterplot matrix which includes all of the variables
# in the data set.

# %%
terms = Auto.columns
terms

# %%
X = MS(terms).fit_transform(Auto)
X

# %% [markdown]
# (b) Compute the matrix of correlations between the variables using
# the DataFrame.corr() method.

# %%
Auto.corr?

# %%
Auto.corr()

# %%
new_terms = Auto.columns.drop(['name','mpg'])
new_terms

# %%
X = MS(new_terms).fit_transform(Auto)
M_model = sm.OLS(y,X)
results = M_model.fit()
summarize(results)

# %%
vals = [VIF(X,i)
        for i in range(1,X.shape[1])]
vif = pd.DataFrame({'vif':vals},index=X.columns[1:])

vif


# %% [markdown]
# Produce some of diagnostic plots of the linear regression fit as
# described in the lab. Comment on any problems you see with the
# fit. Do the residual plots suggest any unusually large outliers?
# Does the leverage plot identify any observations with unusually
# high leverage?

# %%
ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');

# %%
infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)


# %% [markdown]
# Fit some models with interactions as described in the lab. Do
# any interactions appear to be statistically significant?

# %% [markdown]
# 我的疑问：是VIF值大的两个来拟合并且添加了interaction后会变得好吗，还是相反？（不过书上好像有讲到）

# %%
x1 = MS(['acceleration','year','origin']).fit_transform(Auto)
model_no_inter = sm.OLS(y,x1)
summarize(model_no_inter.fit())
model_no_inter.fit().summary()

# %%
x2 = MS(['acceleration','year','origin',('year','origin')]).fit_transform(Auto)
model_inter = sm.OLS(y,x2)
summarize(model_inter.fit())
model_inter.fit().summary()

# %%
poly?



# %% [markdown]
# Try a few different transformations of the variables, such as
# log(X),
# √
# X, X2. Comment on your findings.

# %%
design = MS(['horsepower'])
x = design.fit_transform(Auto)
x[:4]

y = Auto['mpg']
model = sm.OLS(y,x)
results = model.fit()
summarize(results)

ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');


# %%
Auto['log_horsepower'] = np.log(Auto['horsepower'])

x = MS(['log_horsepower']).fit_transform(Auto)

model = sm.OLS(y,x)
results = model.fit()

ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');


# %%
X = MS([poly('horsepower', degree=2)]).fit_transform(Auto)
model3 = sm.OLS(y, X)
results3 = model3.fit()
# summarize(results3)
ax = subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');


# %% [markdown]
# 10. This question should be answered using the Carseats data set.

# %% [markdown]
# (a) Fit a multiple regression model to predict Sales using Price,
# Urban, and US.

# %%
carseats = load_data('Carseats')
carseats.columns

# %%
# x10 = MS(['Price','Urban','US'])
carseats

# %%
x10 = MS(['Price','Urban','US']).fit_transform(carseats)
y = carseats['Sales']

model_10 = sm.OLS(y,x10)
result_10 = model_10.fit()


# %% [markdown]
# (b) Provide an interpretation of each coefficient in the model. Be
# careful—some of the variables in the model are qualitative!

# %%
summarize(result_10)

# %%
result_10.summary()

# %% [markdown]
# (c) Write out the model in equation form, being careful to handle
# the qualitative variables properly.

# %% [markdown]
# y = b1x1 + b2x2 + b3x3 + b0

# %% [markdown]
# (d) For which of the predictors can you reject the null hypothesis
# H0 : βj = 0?

# %% [markdown]
# 书上的原话：Rather than rely on the individual coefficients, we can use an F-test
# to test
# 0; this does not depend on the coding. This F-test
# H0:β
# =
# β
# =
# has a p-value of 0.96, indicating that we cannot reject the null hypothesis
# that there is no relationship between
# and region.
# balance

# %% [markdown]
# 但是ch03中好像没有提到如何进行F-test? 现在看p值只能确定Us 和 Price can reject the H0

# %% [markdown]
# (e) On the basis of your response to the previous question, fit a
# smaller model that only uses the predictors for which there is
# evidence of association with the outcome.

# %%
x11 = MS(['Price','US']).fit_transform(carseats)
y = carseats['Sales']

model_11 = sm.OLS(y,x11)
result_11 = model_11.fit()

# %%
result_11.summary()

# %%
anova_lm(result_10, result_11)

# %% [markdown]
# MY ANSWER: result_11(drop.Urban) better than result_10() 。
# 1、从R统计量
# 2、从F统计量
# 3、从anova_lm()

# %% [markdown]
# (g) Using the model from (e), obtain 95 % confidence intervals for
# the coefficient(s).

# %%
result_11.conf_int(alpha=0.05)

# %% [markdown]
# (h) Is there evidence of outliers or high leverage observations in the
# model from (e)?

# %%
ax = subplots(figsize=(8,8))[1]
ax.scatter(result_11.fittedvalues, result_11.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--');

# %%
infl = result_11.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(x11.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)



# %% [markdown]
# 11. In this problem we will investigate the t-statistic for the null hypothesis
# H0 : β = 0 in simple linear regression without an intercept. To
# begin, we generate a predictor x and a response y as follows.

# %%
rng = np.random.default_rng (1)
x = rng.normal(size =100)
y = 2 * x + rng.normal(size =100)

# %%
model_YonX= sm.OLS(y,x)
result_YonX = model_YonX.fit()
summarize(result_YonX)


# %% [markdown]
# (b) Now perform a simple linear regression of x onto y without an
# intercept, and report the coefficient estimate, its standard error,
# and the corresponding t-statistic and p-values associated with
# the null hypothesis H0 : β = 0. Comment on these results.

# %%
model_XonY = sm.OLS(x,y)
result_XonY = model_XonY.fit()
summarize(result_XonY)

# %% [markdown]
# (c) What is the relationship between the results obtained in (a) and
# (b)?

# %% [markdown]
# MY ANSWER:t统计量完全一样

# %% [markdown]
# (e) Using the results from (d), argue that the t-statistic for the regression
# of y onto x is the same as the t-statistic for the regression
# of x onto y.

# %% [markdown]
# MY ANSWER:这一题请看书上的公式，其实那个公式X和Y互换时不改变其结果

# %%
df = pd.DataFrame({'x': x, 'y': y})
fig, ax = subplots(figsize=(8, 8))

ax  = ax.plot(x,y,'o');

# %%
# abline(ax,
#        result_XonY.params[0],
#        result_XonY.params[1],
#        'r--',
#        linewidth=3)


# %% [markdown]
# 12. This problem involves simple linear regression without an intercept.

# %% [markdown]
# (a) Recall that the coefficient estimate ˆ β for the linear regression of
# Y onto X without an intercept is given by (3.38). Under what
# circumstance is the coefficient estimate for the regression of X
# onto Y the same as the coefficient estimate for the regression of
# Y onto X?

# %% [markdown]
# 请先看（3.38），个人理解（3.38）来自有intercept的拟合公式

# %% [markdown]
# MY ANSWER:当sigma square(Xi) = sigma square(Yi) 时 b0相等
# 

# %% [markdown]
# (b) Generate an example in Python with n = 100 observations in
# which the coefficient estimate for the regression of X onto Y
# is different from the coefficient estimate for the regression of Y
# onto X.

# %%
rng = np.random.default_rng (1)
x = rng.normal(size =100)
y = 2 * x + rng.normal(size =100)

# %% [markdown]
# (c) Generate an example in Python with n = 100 observations in
# which the coefficient estimate for the regression of X onto Y is
# the same as the coefficient estimate for the regression of Y onto
# X.

# %%
if 3 * x is y:
    print('YES') 

# %%
x = rng.normal(size=100)
y = x + rng.normal(loc=0,scale=0.00001,size=100)

# %%
model_XonY = sm.OLS(x,y)
result_XonY = model_XonY.fit()
summarize(result_XonY)

# %%
model_YonX= sm.OLS(y,x)
result_YonX = model_YonX.fit()
summarize(result_YonX)



# %% [markdown]
# 13. In this exercise you will create some simulated data and will fit simple
# linear regression models to it. Make sure to use the default random
# number generator with seed set to 1 prior to starting part (a) to
# ensure consistent results.

# %% [markdown]
# (a) Using the normal() method of your random number generator,
# create a vector, x, containing 100 observations drawn from a
# N(0, 1) distribution. This represents a feature, X.

# %%
x = rng.normal(size=100)

# %% [markdown]
# (b) Using the normal() method, create a vector, eps, containing 100
# observations drawn from a N(0, 0.25) distribution—a normal
# distribution with mean zero and variance 0.25.

# %%
eps = rng.normal(loc=0,scale=0.25,size=100)

# %% [markdown]
# (c) Using x and eps, generate a vector y according to the model
# Y = −1 + 0.5X + ϵ. (3.39)
# What is the length of the vector y? What are the values of β0
# and β1 in this linear model?

# %%
b0 = -x.std()
b0

# %%
b1 = 2*eps.std() 
b1

# %%
y = b1 * x + b0

# %% [markdown]
# 这里我想的方法错了 ，eps对应的是方程里的误差项

# %%
y = -1 + .5*x + eps
print('Length of y = ' + str(len(y)))

# %% [markdown]
# (d) Create a scatterplot displaying the relationship between x and
# y. Comment on what you observe.

# %%
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o');

# %% [markdown]
# (e) Fit a least squares linear model to predict y using x. Comment
# on the model obtained. How do ˆ β0 and ˆ β1 compare to β0 and
# β1?

# %%
model_12_YonX = sm.OLS(y,x)
result_12 = model_12_YonX.fit()
summarize(result_12)

# %% [markdown]
# ↑上面这段代码生成的是没有intercept的OLS线性拟合函数

# %%
# design = MS([x])
MS?


# %% [markdown]
# 现在看到了MS函数中terms参数可以是pd.DataFrame …… column names 所以想到了先建立DateFrame再进行fit_transperant()

# %%
terms = pd.DataFrame({'x':x})
X = MS(['x']).fit_transform(terms)
X[:4]

# %%
model_12_YonX = sm.OLS(y,X)
result_12 = model_12_YonX.fit()
summarize(result_12)

# %% [markdown]
# 回顾一下原式是 " y = -1 + 0.5x + eps "

# %% [markdown]
# (f) Display the least squares line on the scatterplot obtained in (d).
# Draw the population regression line on the plot, in a different
# color. Use the legend() method of the axes to create an appropriate
# legend.

# %%
ax.legend?

# %%
# y1 = np.sin(x)
# y2 = np.cos(x)

# # 绘制图形
# ax.plot(x, y1, label='sin(x)')
# ax.plot(x, y2, label='cos(x)')

# # 添加图例
# ax.legend()

# # 显示图形
# # ax.show()
# fig

# %% [markdown]
# 上面这段代码是ax.legend()的测试代码

# %%
# abline(ax,result_12.para)
abline(ax,
       result_12.params[0],
       result_12.params[1],
       'r--',
       linewidth=3,
       label = 'fit')

# %%


# %%
abline(ax,
       -1,
       0.5,
       'g--',
       linewidth=3,
       label = 'population')

# %% [markdown]
# 经过了多次尝试后，发现在abline的参数最后加上 label = 'fit' label = 'population' 是正确的！

# %%
# ax.plot(X,y,label = 'fit')
# ax.plot(x,y,label = 'pop')

# ax.legend(label = 'fit:red')
# ax.legend(label = 'pop:green')

ax.legend()

# %%
fig

# %% [markdown]
# NBBBBBBBBBBBBBBBBBBBBBBBBBBBBB!!!!!!!!!!!!!!!!!!!!!!!!!!

# %% [markdown]
# 这题的解题过程是想到了*arg and *kwargs是类似 "无限量" 的参数，结合abline函数的构造过程 才解的此题

# %% [markdown]
# (g) Now fit a polynomial regression model that predicts y using x
# and x2. Is there evidence that the quadratic term improves the
# model fit? Explain your answer.

# %%
X2 = MS([poly('x',degree=2)]).fit_transform(terms)
model2 = sm.OLS(y,X2)
result2 = model2.fit()

# %% [markdown]
# method 1:" .summary() " and compare to the R-squared and F-statistic "

# %%
result_12.summary()

# %%
result2.summary()

# %% [markdown]
# mothed 2: " anovan_lm "

# %%
anova_lm(result_12,result2)

# %% [markdown]
# (i) Repeat (a)–(f) after modifying the data generation process in
# such a way that there is more noise in the data. The model
# (3.39) should remain the same. You can do this by increasing
# the variance of the normal distribution used to generate the
# error term ϵ in (b). Describe your results.

# %%
x = rng.normal(size=100)
eps = rng.normal(loc=0,scale=0.5,size=100)

y = -1 + .5*x + eps
# print('Length of y = ' + str(len(y)))

fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o');

# %%
terms = pd.DataFrame({'x':x})
X = MS(['x']).fit_transform(terms)
X[:4]

# %%
model_12_YonX = sm.OLS(y,X)
result_12 = model_12_YonX.fit()
summarize(result_12)

# %%
abline(ax,
       result_12.params[0],
       result_12.params[1],
       'r--',
       linewidth=3,
       label = 'fit')

# %%
abline(ax,
       -1,
       0.5,
       'g--',
       linewidth=3,
       label = 'population')

# %%
ax.legend()
fig

# %%
result_12.summary()

# %% [markdown]
# Right anwer : have a much worse fit.The R-squared is just 0.431 and the confidence intervals for the coefficients are much wider. Still there's no doubt we are in the presence of a statistically significant relationship, with very low p-values.
# 
# 

# %% [markdown]
# 15. This problem involves the Boston data set, which we saw in the lab
# for this chapter. We will now try to predict per capita crime rate
# using the other variables in this data set. In other words, per capita
# crime rate is the response, and the other variables are the predictors.

# %% [markdown]
# (a) For each predictor, fit a simple linear regression model to predict
# the response. Describe your results. In which of the models is
# there a statistically significant association between the predictor
# and the response? Create some plots to back up your assertions

# %%
boston = load_data('Boston')
boston.shape

# %%
y = boston['crim']
boston.columns[1]

# %% [markdown]
# 这里是要挨个准备predictor , 所以想写一个 df 来实现

# %%
def make_xi():
    arr = []
    for i in range(1,boston.shape[1]):
       arr.append(MS(boston.columns[i]).fit_transform(boston))

    model = [] 
    for i in range(0,boston.shape[1]-1):
        pred_name = f'pred\_{i}'
        model[pred_name] = sm.OLS(y,arr[i])
#    xi =  MS(boston.columns[i]).fit_transform(boston)
    
        

# %%
# make_xi()

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %%


# %% [markdown]
# 


