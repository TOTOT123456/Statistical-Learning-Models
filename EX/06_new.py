# %% [markdown]
# 9. In this exercise, we will predict the number of applications received
# using the other variables in the College data set.

# %%
import warnings
warnings.simplefilter("ignore")

# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial


# %%
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
from sklearn.model_selection import train_test_split



# %%
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)
# !pip install l0bnb
from l0bnb import fit_path


# %%
from functools import partial
from sklearn.model_selection import \
     (cross_validate,
      KFold,
      ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm


# %% [markdown]
# (a) Split the data set into a training set and a test set.

# %%
College = load_data('College')
# College = pd.read_csv('College.csv')

# %% [markdown]
# > !!!!!这里使用load_data还是pd.read_csv其实是有区别：一个是按列读取的，一个是按行读取的

# %%
College.shape
# College.dropna

# %%
# College.dropna
College.columns

# %%
College.dropna

# %%
College = pd.get_dummies(College, columns=['Private'], drop_first = True)


# %%
# Auto = load_data('Auto')
# Auto_train, Auto_valid = train_test_split(Auto,
#                                          test_size=196,
#                                          random_state=0)
coll_train,coll_valid = train_test_split(College,
                                         test_size=194,
                                         random_state=0)
coll_train.shape,coll_valid.shape                                    

# %% [markdown]
# (b) Fit a linear model using least squares on the training set, and
# report the test error obtained.

# %%
College[:10]

# %%
# College = College.set_index('name')
# Unnamed: 0

# College_re = College.set_index('Unnamed: 0')
# College_re.columns


# %% [markdown]
# > 这里写College = College.set_index('Unnamed: 0')居然是错误的！！！

# %%
design = MS(College.columns.drop('Accept')).fit(College)
Y = np.array(College['Accept'])
# Y = College['Accept']
X = design.transform(College)
# sigma2 = OLS(Y,X.astype(float)).fit().scale

sigma2 = OLS(Y,X).fit().scale

# %% [markdown]
# 这里
# sigma2 = OLS(Y,X.astype(float)).fit().scale
# 会报错，原因可能是因为College第一列为文字

# %%
# Hitters = load_data('Hitters')
# Hitters[:10]

# %%
College['Apps']

# %%
# College['Private']

# %%
X.info()

# %% [markdown]
# > X.info()，这将显示可用的列及其数据类型。确认它们都是数字性质的。如果它们是 Object 或 bool，则对值进行适当的转换，以便将它们转换为 1 和 0，然后运行 ​​OLS。

# %%
College['Private_Yes']

# %%
strategy = Stepwise.fixed_steps(design,
                                len(design.terms),
                                direction='forward')
full_path = sklearn_selection_path(OLS, strategy)



# %%
full_path.fit(College, Y)
Yhat_in = full_path.predict(College)
Yhat_in.shape



# %%
mse_fig, ax = subplots(figsize=(8,8))
insample_mse = ((Yhat_in - Y[:,None])**2).mean(0)
n_steps = insample_mse.shape[0]
ax.plot(np.arange(n_steps),
        insample_mse,
        'k', # color black
        label='In-sample')
ax.set_ylabel('MSE',
              fontsize=20)
ax.set_xlabel('# steps of forward stepwise',
              fontsize=20)
ax.set_xticks(np.arange(n_steps)[::2])
ax.legend()
# ax.set_ylim([50000,250000]);


# %%
K = 5
kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)
Yhat_cv = skm.cross_val_predict(full_path,
                                College,
                                Y,
                                cv=kfold)
Yhat_cv.shape


# %%
cv_mse = []
for train_idx, test_idx in kfold.split(Y):
    errors = (Yhat_cv[test_idx] - Y[test_idx,None])**2
    cv_mse.append(errors.mean(0)) # column means
cv_mse = np.array(cv_mse).T
# cv_mse.shape

cv_mse


# %%
ax.errorbar(np.arange(n_steps), 
            cv_mse.mean(1),
            cv_mse.std(1) / np.sqrt(K),
            label='Cross-validated',
            c='r') # color red
ax.set_ylim([0,1000000]) # 还以为出错了，原来是y轴的范围没弄对
ax.legend()
mse_fig


# %%
validation = skm.ShuffleSplit(n_splits=1, 
                              test_size=0.2,
                              random_state=0)
for train_idx, test_idx in validation.split(Y):
    full_path.fit(College.iloc[train_idx],
                  Y[train_idx])
    Yhat_val = full_path.predict(College.iloc[test_idx])
    errors = (Yhat_val - Y[test_idx,None])**2
    validation_mse = errors.mean(0)


# %%
ax.plot(np.arange(n_steps), 
        validation_mse,
        'b--', # color blue, broken line
        label='Validation')
ax.set_xticks(np.arange(n_steps)[::2])
# ax.set_ylim([50000,250000])

ax.set_ylim([0,1000000]) # 还以为出错了，原来是y轴的范围没弄对
ax.legend()
mse_fig


# %% [markdown]
# (c) Fit a ridge regression model on the training set, with λ chosen
# by cross-validation. Report the test error obtained.

# %%
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
# Xs.dropna()
# print(Xs.isnull().any())

lambdas = 10**np.linspace(8, -2, 100) / Y.std()
# soln_array = skl.ElasticNet.path(Xs,
#                                  Y,
#                                  l1_ratio=0.,
#                                  alphas=lambdas)[1]
# soln_array.shape


# %%
print(Xs.isnull().any())

# %% [markdown]
# > print(Xs.isnull().any())可以看变量是否为NAN

# %% [markdown]
# > 因此这里需要将截距去掉！

# %%
D = design.fit_transform(College)
D = D.drop('intercept', axis=1)
X = np.asarray(D)


# %%
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
lambdas = 10**np.linspace(8, -2, 100) / Y.std()
soln_array = skl.ElasticNet.path(Xs,
                                 Y,
                                 l1_ratio=0.,
                                 alphas=lambdas)[1]
soln_array.shape


# %%
soln_path = pd.DataFrame(soln_array.T,
                         columns=D.columns,
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'
soln_path


# %%
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');


# %%
beta_hat = soln_path.loc[soln_path.index[39]]
lambdas[39], beta_hat


# %%
np.linalg.norm(beta_hat)


# %%
beta_hat = soln_path.loc[soln_path.index[59]]
lambdas[59], np.linalg.norm(beta_hat)


# %%
ridge = skl.ElasticNet(alpha=lambdas[59], l1_ratio=0)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
pipe.fit(X, Y)


# %%
validation = skm.ShuffleSplit(n_splits=1,
                              test_size=0.5,
                              random_state=0)
ridge.alpha = 0.01
results = skm.cross_validate(ridge,
                             X,
                             Y,
                             scoring='neg_mean_squared_error',
                             cv=validation)
-results['test_score']


# %%
ridge.alpha = 1e10
results = skm.cross_validate(ridge,
                             X,
                             Y,
                             scoring='neg_mean_squared_error',
                             cv=validation)
-results['test_score']


# %%
param_grid = {'ridge__alpha': lambdas}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=validation,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)
grid.best_params_['ridge__alpha']
grid.best_estimator_


# %%
ridge_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
# ax.set_ylim([50000,250000])

ax.set_ylim([0,1000000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);


# %%
ridgeCV = skl.ElasticNetCV(alphas=lambdas, 
                           l1_ratio=0,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('ridge', ridgeCV)])
pipeCV.fit(X, Y)


# %%
tuned_ridge = pipeCV.named_steps['ridge']
ridgeCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(lambdas),
            tuned_ridge.mse_path_.mean(1),
            yerr=tuned_ridge.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_ridge.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);


# %%
np.min(tuned_ridge.mse_path_.mean(1))


# %%
outer_valid = skm.ShuffleSplit(n_splits=1, 
                               test_size=0.25,
                               random_state=1)
inner_cv = skm.KFold(n_splits=5,
                     shuffle=True,
                     random_state=2)
ridgeCV = skl.ElasticNetCV(alphas=lambdas,
                           l1_ratio=0,
                           cv=inner_cv)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('ridge', ridgeCV)]);


# %%
results = skm.cross_validate(pipeCV, 
                             X,
                             Y,
                             cv=outer_valid,
                             scoring='neg_mean_squared_error')
-results['test_score']


# %% [markdown]
# (d) Fit a lasso model on the training set, with λ chosen by crossvalidation.
# Report the test error obtained, along with the number
# of non-zero coefficient estimates.

# %%
lassoCV = skl.ElasticNetCV(n_alphas=100, 
                           l1_ratio=1,
                           cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler),
                         ('lasso', lassoCV)])
pipeCV.fit(X, Y)
tuned_lasso = pipeCV.named_steps['lasso']
tuned_lasso.alpha_


# %%
lambdas, soln_array = skl.Lasso.path(Xs, 
                                    Y,
                                    l1_ratio=1,
                                    n_alphas=100)[:2]
soln_path = pd.DataFrame(soln_array.T,
                         columns=D.columns,
                         index=-np.log(lambdas))


# %%
path_fig, ax = subplots(figsize=(8,8))
soln_path.plot(ax=ax, legend=False)
ax.legend(loc='upper left')
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficiients', fontsize=20);


# %%
np.min(tuned_lasso.mse_path_.mean(1))

# %% [markdown]
# 323…… < 325…… (岭回归和套索)

# %%
lassoCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso.alphas_),
            tuned_lasso.mse_path_.mean(1),
            yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);

# %%

tuned_lasso.coef_

# %% [markdown]
# (e) Fit a PCR model on the training set, with M chosen by crossvalidation.
# Report the test error obtained, along with the value
# of M selected by cross-validation.

# %%
pca = PCA(n_components=2)
linreg = skl.LinearRegression()
pipe = Pipeline([('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_


# %%
pipe = Pipeline([('scaler', scaler), 
                 ('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_


# %%
param_grid = {'pca__n_components': range(1, 17)}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)


# %%
pcr_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['pca__n_components']
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
# ax.set_ylim([50000,250000]);


# %%
# grid.named_steps['linreg'].coef_

# %% [markdown]
# > 这里想看最低点的变量系数是多少

# %%
pca = PCA(n_components=5)
pipe = Pipeline([('scaler', scaler), 
                 ('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_


# %%
pca = PCA(n_components=9)
pipe = Pipeline([('scaler', scaler), 
                 ('pca', pca),
                 ('linreg', linreg)])
pipe.fit(X, Y)
pipe.named_steps['linreg'].coef_


# %% [markdown]
# > 这里想到看最低点系数的方法是重新拟合一个新的pca

# %%
Xn = np.zeros((X.shape[0], 1))
cv_null = skm.cross_validate(linreg,
                             Xn,
                             Y,
                             cv=kfold,
                             scoring='neg_mean_squared_error')
-cv_null['test_score'].mean()


# %%
-cv_null['test_score']

# %%
pipe.named_steps['pca'].explained_variance_ratio_


# %% [markdown]
# (f) Fit a PLS model on the training set, with M chosen by crossvalidation.
# Report the test error obtained, along with the value
# of M selected by cross-validation.

# %%
pls = PLSRegression(n_components=2, 
                    scale=True)
pls.fit(X, Y)


# %%
param_grid = {'n_components':range(1, 20)}
grid = skm.GridSearchCV(pls,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X, Y)


# %%
pls_fig, ax = subplots(figsize=(8,8))
n_comp = param_grid['n_components']
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_ylabel('Cross-validated MSE', fontsize=20)
ax.set_xlabel('# principal components', fontsize=20)
ax.set_xticks(n_comp[::2])
# ax.set_ylim([50000,250000]);


# %% [markdown]
# (g) Comment on the results obtained. How accurately can we predict
# the number of college applications received? Is there much
# difference among the test errors resulting from these five approaches?

# %% [markdown]
# 


