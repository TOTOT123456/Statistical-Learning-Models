# %%
1 + 2

# %%
import numpy as np

# %%
x = np.random.normal(size=10)

x

# %%
import pandas as pd
college = pd.read_csv('College.csv')
college

# %%
college.columns

# %%
idx_apps = college['Apps'] > 10000

college[idx_apps]

# %%
college[idx_apps].shape

# %% [markdown]
# Look at the data used in the notebook by creating and running
# a new cell with just the code college in it. You should notice
# that the first column is just the name of each university in a
# column named something like *Unnamed: 0*. We don’t really want
# pandas to treat this as data. However, it may be handy to have
# these names for later. Try the following commands and similarly
# look at the resulting data frames:

# %%
college2 = pd.read_csv('College.csv', index_col=0)
college3 = college.rename({'Unnamed: 0': 'College'},axis=1)
college3 = college3.set_index('College')

# college
college3
# college2

# %%
college = college3

# %% [markdown]
# (c) Use the describe() method of to produce a numerical summary
# of the variables in the data set.

# %%
college[['Private','Apps']].describe()

# %% [markdown]
# (d) Use the pd.plotting.scatter_matrix() function to produce a
# scatterplot matrix of the first columns [Top10perc, Apps, Enroll].
# Recall that you can reference a list C of columns of a data frame
# A using A[C].

# %%
# mat = pd.plotting.scatter_matrix
pd.plotting.scatter_matrix?

# %%
mat = pd.plotting.scatter_matrix(college[['Top10perc','Apps','Enroll']])

# %% [markdown]
# (e) Use the boxplot() method of college to produce side-by-side
# boxplots of Outstate versus Private.

# %%
college.boxplot?

# %%
college.boxplot('Outstate','Private')

# %%
college.boxplot(column='Outstate',by='Private')

# %% [markdown]
# (f) Create a new qualitative variable, called Elite, by binning the
# Top10perc variable into two groups based on whether or not the
# proportion of students coming from the top 10% of their high
# school classes exceeds 50%.

# %%
college['Elite'] = pd.cut(college['Top10perc'],[0,0.5,1],labels=['No', 'Yes'])

college['Elite']

# idx2 = college['Top10perc'] > 50

# college[idx2]

# %%
college['Elite'].value_counts()

# %%
college['Elite'] = np.where(college['Top10perc'] > 50, 'Yes', 'No')

college['Elite']

# %%
np.where?

# %%
college['Elite'].value_counts(normalize=True)

# %%
college.boxplot(column='Outstate',by='Elite',figsize=(10,6))

# %%



