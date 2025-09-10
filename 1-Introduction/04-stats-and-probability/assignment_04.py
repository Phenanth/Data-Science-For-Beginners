import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats

from scipy.stats import ttest_ind


cols = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6", "Y"]

df = pd.read_csv("./diabetes.tsv", sep="\t", header=0, dtype=float)

# Task 1: Compute mean values and variance for all values

means = df[['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y']].mean()
print(means)

vars = df[['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y']].var()
print(vars)

# Task 2: Plot boxplots for BMI, BP and Y depending on gender

df.boxplot(column=['BMI'], by='SEX', figsize=(5,4))
plt.xticks(rotation='horizontal')
plt.tight_layout()
plt.show()

df.boxplot(column=['BP'], by='SEX', figsize=(5,4))
plt.xticks(rotation='horizontal')
plt.tight_layout()
plt.show()

df.boxplot(column=['Y'], by='SEX', figsize=(5,4))
plt.xticks(rotation='horizontal')
plt.tight_layout()
plt.show()

# Task 3: What is the the distribution of Age, Sex, BMI and Y variables?

# Age: Normal
df['AGE'].hist(bins=100, figsize=(10,6))
plt.suptitle('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Sex: Uniform
df['SEX'].hist(bins=3, figsize=(10,6))
plt.suptitle('Sex distribution')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# BMI: Normal
df['BMI'].hist(bins=40, figsize=(10,6))
plt.suptitle('BMI distribution')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Y: Normal
df['Y'].hist(bins=500, figsize=(10,6))
plt.suptitle('Y distribution')
plt.xlabel('Y')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Task 4: Test the correlation between different variables and disease progression (Y)
# Hint Correlation matrix would give you the most useful information on which values are dependent.

cor = np.cov(df["AGE"], df["Y"])
coef = np.corrcoef(df["AGE"], df["Y"])
print(cor[0,1])
print(coef[0,1])

# other facotrs are alike.

# Task 5: Test the hypothesis that the degree of diabetes progression is different between men and women

# 1. Test the confidence intervals

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

df_sex = df.groupby('SEX').agg({'Y': 'mean'}).rename(columns={'Y': 'mean'})
print(df_sex)

for p in [0.85,0.9,0.95]:
	m1, y1 = mean_confidence_interval(df.loc[df['SEX']==1.0, ['Y']], p)
	m2, y2 = mean_confidence_interval(df.loc[df['SEX']==2.0, ['Y']], p)
	print(f'Conf={p:.2f}, progress of man: {m1-y1[0]:.2f}~{m1+y1[0]:.2f}, progress of woman: {m2-y1[0]:.2f}~{m2+y2[0]:.2f}')

# 2. Student t-test
# P value: The probability that two distributions has the same mean. (the lower, the less likely that two distributions are alike.)
# T value: An intermidiate value of normalized mean difference that is used in the t-test. It is compared against a threshold value for a given confidence value. (?)

tval, pval = ttest_ind(df.loc[df['SEX'] == 1.0, ['Y']], df.loc[df['SEX'] == 2.0, ['Y']], equal_var=False)
print(f"T-value = {tval[0]:.2f}", f"P-value = {pval[0]}")

# Answer: Since the result 0.36 > 0.05, there is no confidencial conclusion that the degree of diabetes progression is different between men and women.