import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

"""
AGE
SEX
BMI
BP
S1
S2
S3
S4
S5
S6
Y
"""

cols = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "y"]

df = pd.read_csv("./diabetes.tsv", sep="\t", header=0, dtype=float)
# df.head()

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
# Sex: Uniform
# BMI: Normal
# Y: Normal

# Task 4: Test the correlation between different variables and disease progression (Y)
# Hint Correlation matrix would give you the most useful information on which values are dependent.

# Task 5: Test the hypothesis that the degree of diabetes progression is different between men and women
