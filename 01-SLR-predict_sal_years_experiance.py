# import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Getting dataset into environment
year_sal = pd.read_csv("D:\Data Science - ExcelR\Assignments\Simple Linear Regression\Salary_Data.csv")
year_sal.info()

# Satistic summary of the dataset
year_sal.describe()
# Shape of the Dataset(rows * columns)
year_sal.shape


# Measures of central tendency
np.mean(year_sal)
np.median(year_sal.YearsExperience)
np.median(year_sal.Salary)

# Measures of Dispersion
np.var(year_sal.YearsExperience)
np.var(year_sal.Salary)
np.std(year_sal)

# Skewness and Kurtosis
import scipy.stats as stats
from scipy.stats import skew, kurtosis

skew(year_sal)
kurtosis(year_sal)

x = np.array(year_sal.YearsExperience)
y = np.array(year_sal.Salary)

# Normal Q-Q plot

import pylab
plt.plot(year_sal);plt.legend(['Years_Experiance','Salary_hike']);plt.show()

stats.probplot(x,dist='norm',plot=pylab)
stats.probplot(y,dist='norm',plot=pylab)

# Normal Probalility Distribution plot

x1 = np.linspace(np.min(x),np.max(x))
y1 = stats.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlabel('Years_Experiance');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution');
help(plt.plot)

x2 = np.linspace(np.min(y),np.max(y))
y2 = stats.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlabel('Salary_hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution');

# Histogram 
plt.hist(year_sal['YearsExperience'],color='coral')
plt.hist(year_sal['Salary'],color='skyblue')

#Boxplot

sns.boxplot(year_sal,orient='v')
sns.boxplot(year_sal['YearsExperience'],orient='v',color='orange')
sns.boxplot(year_sal['Salary'],orient='v',color='yellow')

sns.pairplot(year_sal)
sns.countplot(year_sal['YearsExperience'])
sns.countplot(year_sal['Salary'])

# Scatter plot
plt.scatter(x,y,color='red',s=20);plt.xlabel('Years_Experiance');plt.ylabel('Salary_hike');plt.title('Scatter plot')

# Correlation coefficient
np.corrcoef(year_sal['YearsExperience'],year_sal['Salary'])
year_sal.corr()
sns.heatmap(year_sal.corr())

# Build the Simple Linear Regression Model1
model1 = smf.ols('Salary~YearsExperience',data=year_sal).fit()
model1.summary()  # R^2 =0.957 > 0.80
model1.params

pred1 = model1.predict(year_sal)
error1 = year_sal.Salary-pred1
# Sum of errors should be equal to zero
sum(error1)

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_Experiance');plt.title('Scatter Plot')
# Correlation Coefficients(r)
np.corrcoef(x,y) #  r = 0.978 > 0.85

# RMSE = Root mean square of sum of Errors 
np.sqrt(np.mean(error1**2))  # RMSE = 5592,044

# Build the Simple Linear Regression Model2, Apply Log transformation to X- variable
model2 = smf.ols('Salary~np.log(YearsExperience)',data=year_sal).fit()
model2.summary() # R^2 = 0.854 > 0.80
model2.params

pred2 = model2.predict(year_sal)
error2 = year_sal.Salary-pred2
# Sum of errors should be equal to zero
sum(error2)

# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_Experiance');plt.title('Scatter Plot')
help(plt.plot)

# Correlation Coefficients(r)
np.corrcoef(np.log(x),y) # r= 0.924 > 0.85

# RMSE
np.sqrt(np.mean(error2**2))  # RMSE = 10302.8937

# Build the Simple Linear Regression Model3 , Apply Log transformation on 'Y'

model3 = smf.ols('np.log(Salary)~YearsExperience',data=year_sal).fit()
model3.summary() # R^2 = 0.932 > 0.80
model3.params

pred3 = model3.predict(year_sal)
error3 = year_sal.Salary-np.exp(pred3)

# Sum of Errors should be equal to zero
sum(error3)

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred3,color='black');plt.xlabel('Salary_hike');plt.ylabel('Years_Experiance');plt.title('Scatter Plot')

# Correlation Coefficients (r)
np.corrcoef(x,np.log(y)) # r = 0.9654 > 0.85

# RMSE 
np.sqrt(np.mean(error3**2)) #RMSE = 7213.235
