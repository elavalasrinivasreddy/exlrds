# import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import statsmodels.formula.api as smf
import pylab

%matplotlib inline
sns.set()
plt.style.use('ggplot')

# get the dataset into environment
sal_churn = pd.read_csv("D:\Data Science - ExcelR\Assignments\Simple Linear Regression\emp_data.csv")
sal_churn.info()
# Rename the column names

#sal_churn.rename(columns={'Salary_hike':'Salary_hike'},inplace=True)
#sal_churn.rename(columns={'Churan_out_rate':'Churn_rate'},inplace=True)
sal_churn.describe()

# getting the shape
sal_churn.shape

# Measures of Central Tendency
np.mean(sal_churn)
np.median(sal_churn.Salary_hike)
np.median(sal_churn.Churn_out_rate)

# Measures of Dispersion
np.var(sal_churn)
np.std(sal_churn)

# Skewness and Kurtosis
skew(sal_churn.Salary_hike)
skew(sal_churn.Churn_out_rate)

kurtosis(sal_churn.Salary_hike)
kurtosis(sal_churn.Churn_out_rate)


x = np.array(sal_churn.Salary_hike)
y = np.array(sal_churn.Churn_out_rate)

# Normal Q-Q plot
plt.plot(sal_churn.Salary_hike)
plt.plot(sal_churn.Churn_out_rate)

plt.plot(sal_churn);plt.legend(['Salary_hike','Churn_out_rate']);

stats.probplot(x,dist='norm',plot=pylab)
stats.probplot(y,dist='norm',plot=pylab)

# Normal Probability Distribution plot 

x1 = np.linspace(np.min(x),np.max(x))
y1 = stats.norm.pdf(x1,np.mean(x),np.std(x))
plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Salary_Hike');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = stats.norm.pdf(x2,np.mean(y),np.std(y))
plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Churn_out_rate');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(sal_churn['Salary_hike'],color='coral')

plt.hist(sal_churn['Churn_out_rate'],color='skyblue')

# Boxplot 
sns.boxplot(sal_churn,orient='v')
sns.boxplot(sal_churn['Salary_hike'],orient = 'v',color='coral')
sns.boxplot(sal_churn['Churn_out_rate'],orient = 'v',color='skyblue')

sns.pairplot(sal_churn)
sns.countplot(sal_churn['Salary_hike'])
sns.countplot(sal_churn['Churn_out_rate'])

# Scatter plot

plt.scatter(x,y,label='Scatter plot',color='coral',s=20);plt.xlabel('Salary_Hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot');

# Correlation coefficient(r)
np.corrcoef(sal_churn['Salary_hike'],sal_churn['Churn_out_rate']) #corrcoef(x,y)
sal_churn.corr()

sns.heatmap(sal_churn.corr())

# Build the Simple Linear Regression Model1
model1 = smf.ols('Churn_out_rate~Salary_hike',data=sal_churn).fit()
model1.summary()  # R^2 =0.831 > 0.80
model1.params

pred1 = model1.predict(sal_churn)
error1 = sal_churn.Churn_out_rate-pred1
# Sum of errors should be equal to zero
sum(error1)

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred1,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
# Correlation Coefficients(r)
np.corrcoef(x,y) #  r = -0.918 < 0.85

# RMSE = Root mean square of sum of Errors 
np.sqrt(np.mean(error1**2))  # RMSE = 3.9975

# Build the Simple Linear Regression Model2, Apply Log transformation to X- variable
model2 = smf.ols('Churn_out_rate~np.log(Salary_hike)',data=sal_churn).fit()
model2.summary() # R^2 = 0.849 > 0.80
model2.params

pred2 = model2.predict(sal_churn)
error2 = sal_churn.Churn_out_rate-pred2
# Sum of errors should be equal to zero
sum(error2)

# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred2,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')
help(plt.plot)

# Correlation Coefficients(r)
np.corrcoef(np.log(x),y) # r=-0.921 < 0.85

# RMSE
np.sqrt(np.mean(error2**2))  # RMSE = 3.786

# Build the Simple Linear Regression Model3 , Apply Log transformation on 'Y'

model3 = smf.ols('np.log(Churn_out_rate)~Salary_hike',data=sal_churn).fit()
model3.summary() # R^2 = 0.874 > 0.80
model3.params

pred3 = model3.predict(sal_churn)
error3 = sal_churn.Churn_out_rate-np.exp(pred3)

# Sum of Errors should be equal to zero
sum(error3)

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred3,color='black');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate');plt.title('Scatter Plot')

# Correlation Coefficients (r)
np.corrcoef(x,np.log(y)) # r = -0.9346 < 0.85

# RMSE 
np.sqrt(np.mean(error3**2)) #RMSE = 3.5415
