# Import the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import pylab

%matplotlib inline
plt.style.use('ggplot')
sns.set()

# Getting the dataset into environment
wt_cal = pd.read_csv("D:\Data Science - ExcelR\Assignments\Simple Linear Regression\calories_consumed.csv")
wt_cal.info()
# Change the Column names
wt_cal.rename(columns={'Weight gained (grams)':'weight_gained'},inplace =True)
wt_cal.rename(columns={'Calories Consumed':'calories_consumed'},inplace=True)
wt_cal.describe()
wt_cal.shape

# Measures of Central Tendency
np.mean(wt_cal)
np.median(wt_cal.weight_gained)
np.median(wt_cal.calories_consumed)

# Measures of Dispersion
np.var(wt_cal)
np.std(wt_cal)
# Skewness
skew(wt_cal)
# skew(df.Weight)
# skew(df.Calories)
    
# Kurtosis
kurtosis(wt_cal)
# kurtosis(df.Weight)
# kurtosis(df.Calories)

x = np.array(wt_cal.calories_consumed)
y = np.array(wt_cal.weight_gained)

# Normal Q-Q plot
plt.plot(wt_cal);plt.legend(['Calories_consumed','Weight_gained'])

stats.probplot(x,dist='norm',plot=pylab)
stats.probplot(y,dist='norm',plot=pylab)

#Normal Probability Distribution

x1 = np.linspace(np.min(x),np.max(x))
y1 = stats.norm.pdf(x1,np.mean(x),np.std(x))
#y1 = stats.norm.pdf(x1,np.mean(wt_cal.calories_consumed),np.std(wt_cal.calories_consumed))

plt.plot(x1,y1,color='coral');plt.xlim(np.min(x),np.max(x));plt.xlabel('Calories_Consumed');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = stats.norm.pdf(x2,np.mean(y),np.std(y))
#y2 = stats.norm.pdf(x2,np.mean(df.Weight),np.std(df.Weight))
plt.plot(x2,y2,color = 'orange');plt.xlim(np.min(y),np.max(y)) ;plt.xlabel('Weight_gained(grams)');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram 
plt.hist(wt_cal['calories_consumed'])
plt.hist(wt_cal['weight_gained'])

# Boxplot
sns.boxplot(wt_cal,color='coral',orient='v')
sns.boxplot(wt_cal['calories_consumed'],orient='v',color='red') # orient = 'v' -> Vertival
sns.boxplot(wt_cal['weight_gained'],orient='v',color='yellow')  # orient = 'h' ->horizontal
help(sns.boxplot)

# Scatterplot

plt.scatter(x,y,label='Scatter_plot',color='r',s=20);plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained');plt.title('Scatter Plot ');

# Correlation Coefficient (r)
np.corrcoef(x,y) #np.corrcoef(x,y)
wt_cal.corr()
sns.heatmap(wt_cal.corr())

sns.pairplot(wt_cal)
sns.countplot(x)
sns.countplot(y)


# Build the simple Regression model
model1 = smf.ols('weight_gained~calories_consumed',data=wt_cal).fit()
model1.summary()  # R^2 =0.897 >0.80
model1.params

pred1 = model1.predict(wt_cal)
error1 = wt_cal.weight_gained-pred1
# Sum of Errors should be Zero
sum(error1)  

# Scatter plot between 'x' and 'y'
plt.scatter(x,y,color='red');plt.plot(x,pred1,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation Coefficients
np.corrcoef(x,y) # r = 0.94699 > 0.85

# RMSE -> Root mean square sum of errors

np.sqrt(np.mean(error1**2)) #RMSE value = 103.30

# Build the simple Regression model2 , Apply log transformation on x-variables
model2 = smf.ols('weight_gained~np.log(calories_consumed)',data=wt_cal).fit()
model2.summary()  # R^2 = 0.80 = 0.80
model2.params

pred2 = model2.predict(wt_cal)
error2 = wt_cal.weight_gained-pred2

# Sum of Error should be zero
sum(error2)

# Scatter Plot between log(x) and y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred2,color='black');plt.xlabel('log(Calories_Consumed)');plt.ylabel('Weight_gained(grams)');plt.title('Scatter Plot')
# Correlation coefficient (r)
np.corrcoef(np.log(x),y)  # r = 0.8987 > 0.85
# RMSE = Root mean square of sum of errors 
np.sqrt(np.mean(error2**2)) #RMSE value = 141.0054

# Build the simple Regression Model3 , Apply log transformation to Y-variable

model3 = smf.ols('np.log(weight_gained)~calories_consumed',data=wt_cal).fit()
model3.summary()  # R^2 =0.878 > 0.80
model3.params

pred3 = model3.predict(wt_cal)
error3 = wt_cal.weight_gained-np.exp(pred3)
# Sum of Errors should be Zero
sum(error3)  # 73.78

# Scatter Plot between X and log(Y)

plt.scatter(x,np.log(y),color='red');plt.plot(x,pred3,color='black');plt.xlabel('Calories_Consumed');plt.ylabel('log(Weight_gained(grams))');plt.title('Scatter Plot')
# Correlation Coefficient
np.corrcoef(x,np.log(y)) # 0.9368 > 0.85
# RMSE = Root mean square of sum of errors
np.sqrt(np.mean(error3**2)) #RMSE value = 118.045

