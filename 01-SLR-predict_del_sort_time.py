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

# Get the dataset into environment
del_sort = pd.read_csv("D:\Data Science - ExcelR\Assignments\Simple Linear Regression\delivery_time.csv")
del_sort.info()

# Rename the column names
del_sort.rename(columns={'Delivery Time':'Delivery_time'},inplace=True)
del_sort.rename(columns={'Sorting Time':'Sorting_time'},inplace=True)
del_sort.describe()

# getting the shape of the dataset
del_sort.shape

# Measures of Central Tendency
np.mean(del_sort)
np.median(del_sort.Delivery_time)
np.median(del_sort.Sorting_time)

# Measures of Dispersion
np.var(del_sort)
np.std(del_sort)

# Skewness and Kurtosis

skew(del_sort.Sorting_time)
skew(del_sort.Delivery_time)

kurtosis(del_sort.Sorting_time)
kurtosis(del_sort.Delivery_time)


x = np.array(del_sort.Sorting_time)
y = np.array(del_sort.Delivery_time)

# Normal Q-Q plot

plt.plot(del_sort);plt.legend(['Delivery_time','Sorting_time']);

stats.probplot(x,dist='norm',plot=pylab)
stats.probplot(y,dist='norm',plot=pylab)

# Normal Probability Distribution 

x1 = np.linspace(np.min(x),np.max(x))
y1 = stats.norm.pdf(x1,np.mean(x),np.std(y))

plt.plot(x1,y1,color='red');plt.xlim(np.min(x),np.max(x));plt.xlabel('Sorting_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

x2 = np.linspace(np.min(y),np.max(y))
y2 = stats.norm.pdf(x2,np.mean(y),np.std(y))

plt.plot(x2,y2,color='blue');plt.xlim(np.min(y),np.max(y));plt.xlabel('Delivery_Time');plt.ylabel('Probability_Distribution');plt.title('Normal Probability Distribution')

# Histogram
plt.hist(del_sort['Sorting_time'],color='coral')
plt.hist(del_sort['Delivery_time'],color = 'skyblue')

# Boxplot 
sns.boxplot(del_sort,orient='v')
sns.boxplot(del_sort['Sorting_time'],orient = 'v',color='coral')
sns.boxplot(del_sort['Delivery_time'],orient = 'v',color='yellow')

sns.pairplot(del_sort)
sns.countplot(del_sort['Sorting_time'])
sns.countplot(del_sort['Delivery_time'])

# Scatter plot

plt.scatter(x,y,label='Scatter Plot',color='coral',s=20);plt.xlabel('Sorting_Time');plt.ylabel('Delivery_Time');plt.title('Scatter Plot')

# Correlation coefficient(r)
np.corrcoef(del_sort['Sorting_time'],del_sort['Delivery_time'])
del_sort.corr()

sns.heatmap(del_sort.corr())

# Build the Simple Linear Regression Model1
model1 = smf.ols('Delivery_time~Sorting_time',data=del_sort).fit()
model1.summary()  # R^2 =0.682 < 0.80
model1.params

pred1 = model1.predict(del_sort)
error1 = del_sort.Delivery_time-pred1
# Sum of errors should be equal to zero
sum(error1)

# Scatter plot between X and Y
plt.scatter(x,y,color='red');plt.plot(x,pred1,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
# Correlation Coefficients(r)
np.corrcoef(x,y) #  r = 0.826 < 0.85

# RMSE = Root mean square of sum of Errors 
np.sqrt(np.mean(error1**2))  # RMSE = 2.79165

# Build the Simple Linear Regression Model2, Apply Log transformation to X- variable
model2 = smf.ols('Delivery_time~np.log(Sorting_time)',data=del_sort).fit()
model2.summary() # R^2 = 0.695 < 0.80
model2.params

pred2 = model2.predict(del_sort)
error2 = del_sort.Delivery_time-pred2
# Sum of errors should be equal to zero
sum(error2)

# Scatter plot between log(X) and Y
plt.scatter(np.log(x),y,color='red');plt.plot(np.log(x),pred2,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')
help(plt.plot)

# Correlation Coefficients(r)
np.corrcoef(np.log(x),y) # r=0.833 <0.85

# RMSE
np.sqrt(np.mean(error2**2))  # RMSE = 2.733

# Build the Simple Linear Regression Model3 , Apply Log transformation on 'Y'

model3 = smf.ols('np.log(del_sort.Delivery_time)~Sorting_time',data=del_sort).fit()
model3.summary() # R^2 = 0.711 < 0.80
model3.params

pred3 = model3.predict(del_sort)
error3 = del_sort.Delivery_time-np.exp(pred3)

# Sum of Errors should be equal to zero
sum(error3)

# Scatter plot between X and log(Y)
plt.scatter(x,np.log(y),color='red');plt.plot(x,pred3,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time');plt.title('Scatter Plot')

# Correlation Coefficients (r)
np.corrcoef(x,np.log(y)) # r = 0.8431 < 0.85

# RMSE 
np.sqrt(np.mean(error3**2)) #RMSE = 2.940
