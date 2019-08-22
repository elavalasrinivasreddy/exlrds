# reset the console
%reset -f

# Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset = pd.read_csv('Fraud_check.csv')
dataset.head()
dataset.dtypes
dataset.isnull().sum() # No missing values
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # No duplicates
dataset.columns

# Statistical Description
dataset.describe()  # Only 3 numerical columns

num_data = dataset.select_dtypes(include='int64')

# Measures of Dispersion
np.var(num_data)
np.std(num_data)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(num_data)
kurtosis(num_data)

# Histogram 
plt.hist(dataset['Taxable.Income']);plt.title('Histogram of Taxable Income');plt.xlabel('Taxable Income');plt.ylabel('Frequency')
plt.hist(dataset['City.Population'], color='coral');plt.title('Histogram of City Population');plt.xlabel('City Population');plt.ylabel('Frequency')
plt.hist(dataset['Work.Experience'], color='orange');plt.title('Histogram of Work Experience');plt.xlabel('Work Experience');plt.ylabel('Frequency')

# Barplot for categorical data
import seaborn as sns
sns.countplot(dataset['Undergrad']).set_title('Count of Undergrad')
sns.countplot(dataset['Marital.Status']).set_title('Count of Marital Status')
sns.countplot(dataset['Urban']).set_title('Count of Urban') # Both classes are equal

# Normal Q-Q plot
plt.plot(num_data);plt.legend(list(num_data))

tax = np.array(num_data['Taxable.Income'])
city = np.array(num_data['City.Population'])
work = np.array(num_data['Work.Experience'])

from scipy import stats
stats.probplot(tax, dist='norm', plot=plt);plt.title('Q-Q plot of Tax Income')
stats.probplot(city, dist='norm', plot=plt);plt.title('Q-Q plot of City Population')
stats.probplot(work, dist='norm', plot=plt);plt.title('Q-Q plot of Work Experience')

# Normal Probability Distribution
x_tax = np.linspace(np.min(tax), np.max(tax))
y_tax = stats.norm.pdf(x_tax, np.median(x_tax), np.std(x_tax))
plt.plot(x_tax, y_tax);plt.xlim(np.min(tax), np.max(tax));plt.title('Normal Probability Distribution of Tax Income');plt.xlabel('Tax Income');plt.ylabel('Probability')

x_city = np.linspace(np.min(city), np.max(city))
y_city = stats.norm.pdf(x_city, np.median(x_city), np.std(x_city))
plt.plot(x_city, y_city);plt.xlim(np.min(city), np.max(city));plt.title('Normal Probability Distribution of City Population');plt.xlabel('City Population');plt.ylabel('Probability')

x_work = np.linspace(np.min(work), np.max(work))
y_work = stats.norm.pdf(x_work, np.median(x_work), np.std(x_work))
plt.plot(x_work, y_work);plt.xlim(np.min(work), np.max(work));plt.title('Normal Probability Distribution of Work Experience');plt.xlabel('Work Experience');plt.ylabel('Probability')
# All are follows Normal Probability Distribution

#Boxplot 
sns.boxplot(num_data['Taxable.Income'], orient='v').set_title('Boxplot of Taxable Income')
sns.boxplot(num_data['City.Population'], orient='v').set_title('Boxplot of City Population')
sns.boxplot(num_data['Work.Experience'], orient='v').set_title('Boxplot of Work Experience')

# wrt to categorical data
sns.boxplot(x='Undergrad', y='Taxable.Income', data=dataset).set_title('Boxplot of Undergrad & Taxable Income')
sns.boxplot(x='Undergrad', y='City.Population', data=dataset).set_title('Boxplot of Undergrad & City Population')
sns.boxplot(x='Undergrad', y='Work.Experience', data=dataset).set_title('Boxplot of Undergrad & Work Experience')

sns.boxplot(x='Marital.Status', y='Taxable.Income', data=dataset).set_title('Boxplot of Marital Status & Taxable Income')
sns.boxplot(x='Marital.Status', y='City.Population', data=dataset).set_title('Boxplot of Marital Status & City Population')
sns.boxplot(x='Marital.Status', y='Work.Experience', data=dataset).set_title('Boxplot of Marital Status & Work Experience')

sns.boxplot(x='Urban', y='Taxable.Income', data=dataset).set_title('Boxplot of Urban & Taxable Income')
sns.boxplot(x='Urban', y='City.Population', data=dataset).set_title('Boxplot of Urban & City Population')
sns.boxplot(x='Urban', y='Work.Experience', data=dataset).set_title('Boxplot of Urban & Work Experience')

# Scatterplot
sns.scatterplot(x='Taxable.Income', y='City.Population', data=dataset).set_title('Scatterplot of Tax Income & City Population')
sns.scatterplot(x='Taxable.Income', y='Work.Experience', data=dataset).set_title('Scatterplot of Tax Income & Work Experience')
sns.scatterplot(x='City.Population', y='Work.Experience', data=dataset).set_title('Scatterplot of City Population & Work Experience')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot=True)

# Creating Dummy variables
cat_data = dataset.select_dtypes(include='object')
dum_data = pd.get_dummies(cat_data)

final_data = pd.concat([num_data, dum_data], axis=1)

# Outliers
from scipy import stats
Z = np.abs(stats.zscore(final_data))
print(np.where(Z>3))  # No outliers

# Treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
final_data['Taxable.Income'] = ['Risky' if x<=30000 else 'Good' for x in final_data['Taxable.Income']]
print('No.of Good transactions: {}'.format(sum(final_data['Taxable.Income']=='Good')))
print('No.of Risky transactions: {}'.format(sum(final_data['Taxable.Income']=='Risky')))

# Metrics of features
colnames = list(final_data.columns)
predictors = colnames[1:]
target = colnames[0]

# Splitting the dataset into trainset and testset
from sklearn.model_selection import train_test_split
train, test = train_test_split(final_data, test_size =0.30, random_state=0)

# fitting the Classifier on the data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(train[predictors], train[target])

# predicting the trainset results
pred_train = classifier.predict(train[predictors])
from sklearn.metrics import confusion_matrix, accuracy_score
cm_train = confusion_matrix(train[target], pred_train)
print(cm_train)
acc_train = accuracy_score(train[target], pred_train)
print(acc_train) # 100%

# Predicting the testset results
pred_test = classifier.predict(test[predictors])
cm_test = confusion_matrix(test[target],pred_test)
print(cm_test)
acc_test = accuracy_score(test[target],pred_test)
print(acc_test) # 65%

# Classes in target variable is imbalanced.
# Good : Risky = 4: 1 ratio

# Ensembling method
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create an object of the classifier
bbc = BalancedBaggingClassifier(base_estimator =DecisionTreeClassifier(criterion='entropy', random_state=0),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

Y_train = train['Taxable.Income']
X_train = train.drop(['Taxable.Income'], axis=1)
X_test = test.drop(['Taxable.Income'], axis=1)
Y_test = test['Taxable.Income']

# Train the classifier
bbc.fit(X_train, Y_train)
preds = bbc.predict(X_test)

pd.Series(preds).value_counts()
# Confusion matrix
pd.crosstab(Y_test,preds)
# Accuracy
np.mean(preds==Y_test) # 58% [# gini 57%]

# Cross validation K-fold
X = final_data.iloc[:,1:].values
Y = final_data.iloc[:,0].values

from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)

for train_index, test_index in kf.split(X):
    print('Train:', train_index, 'Test:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

# fitting the Classifier on the data
from sklearn.tree import DecisionTreeClassifier
k_model = DecisionTreeClassifier(criterion='entropy', random_state=0) # ['gini']
k_model.fit(X_train, Y_train)

# predicting the trainset results
pred_train = k_model.predict(X_train)
from sklearn.metrics import confusion_matrix, accuracy_score
cm_train = confusion_matrix(Y_train, pred_train)
print(cm_train)
acc_train = accuracy_score(Y_train, pred_train)
print(acc_train) # 100%

# Predicting the testset results
pred_test = k_model.predict(X_test)
cm_test = confusion_matrix(Y_test,pred_test)
print(cm_test)
acc_test = accuracy_score(Y_test,pred_test)
print(acc_test) # 61% [#gini 61%]

# SMOTE method 
# In this method it only allows the target variable in binary format
sm_dataset = final_data.copy()
sm_dataset['Taxable.Income'] = [1 if x =='Good' else 0 for x in sm_dataset['Taxable.Income']]
sm_dataset['Taxable.Income'].value_counts()
final_data['Taxable.Income'].value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sm_dataset.drop(['Taxable.Income'], axis=1), sm_dataset['Taxable.Income'], test_size = 0.3, random_state=0)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

print("Before oversampling, count of label '1' :{}".format(sum(pd.Series(Y_train)==1)))
print("Before oversampling, count of label '0' :{}".format(sum(pd.Series(Y_train)==0)))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train.ravel())

print("After oversampling, count of label '1' :{}".format(sum(pd.Series(Y_train_sm)==1)))
print("After oversampling, count of label '0' :{}".format(sum(pd.Series(Y_train_sm)==0)))

# fitting the Classifier on the data
from sklearn.tree import DecisionTreeClassifier
sm_model = DecisionTreeClassifier(criterion='entropy', random_state=0) # ['gini']
sm_model.fit(X_train_sm, Y_train_sm)

# predicting the trainset results
pred_train = sm_model.predict(X_train_sm)
from sklearn.metrics import confusion_matrix, accuracy_score
cm_train = confusion_matrix(Y_train_sm, pred_train)
print(cm_train)
acc_train = accuracy_score(Y_train_sm, pred_train)
print(acc_train) # 100%

# Predicting the testset results
pred_test = sm_model.predict(X_test)
cm_test = confusion_matrix(Y_test,pred_test)
print(cm_test)
acc_test = accuracy_score(Y_test,pred_test)
print(acc_test) # 64% [66%]

# StratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X,Y)
print(skf)

for train_index, test_index in skf.split(X,Y):
    print('Train:', train_index, 'Test:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
# fitting the Classifier on the data
from sklearn.tree import DecisionTreeClassifier
sfk_model = DecisionTreeClassifier(criterion='entropy', random_state=0) # ['gini']
sfk_model.fit(X_train, Y_train)

# predicting the trainset results
pred_train = sfk_model.predict(X_train)
from sklearn.metrics import confusion_matrix, accuracy_score
cm_train = confusion_matrix(Y_train, pred_train)
print(cm_train)
acc_train = accuracy_score(Y_train, pred_train)
print(acc_train) # 100%

# Predicting the testset results
pred_test = sfk_model.predict(X_test)
cm_test = confusion_matrix(Y_test,pred_test)
print(cm_test)
acc_test = accuracy_score(Y_test,pred_test)
print(acc_test) # 64% [61%]

# Visualization of decision tree
# SMOTE method is best among all
import graphviz
from sklearn import tree
feature_names = predictors.copy()
class_names = final_data['Taxable.Income'].unique()

dot_data = tree.export_graphviz(final_model, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,class_names=class_names)

graph = graphviz.Source(dot_data)
graph

# if not working graph
#import os
#os.environ["PATH"] += os.pathsep + 'C:\Users\elava\Anaconda3\pkgs\graphviz-2.38\bin'
# replace "C:\Users\elava\Anaconda3\pkgs\graphviz-2.38\bin" with your bin directory

''' Optional: I try to write own for loop '''

# Manually splitting the data
#total 600 Good Risky
#train 80% 380  100
#test  20% 96   24     = 120

train = []
test = []
count_good = 0
count_risky = 0
for i in final_data['Taxable.Income']:
    if i == 'Good':
        y = int(final_data[final_data['Taxable.Income']=='Good'].index[0])
        if count_good < 380:
            train.append(list(final_data.iloc[y,:]))
            count_good = count_good + 1
        else:
            test.append(list(final_data.iloc[y,:]))
    else:
        y = int(final_data[final_data['Taxable.Income']=='Risky'].index[0])
        if count_risky <100:
            train.append(list(final_data.iloc[y,:]))
            count_risky = count_risky + 1
        else:
            test.append(list(final_data.iloc[y,:]))

colnames = list(final_data.columns)
predictors = colnames[1:]
target = colnames[0]

train = pd.DataFrame(train, columns=colnames)
test = pd.DataFrame(test, columns=colnames)

train.shape
test.shape

# fitting the Classifier on the data
from sklearn.tree import DecisionTreeClassifier
final_model = DecisionTreeClassifier(criterion='entropy')
final_model.fit(train[predictors], train[target])

# predicting the trainset results
pred_train = final_model.predict(train[predictors])
from sklearn.metrics import confusion_matrix, accuracy_score
cm_train = confusion_matrix(train[target], pred_train)
print(cm_train)
acc_train = accuracy_score(train[target], pred_train)
print(acc_train) # 100%

# Predicting the testset results
pred_test = final_model.predict(test[predictors])
cm_test = confusion_matrix(test[target],pred_test)
print(cm_test)
acc_test = accuracy_score(test[target],pred_test)
print(acc_test) # 100%

