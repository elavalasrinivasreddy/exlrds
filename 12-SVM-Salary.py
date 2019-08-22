# reset the console
%reset -f

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset_train = pd.read_csv('SalaryData_Train.csv')
dataset_train.head()
dataset_train.columns
dataset_train.shape #(30161, 14)
dataset_train.info()
dataset_train.dtypes

dataset_test = pd.read_csv('SalaryData_Test.csv')
dataset_test.head()
dataset_test.columns
dataset_test.shape #(15060, 14)
dataset_test.info()
dataset_test.dtypes

# Missing values
dataset_train.isnull().sum() # No missing values
dataset_train.drop_duplicates(keep='first', inplace=True) # 3258 rows are duplicates

dataset_test.isnull().sum() # No missing values
dataset_test.drop_duplicates(keep='first', inplace=True) # 930 rows are duplicates

# splite the data by type
train_num = dataset_train.select_dtypes(include='int64')
train_num.shape # (26903, 5)

train_cat = dataset_train.select_dtypes(include='object')
train_cat.shape # (26903, 9)

test_num = dataset_test.select_dtypes(include='int64')
test_num.shape # (14130, 5)

test_cat = dataset_test.select_dtypes(include='object')
test_cat.shape # (14130, 9)

# Exploratory Data Analysis is done only on train data
# Statistical Description
dataset_train.describe()
# Measures of Dispersion
np.var(dataset_train)
np.std(dataset_train)
# Skewnes ans Kurtosis
from scipy.stats import skew, kurtosis
skew(train_num)
kurtosis(train_num)

# Histograms
plt.hist(dataset_train['age']);plt.title('Histogram of Age');plt.xlabel('Age');plt.ylabel('Frequency')
plt.hist(dataset_train['educationno'], color='coral');plt.title('Histogram of Education');plt.xlabel('Education');plt.ylabel('Frequency')
plt.hist(dataset_train['capitalgain'], color='teal');plt.title('Histogram of Capital Gain');plt.xlabel('Capital Gain');plt.ylabel('Frequency')
plt.hist(dataset_train['capitalloss'], color='brown');plt.title('Histogram of Capital Loss');plt.xlabel('Capital Loss');plt.ylabel('Frequency')
plt.hist(dataset_train['hoursperweek'], color='purple');plt.title('Histogram of Hours');plt.xlabel('Hours per Week');plt.ylabel('Frequency')

# Count plot for categorical data
import seaborn as sns
sns.countplot(dataset_train['workclass']).set_title('Countplot of Work Class')
pd.crosstab(dataset_train.workclass,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['education']).set_title('Countplot of Education')
pd.crosstab(dataset_train.education,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['maritalstatus']).set_title('Countplot of Marital Status')
pd.crosstab(dataset_train.maritalstatus,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['occupation']).set_title('Countplot of Occupation')
pd.crosstab(dataset_train.occupation,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['relationship']).set_title('Countplot of RelationShip')
pd.crosstab(dataset_train.relationship,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['race']).set_title('Countplot of Race')
pd.crosstab(dataset_train.race,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['sex']).set_title('Countplot of Sex')
pd.crosstab(dataset_train.sex,dataset_train.Salary).plot(kind="bar")

sns.countplot(dataset_train['native']).set_title('Countplot of Native')
pd.crosstab(dataset_train.native,dataset_train.Salary).plot(kind="bar")

# Normal Q-Q plot 
plt.plot(train_num);plt.legend(list(train_num))

age = np.array(dataset_train['age'])
edu = np.array(dataset_train['educationno'])
capgain = np.array(dataset_train['capitalgain'])
caploss = np.array(dataset_train['capitalloss'])
hpw = np.array(dataset_train['hoursperweek'])

from scipy import stats
stats.probplot(age, dist='norm', plot=plt);plt.title('Q-Q plot of Age')
stats.probplot(edu, dist='norm', plot=plt);plt.title('Q-Q plot of Educationno')
stats.probplot(capgain, dist='norm', plot=plt);plt.title('Q-Q plot of Capital Gain')
stats.probplot(caploss, dist='norm', plot=plt);plt.title('Q-Q plot of Capital Loss')
stats.probplot(hpw, dist='norm', plot=plt);plt.title('Q-Q plot of Hours Per Week')

# Normal Probability Distribution
from scipy import stats
x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.median(x_age), np.std(x_age))
plt.plot(x_age, y_age);plt.xlim(np.min(age), np.max(age));plt.title('Normal Probability Distribution of Age');plt.xlabel('Age');plt.ylabel('Probability')

x_edu = np.linspace(np.min(edu), np.max(edu))
y_edu = stats.norm.pdf(x_edu, np.median(x_edu), np.std(x_edu))
plt.plot(x_edu, y_edu);plt.xlim(np.min(edu), np.max(edu));plt.title('Normal Probability Distribution of Educationno');plt.xlabel('Educationno');plt.ylabel('Probability')

x_capgain = np.linspace(np.min(capgain), np.max(capgain))
y_capgain = stats.norm.pdf(x_capgain, np.median(x_capgain), np.std(x_capgain))
plt.plot(x_capgain, y_capgain);plt.xlim(np.min(capgain), np.max(capgain));plt.title('Normal Probability Distribution of Capital Gain');plt.xlabel('Capital Gain');plt.ylabel('Probability')

x_caploss = np.linspace(np.min(caploss), np.max(caploss))
y_caploss = stats.norm.pdf(x_caploss, np.median(x_caploss), np.std(x_caploss))
plt.plot(x_caploss, y_caploss);plt.xlim(np.min(caploss), np.max(caploss));plt.title('Normal Probability Distribution of Capital Loss');plt.xlabel('Capital Loss');plt.ylabel('Probability')

x_hpw = np.linspace(np.min(hpw), np.max(hpw))
y_hpw = stats.norm.pdf(x_hpw, np.median(x_hpw), np.std(x_hpw))
plt.plot(x_hpw, y_hpw);plt.xlim(np.min(hpw), np.max(hpw));plt.title('Normal Probability Distribution of Hours per Week');plt.xlabel('Hours per Week');plt.ylabel('Probability')

# Boxplot 
sns.boxplot(dataset_train['age'],orient='v', color='orange').set_title('Boxplot of Age')
sns.boxplot(dataset_train['educationno'],orient='v', color='teal').set_title('Boxplot of Age')
sns.boxplot(dataset_train['capitalgain'],orient='v').set_title('Boxplot of Age')
sns.boxplot(dataset_train['capitalloss'],orient='v').set_title('Boxplot of Age')
sns.boxplot(dataset_train['hoursperweek'],orient='v').set_title('Boxplot of Hours per Week')

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sns.boxplot(x="workclass",y="age",data=dataset_train).set_title('Boxplot of WorkClass & Age')
sns.boxplot(x="workclass",y="educationno",data=dataset_train).set_title('Boxplot of WorkClass & Educationno')
sns.boxplot(x="workclass",y="capitalgain",data=dataset_train).set_title('Boxplot of WorkClass & CapitalGain')
sns.boxplot(x="workclass",y="capitalloss",data=dataset_train).set_title('Boxplot of WorkClass & CapitalLoss')
sns.boxplot(x="workclass",y="hoursperweek",data=dataset_train).set_title('Boxplot of WorkClass & HoursPerWeek')

sns.boxplot(x="education",y="age",data=dataset_train).set_title('Boxplot of Education & Age')
sns.boxplot(x="education",y="educationno",data=dataset_train).set_title('Boxplot of Education & Educationno')
sns.boxplot(x="education",y="capitalgain",data=dataset_train).set_title('Boxplot of Education & CapitalGain')
sns.boxplot(x="education",y="capitalloss",data=dataset_train).set_title('Boxplot of Education & CapitalLoss')
sns.boxplot(x="education",y="hoursperweek",data=dataset_train).set_title('Boxplot of Education & HoursPerWeek')

sns.boxplot(x="maritalstatus",y="age",data=dataset_train).set_title('Boxplot of MaritalStatus & Age')
sns.boxplot(x="maritalstatus",y="educationno",data=dataset_train).set_title('Boxplot of MaritalStatus & Educationno')
sns.boxplot(x="maritalstatus",y="capitalgain",data=dataset_train).set_title('Boxplot of MaritalStatus & CapitalGain')
sns.boxplot(x="maritalstatus",y="capitalloss",data=dataset_train).set_title('Boxplot of MaritalStatus & CapitalLoss')
sns.boxplot(x="maritalstatus",y="hoursperweek",data=dataset_train).set_title('Boxplot of MaritalStatus & HoursPerWeek')

sns.boxplot(x="occupation",y="age",data=dataset_train).set_title('Boxplot of Occupation & Age')
sns.boxplot(x="occupation",y="educationno",data=dataset_train).set_title('Boxplot of Occupation & Educationno')
sns.boxplot(x="occupation",y="capitalgain",data=dataset_train).set_title('Boxplot of Occupation & CapitalGain')
sns.boxplot(x="occupation",y="capitalloss",data=dataset_train).set_title('Boxplot of Occupation & CapitalLoss')
sns.boxplot(x="occupation",y="hoursperweek",data=dataset_train).set_title('Boxplot of Occupation & HoursPerWeek')

sns.boxplot(x="relationship",y="age",data=dataset_train).set_title('Boxplot of RelationShip & Age')
sns.boxplot(x="relationship",y="educationno",data=dataset_train).set_title('Boxplot of RelationShip & Educationno')
sns.boxplot(x="relationship",y="capitalgain",data=dataset_train).set_title('Boxplot of RelationShip & CapitalGain')
sns.boxplot(x="relationship",y="capitalloss",data=dataset_train).set_title('Boxplot of RelationShip & CapitalLoss')
sns.boxplot(x="relationship",y="hoursperweek",data=dataset_train).set_title('Boxplot of RelationShip & HoursPerWeek')

sns.boxplot(x="race",y="age",data=dataset_train).set_title('Boxplot of Race & Age')
sns.boxplot(x="race",y="educationno",data=dataset_train).set_title('Boxplot of Race & Educationno')
sns.boxplot(x="race",y="capitalgain",data=dataset_train).set_title('Boxplot of Race & CapitalGain')
sns.boxplot(x="race",y="capitalloss",data=dataset_train).set_title('Boxplot of Race & CapitalLoss')
sns.boxplot(x="race",y="hoursperweek",data=dataset_train).set_title('Boxplot of Race & HoursPerWeek')

sns.boxplot(x="sex",y="age",data=dataset_train).set_title('Boxplot of Sex & Age')
sns.boxplot(x="sex",y="educationno",data=dataset_train).set_title('Boxplot of Sex & Educationno')
sns.boxplot(x="sex",y="capitalgain",data=dataset_train).set_title('Boxplot of Sex & CapitalGain')
sns.boxplot(x="sex",y="capitalloss",data=dataset_train).set_title('Boxplot of Sex & CapitalLoss')
sns.boxplot(x="sex",y="hoursperweek",data=dataset_train).set_title('Boxplot of Sex & HoursPerWeek')

sns.boxplot(x="native",y="age",data=dataset_train).set_title('Boxplot of Native & Age')
sns.boxplot(x="native",y="educationno",data=dataset_train).set_title('Boxplot of Native & Educationno')
sns.boxplot(x="native",y="capitalgain",data=dataset_train).set_title('Boxplot of Native & CapitalGain')
sns.boxplot(x="native",y="capitalloss",data=dataset_train).set_title('Boxplot of Native & CapitalLoss')
sns.boxplot(x="native",y="hoursperweek",data=dataset_train).set_title('Boxplot of Native & HoursPerWeek')

# Scatterplot 
sns.scatterplot(x='age', y='educationno', data=dataset_train).set_title('Scatterplot Age & Educationno')
sns.scatterplot(x='age', y='capitalgain', data=dataset_train).set_title('Scatterplot Age & Capital Gain')
sns.scatterplot(x='age', y='capitalloss', data=dataset_train).set_title('Scatterplot Age & Capital Loss')
sns.scatterplot(x='age', y='hoursperweek', data=dataset_train).set_title('Scatterplot Age & Hours Per Week')

sns.scatterplot(x='educationno', y='capitalgain', data=dataset_train).set_title('Scatterplot Educationno & Capital Gain')
sns.scatterplot(x='educationno', y='capitalloss', data=dataset_train).set_title('Scatterplot Educationno & Capital Loss')
sns.scatterplot(x='educationno', y='hoursperweek', data=dataset_train).set_title('Scatterplot Educationno & Hours Per Week')

sns.scatterplot(x='capitalgain', y='capitalloss', data=dataset_train).set_title('Scatterplot Capital Gain & Capital Loss')
sns.scatterplot(x='capitalgain', y='hoursperweek', data=dataset_train).set_title('Scatterplot Capital Gain & Hours Per Week')

sns.scatterplot(x='capitalloss', y='hoursperweek', data=dataset_train).set_title('Scatterplot Capital Loss & Hours Per Week')

sns.pairplot(dataset_train)
sns.pairplot(dataset_train, diag_kind = 'kde')
sns.pairplot(dataset_train, hue='Salary')

# Heatmap
corr = dataset_train.corr()
print(corr)
sns.heatmap(corr, annot=True)

# Encoding the Categorical data
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in string_columns:
    dataset_train[i] = labelencoder.fit_transform(dataset_train[i])
    dataset_test[i] = labelencoder.fit_transform(dataset_test[i])

# Outliers
from scipy import stats
Z = np.abs(stats.zscore(dataset_train.iloc[:,0:-1]))
print(np.where(Z>3))
print(Z[11][7])
df_out = dataset_train[(Z<3).all(axis=1)] # 4062 outliers are removed

# Splitting the dataset
colnames = dataset_train.columns
len(colnames[0:13])
X_train = df_out[colnames[0:13]]
Y_train = df_out[colnames[13]]
X_test  = dataset_test[colnames[0:13]]
Y_test  = dataset_test[colnames[13]]

df_out.Salary.value_counts() # classes are imbalanced

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Fitting the SVM classifier
from sklearn.svm import SVC
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

'''################### kernel = linear #########################'''

linear_model = SVC(kernel = "linear", random_state=0)
linear_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
linear_train_pred = linear_model.predict(X_train)
linear_train_cm = confusion_matrix(Y_train, linear_train_pred)
print(linear_train_cm)
linear_train_acc = accuracy_score(Y_train, linear_train_pred)
print(linear_train_acc) # 80%

# Predicting results on test-set data
linear_test_pred = linear_model.predict(X_test)
linear_test_cm = confusion_matrix(Y_test, linear_test_pred)
print(linear_test_cm)
linear_test_acc = accuracy_score(Y_test, linear_test_pred)
print(linear_test_acc) # 79%

'''#################### kernel = poly ############################'''

poly_model = SVC(kernel = "poly", random_state=0)
poly_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
poly_train_pred = poly_model.predict(X_train)
poly_train_cm = confusion_matrix(Y_train, poly_train_pred)
print(poly_train_cm)
poly_train_acc = accuracy_score(Y_train, poly_train_pred)
print(poly_train_acc) # 84%

# Predicting results on test-set data
poly_test_pred = poly_model.predict(X_test)
poly_test_cm = confusion_matrix(Y_test, poly_test_pred)
print(poly_test_cm)
poly_test_acc = accuracy_score(Y_test, poly_test_pred)
print(poly_test_acc) # 82%

'''################## kernel = rbf ###########################'''

rbf_model = SVC(kernel = "rbf", random_state=0)
rbf_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
rbf_train_pred = rbf_model.predict(X_train)
rbf_train_cm = confusion_matrix(Y_train, rbf_train_pred)
print(rbf_train_cm)
rbf_train_acc = accuracy_score(Y_train, rbf_train_pred)
print(rbf_train_acc) # 84%

# Predicting results on test-set data
rbf_test_pred = rbf_model.predict(X_test)
rbf_test_cm = confusion_matrix(Y_test, rbf_test_pred)
print(rbf_test_cm)
rbf_test_acc = accuracy_score(Y_test, rbf_test_pred)
print(rbf_test_acc) # 81%

'''################## kernel = sigmoid ###########################'''

sigmoid_model = SVC(kernel = "sigmoid", random_state=0)
sigmoid_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
sigmoid_train_pred = sigmoid_model.predict(X_train)
sigmoid_train_cm = confusion_matrix(Y_train, sigmoid_train_pred)
print(sigmoid_train_cm)
sigmoid_train_acc = accuracy_score(Y_train, sigmoid_train_pred)
print(sigmoid_train_acc) # 74%

# Predicting results on test-set data
sigmoid_test_pred = sigmoid_model.predict(X_test)
sigmoid_test_cm = confusion_matrix(Y_test, sigmoid_test_pred)
print(sigmoid_test_cm)
sigmoid_test_acc = accuracy_score(Y_test, sigmoid_test_pred)
print(sigmoid_test_acc) # 72%

''' After Applying stratified KFold we get same above accuracy results on each kernel '''

# Stratified Method
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
metric_names = ['f1', 'roc_auc', 'average_precision', 'accuracy', 'precision', 'recall']
scores_df = pd.DataFrame(index=metric_names, columns=['Random-CV', 'Stratified-CV']) # to store the Scores
cv = KFold(n_splits=3)
scv = StratifiedKFold(n_splits=3)

Y_train = Y_train.astype('category')
Y_train = Y_train.cat.codes
Y_test = Y_test.astype('category')
Y_test = Y_test.cat.codes

'''################### kernel = linear #########################'''

st_linear_model = SVC(kernel = "linear", random_state=0)
st_linear_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
st_linear_train_pred = st_linear_model.predict(X_train)
st_linear_train_cm = confusion_matrix(Y_train, st_linear_train_pred)
print(st_linear_train_cm)
st_linear_train_acc = accuracy_score(Y_train, st_linear_train_pred)
print(st_linear_train_acc) # 80%

# Predicting results on test-set data
st_linear_test_pred = st_linear_model.predict(X_test)
st_linear_test_cm = confusion_matrix(Y_test, st_linear_test_pred)
print(st_linear_test_cm)
st_linear_test_acc = accuracy_score(Y_test, st_linear_test_pred)
print(st_linear_test_acc) # 79%

'''#################### kernel = poly ############################'''

st_poly_model = SVC(kernel = "poly", random_state=0)
st_poly_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
st_poly_train_pred = st_poly_model.predict(X_train)
st_poly_train_cm = confusion_matrix(Y_train, st_poly_train_pred)
print(st_poly_train_cm)
st_poly_train_acc = accuracy_score(Y_train, st_poly_train_pred)
print(st_poly_train_acc) # 84%

# Predicting results on test-set data
st_poly_test_pred = st_poly_model.predict(X_test)
st_poly_test_cm = confusion_matrix(Y_test, st_poly_test_pred)
print(st_poly_test_cm)
st_poly_test_acc = accuracy_score(Y_test, st_poly_test_pred)
print(st_poly_test_acc) # 82%

'''################## kernel = rbf ###########################'''

st_rbf_model = SVC(kernel = "rbf", random_state=0)
st_rbf_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
st_rbf_train_pred = st_rbf_model.predict(X_train)
st_rbf_train_cm = confusion_matrix(Y_train, st_rbf_train_pred)
print(st_rbf_train_cm)
st_rbf_train_acc = accuracy_score(Y_train, st_rbf_train_pred)
print(st_rbf_train_acc) # 84%

# Predicting results on test-set data
st_rbf_test_pred = st_rbf_model.predict(X_test)
st_rbf_test_cm = confusion_matrix(Y_test, st_rbf_test_pred)
print(st_rbf_test_cm)
st_rbf_test_acc = accuracy_score(Y_test, st_rbf_test_pred)
print(st_rbf_test_acc) # 81%

'''################## kernel = sigmoid ###########################'''

st_sigmoid_model = SVC(kernel = "sigmoid", random_state=0)
st_sigmoid_model.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting the results on train-set data
st_sigmoid_train_pred = st_sigmoid_model.predict(X_train)
st_sigmoid_train_cm = confusion_matrix(Y_train, st_sigmoid_train_pred)
print(st_sigmoid_train_cm)
st_sigmoid_train_acc = accuracy_score(Y_train, st_sigmoid_train_pred)
print(st_sigmoid_train_acc) # 74%

# Predicting results on test-set data
st_sigmoid_test_pred = st_sigmoid_model.predict(X_test)
st_sigmoid_test_cm = confusion_matrix(Y_test, st_sigmoid_test_pred)
print(st_sigmoid_test_cm)
st_sigmoid_test_acc = accuracy_score(Y_test, st_sigmoid_test_pred)
print(st_sigmoid_test_acc) # 72%
