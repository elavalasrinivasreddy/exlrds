# Reset Console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
import statsmodels.api as sm
dataset = sm.datasets.get_rdataset(dataname='Fair', package='Ecdat').data
dataset.head()
dataset.rename(columns={'nbaffairs':'affair'}, inplace=True)

dataset.info()
dataset.describe()
dataset.shape
dataset.dtypes

# Drop duplicated rows if any
dataset = dataset.drop_duplicates(keep='first') # Removed 24 duplicated rows
dataset.shape
 
dataset.isnull().sum() # No missing values

# Move numerical data into new dataset
num_data = dataset.select_dtypes(include=('int64','float64')).copy()

# Measures of Dispersion
np.var(num_data)
np.std(num_data)

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(num_data)
kurtosis(num_data)

# Histogram 
plt.hist(dataset['age']); plt.title('Histogram of Age');plt.xlabel('Age');plt.ylabel('Frequency')
plt.hist(dataset['ym'], color='coral'); plt.title('Histogram of Years_married');plt.xlabel('Years_married');plt.ylabel('Frequency')
plt.hist(dataset['religious'], color='skyblue'); plt.title('Histogram of Religious');plt.xlabel('Religious');plt.ylabel('Frequency')
plt.hist(dataset['education'], color='orange'); plt.title('Histogram of Education');plt.xlabel('Education');plt.ylabel('Frequency')
plt.hist(dataset['occupation'], color='lightblue'); plt.title('Histogram of Occupation');plt.xlabel('Occupation');plt.ylabel('Frequency')
plt.hist(dataset['rate'], color='brown'); plt.title('Histogram of Rating');plt.xlabel('Rating');plt.ylabel('Frequency')
plt.hist(dataset['affair'], color='violet');plt.title('Histogram of Affairs');plt.xlabel('Affairs');plt.ylabel('Frequency')

# Getting Barplot for catrgorical data
import seaborn as sns
sns.countplot(dataset['sex']).set_title('Count plot of Sex')
pd.crosstab(dataset.sex,dataset.affair).plot(kind="bar")

sns.countplot(x = 'child', data=dataset, palette='hls').set_title('Count plot of Children')
pd.crosstab(dataset.child,dataset.affair).plot(kind="bar")

# Normal Q-Q plot
plt.plot(num_data);plt.legend(['age', 'ym', 'religious', 'education', 'occupation','rate','affair'])

age  = np.array(dataset['age'])
ym   = np.array(dataset['ym'])
rel  = np.array(dataset['religious'])
edu  = np.array(dataset['education'])
occu = np.array(dataset['occupation'])
rate = np.array(dataset['rate'])
aff  = np.array(dataset['affair'])

from scipy import stats
stats.probplot(age, dist='norm',plot=plt);plt.title('Q-Q plot of Age')
stats.probplot(ym, dist='norm',plot=plt);plt.title('Q-Q plot of Years_married')
stats.probplot(rel, dist='norm',plot=plt);plt.title('Q-Q plot of Religious')
stats.probplot(edu, dist='norm',plot=plt);plt.title('Q-Q plot of Education')
stats.probplot(occu, dist='norm',plot=plt);plt.title('Q-Q plot of Occupation')
stats.probplot(rate, dist='norm',plot=plt);plt.title('Q-Q plot of Rating')
stats.probplot(aff, dist='norm', plot=plt);plt.title('Q-Q plot of Affairs')

# Normal probability distribution
x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.mean(x_age), np.std(x_age))
plt.plot(x_age, y_age);plt.xlim(np.min(x_age), np.max(x_age));plt.title('Normal Probability Distribution of Age');plt.xlabel('Age');plt.ylabel('Probability')

x_ym = np.linspace(np.min(ym), np.max(ym))
y_ym = stats.norm.pdf(x_ym, np.mean(x_ym), np.std(x_ym))
plt.plot(x_ym, y_ym, color='coral');plt.xlim(np.min(x_ym), np.max(x_ym));plt.title('Normal Probability Distribution of Years_married');plt.xlabel('Years_married');plt.ylabel('Probability')

x_rel = np.linspace(np.min(rel), np.max(rel))
y_rel = stats.norm.pdf(x_rel, np.mean(x_rel), np.std(x_rel))
plt.plot(x_rel, y_rel, color='blue');plt.xlim(np.min(x_rel), np.max(x_rel));plt.title('Normal Probability Distribution of Religious');plt.xlabel('Religious');plt.ylabel('Probability')

x_edu = np.linspace(np.min(edu), np.max(edu))
y_edu = stats.norm.pdf(x_edu, np.mean(x_edu), np.std(x_edu))
plt.plot(x_edu, y_edu, color='red');plt.xlim(np.min(x_edu), np.max(x_edu));plt.title('Normal Probability Distribution of Education');plt.xlabel('Education');plt.ylabel('Probability')

x_occu = np.linspace(np.min(occu), np.max(occu))
y_occu = stats.norm.pdf(x_occu, np.mean(x_occu), np.std(x_occu))
plt.plot(x_occu, y_occu, color='brown');plt.xlim(np.min(x_occu), np.max(x_occu));plt.title('Normal Probability Distribution of Occupation');plt.xlabel('Occupation');plt.ylabel('Probability')

x_rate = np.linspace(np.min(rate), np.max(rate))
y_rate = stats.norm.pdf(x_rate, np.mean(x_rate), np.std(x_rate))
plt.plot(x_rate, y_rate, color='violet');plt.xlim(np.min(x_rate), np.max(x_rate));plt.title('Normal Probability Distribution of Rating');plt.xlabel('Rating');plt.ylabel('Probability')

x_aff = np.linspace(np.min(aff), np.max(aff))
y_aff = stats.norm.pdf(x_aff, np.mean(x_aff), np.std(x_aff))
plt.plot(x_aff, y_aff, color='orange');plt.xlim(np.min(x_aff), np.max(x_aff));plt.title('Normal Probability Distribution of Affairs');plt.xlabel('Affairs');plt.ylabel('Probability')

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sns.boxplot(x="sex",y="age",data=dataset,palette="hls")
sns.boxplot(x="sex",y="ym",data=dataset,palette="hls")
sns.boxplot(x="sex",y="religious",data=dataset,palette="hls")
sns.boxplot(x="sex",y="education",data=dataset,palette="hls")
sns.boxplot(x="sex",y="occupation",data=dataset,palette="hls")
sns.boxplot(x="sex",y="rate",data=dataset,palette="hls")
sns.boxplot(x='sex',y='affair',data=dataset,palette='hls')

sns.boxplot(x="child",y="age",data=dataset,palette="hls")
sns.boxplot(x="child",y="ym",data=dataset,palette="hls")
sns.boxplot(x="child",y="religious",data=dataset,palette="hls")
sns.boxplot(x="child",y="education",data=dataset,palette="hls")
sns.boxplot(x="child",y="occupation",data=dataset,palette="hls")
sns.boxplot(x="child",y="rate",data=dataset,palette="hls")
sns.boxplot(x="child",y="affair",data=dataset,palette="hls")

# Boxplot of continuous variables 
sns.boxplot(x='age',data=dataset,orient='v')
sns.boxplot(x='ym',data=dataset,orient='v',color='coral')
sns.boxplot(x='religious',data=dataset,orient='v',color='lightblue')
sns.boxplot(x='education',data=dataset,orient='v',color='brown')
sns.boxplot(x='occupation',data=dataset,orient='v',color='orange')
sns.boxplot(x='rate',data=dataset,orient='v',color='lightgreen')
sns.boxplot(x='affair',data=dataset,orient='v',color='skyblue')

# Scatterplot 
sns.scatterplot(x='age', y='ym', data=dataset).set_title('Scatterplot of Age & Years_married')
sns.scatterplot(x='age', y='religious', data=dataset).set_title('Scatterplot of Age & Religious')
sns.scatterplot(x='age', y='education', data=dataset).set_title('Scatterplot of Age & Education')
sns.scatterplot(x='age', y='occupation', data=dataset).set_title('Scatterplot of Age & Occupation')
sns.scatterplot(x='age', y='rate', data=dataset).set_title('Scatterplot of Age & Rating')
sns.scatterplot(x='age', y='affair', data=dataset).set_title('Scatterplot of Age & Affairs')

sns.scatterplot(x='ym', y='religious', data=dataset).set_title('Scatterplot of Years_married & Religious')
sns.scatterplot(x='ym', y='education', data=dataset).set_title('Scatterplot of Years_married & Education')
sns.scatterplot(x='ym', y='occupation', data=dataset).set_title('Scatterplot of Years_married & Occupation')
sns.scatterplot(x='ym', y='rate', data=dataset).set_title('Scatterplot of Years_married & Rating')
sns.scatterplot(x='ym', y='affair', data=dataset).set_title('Scatterplot of Years_married & Affairs')

sns.scatterplot(x='religious', y='education', data=dataset).set_title('Scatterplot of Religious & Education')
sns.scatterplot(x='religious', y='occupation', data=dataset).set_title('Scatterplot of Religious & Occupation')
sns.scatterplot(x='religious', y='rate', data=dataset).set_title('Scatterplot of Religious & Rating')
sns.scatterplot(x='religious', y='affair', data=dataset).set_title('Scatterplot of Religious & Affairs')

sns.scatterplot(x='education', y='occupation', data=dataset).set_title('Scatterplot of Education & Occupation')
sns.scatterplot(x='education', y='rate', data=dataset).set_title('Scatterplot of Education & Rating')
sns.scatterplot(x='education', y='affair', data=dataset).set_title('Scatterplot of Education & Affairs')

sns.scatterplot(x='occupation', y='rate', data=dataset).set_title('Scatterplot of Occupation & Rating')
sns.scatterplot(x='occupation', y='affair', data=dataset).set_title('Scatterplot of Occupation & Affairs')

sns.scatterplot(x='rate', y='affair', data=dataset).set_title('Scatterplot of Rating & Affairs')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='sex')
sns.pairplot(dataset, hue='affair')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Outliers
# Z-score values
from scipy import stats
z = np.abs(stats.zscore(num_data))
threshold = 3
print(np.where(z>3))
num_data.shape # 577 rows
df_out = num_data[(z<3).all(axis=1)]
df_out.shape # 37 rows are removed

# creating dummy variables
dummies = pd.get_dummies(dataset[['sex','child']],drop_first=True)

final_data = pd.concat([dummies, df_out], axis=1)
final_data.shape
final_data.isnull().sum()
final_data.dropna(inplace=True)

# Converting affairs class into binary format
for val in final_data.affair:
	if val >=1:
		final_data['affair'].replace(val, 1,inplace=True)

final_data['affair'] = final_data['affair'].astype('category')
final_data.dtypes
final_data['affair'].value_counts()

# Matrics of features
X = final_data.iloc[:,0:8].values
X_col = final_data.drop('affair', axis=1).columns.tolist()
Y = final_data.iloc[:, -1].values

'''# Copy the categorical data into new datset
X_obj = X.select_dtypes(include=['object']).copy()
X_obj.head()
X_obj = X_obj.columns.tolist()

for var in X_obj:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(X[var],prefix=var)
    X = X.join(cat_list)
    
X_col = X.columns.values.tolist()
to_keep = [i for i in X_col if i not in X_obj]

X = X[to_keep]
X.columns.values'''

# Model building Before sampling
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # Coefficients of features
classifier.predict_proba(X) # probability values

Y_pred = classifier.predict(X)
final_data['y_pred'] = Y_pred
Y_prob = pd.DataFrame(classifier.predict_proba(X))
new_df = pd.concat([final_data, Y_prob], axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,Y_pred) #[423 5/108 4] 
print(cm)
type(Y_pred)
acc = sum(Y==Y_pred)/dataset.shape[0]
print(acc)  # 74%
pd.crosstab(Y_pred, Y)

# Target variable is imbalanced
final_data['affair'].value_counts()

# Over sampling SMOTE technique
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0)
os_data_X, os_data_Y = os.fit_sample(X, Y)

os_data_X = pd.DataFrame(data=os_data_X, columns=X_col) #X.columns
os_data_Y = pd.DataFrame(data=os_data_Y, columns=['affair']) # Y.columns

# we can check the numbers of our data
print("length of oversampled data is", len(os_data_X))
print("Number of no-affairs in over sampled data", len(os_data_Y[os_data_Y['affair']==0]))
print('Number of affairs in over sampled data', len(os_data_Y[os_data_Y['affair']==1]))
print("Proportion of no-affairs data in oversampled data is ", len(os_data_Y[os_data_Y['affair']==0])/ len(os_data_X))
print("Proportion of affairs data in oversampled data is ", len(os_data_Y[os_data_Y['affair']==1])/ len(os_data_X))

# Recursive Feature Elimination = RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

rfe = RFE(classifier) # RFE(classifier, n_features=20)
rfe = rfe.fit(os_data_X, os_data_Y)
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)

print("The Recursive Feature Elimination(RFE) has helped us select the following features:")
# Iterate from array de rfe.support_ and pick columns that are == True
fea_cols = rfe.get_support(1) # Most important features
X_final = os_data_X[os_data_X.columns[fea_cols]] # final features
Y_final = os_data_Y['affair']

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.25, random_state=0)

# Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Predicting the Trainset Results
classifier.coef_ # coefficients of features 
classifier.predict_proba (X_train) # Probability values 

y_pred_train = classifier.predict(X_train)
y_prob_train = pd.DataFrame(classifier.predict_proba(X_train.iloc[:,:]))

X_train["y_pred"] = y_pred_train
new_df_train = pd.concat([X_train,y_prob_train],axis=1)

from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(Y_train,y_pred_train)
print (cm_train) #[190 129/117 206]
acc_train = sum(Y_train==y_pred_train)/X_train.shape[0]
print(acc_train) # 61% 
pd.crosstab(y_pred_train,Y_train)

# Predicting the test results and calculating the accuracy
Y_pred_test = classifier.predict(X_test)
print('Accuracy of Logistic Regression classifier on test set:{:.2f}'.format(classifier.score(X_test, Y_test)))

# Confusion Matrix
cm_test = confusion_matrix(Y_test,Y_pred_test)
print(cm_test) #[72 37/37 68]
acc_test = sum(Y_test==Y_pred_test)/X_test.shape[0]
print(acc_test) # 65% 

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class_roc_auc = roc_auc_score(Y_test, Y_pred_test)
print(class_roc_auc)
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%class_roc_auc)
plt.plot([0,1], [0,1],'r--');plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic');plt.legend(loc="lower right")

# Logistic Regression on total dataset
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression()
classifier1.fit(X_final, Y_final)

classifier1.coef_ # coefficients of features 
classifier1.predict_proba (X_final) # Probability values 

y_pred1 = classifier1.predict(X_final)
os_data_X["y_pred"] = y_pred1
y_prob1 = pd.DataFrame(classifier1.predict_proba(X_final.iloc[:,:]))
new_df1 = pd.concat([os_data_X,y_prob1],axis=1)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_final,y_pred1)
print (cm1) # [266 162/156 272]
type(y_pred1)
acc1 = sum(Y_final==y_pred1)/os_data_X.shape[0]
print(acc1) # 62%
pd.crosstab(y_pred1,Y_final)

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_final, y_pred1))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

class1_roc_auc = roc_auc_score(Y_final, y_pred1)
print(class1_roc_auc)
fpr, tpr, thresholds = roc_curve(Y_final, classifier1.predict_proba(X_final)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%class1_roc_auc)
plt.plot([0,1], [0,1],'r--');plt.xlim([0.0, 1.0]);plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic');plt.legend(loc="lower right")
