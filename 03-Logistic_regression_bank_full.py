#reset console
%reset -f
# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('bank-full.csv', sep=';')
dataset.head()
dataset.shape

dataset.info()
# Missing values
dataset.isnull().sum()
# Drop duplicate rows
dataset.drop_duplicates(keep='first', inplace=True) # No duplicate rows
dataset.shape
dataset.columns
dataset.dtypes

# Seperate the numerical and categorical data
num_data = dataset.select_dtypes(include=('int64')).copy()
cat_data = dataset.select_dtypes(include=('object')).copy()

num_data.describe()

# Measures of Dispersion
np.var(num_data)
np.std(num_data)

# Skewness and kurtosis
from scipy.stats import skew, kurtosis
skew(num_data)
kurtosis(num_data)

# Histogram
plt.hist(num_data['age']);plt.title('Histogram of Age');plt.xlabel('Age');plt.ylabel('Frequency')
plt.hist(num_data['balance'], color='skyblue');plt.title('Histogram of Balance');plt.xlabel('Balance');plt.ylabel('Frequency')
plt.hist(num_data['day'], color='coral');plt.title('Histogram of Day');plt.xlabel('Day');plt.ylabel('Frequency')
plt.hist(num_data['duration'], color='lightgreen');plt.title('Histogram of Duration');plt.xlabel('Duration');plt.ylabel('Frequency')
plt.hist(num_data['campaign'], color='brown');plt.title('Histogram of Campaign');plt.xlabel('Campaign');plt.ylabel('Frequency')
plt.hist(num_data['pdays'], color='lightblue');plt.title('Histogram of PDays');plt.xlabel('PDays');plt.ylabel('Frequency')
plt.hist(num_data['previous'], color='violet');plt.title('Histogram of Previous');plt.xlabel('Previous');plt.ylabel('Frequency')

# Barpot for categorical data
import seaborn as sns
sns.countplot(cat_data['job']).set_title('Countplot')
pd.crosstab(cat_data.job, cat_data.education).plot(kind='bar')

sns.countplot(cat_data['marital']).set_title('Countplot')
pd.crosstab(cat_data.marital, cat_data.loan).plot(kind='bar')

sns.countplot(cat_data['education']).set_title('Countplot')
sns.countplot(cat_data['default']).set_title('Countplot')
sns.countplot(cat_data['housing']).set_title('Countplot')
sns.countplot(cat_data['loan']).set_title('Countplot')
sns.countplot(cat_data['contact']).set_title('Countplot')
sns.countplot(cat_data['month']).set_title('Countplot')
sns.countplot(cat_data['poutcome']).set_title('Countplot')
sns.countplot(cat_data['y']).set_title('Countplot')

count_no_sub = len(dataset[dataset['y']=='no'])
count_sub = len(dataset[dataset['y']=='yes'])
pct_of_no_sub = count_no_sub / len(dataset['y'])
print("Percentage of no subscription is", pct_of_no_sub * 100)
pct_of_sub = count_sub / len(dataset['y'])
print("Percentage of subscription", pct_of_sub * 100)

''' Our classes are imbalanced and ratio of No-Subscription and Subscription 
is 89:11. So we need to balance them , do some more exploration '''

dataset.groupby('y').mean()
dataset.groupby('job').mean()
dataset.groupby('marital').mean()
dataset.groupby('education').mean()

''' purchase of the deposit depends on job title mostly. its good predictor for out come. '''
#%matplotlib inline
pd.crosstab(dataset.job, dataset.y).plot(kind='bar');plt.title('Purchase Frequency for Job Title');plt.ylabel('Frequency of Purchase')

table = pd.crosstab(dataset.marital, dataset.y)
table.div(table.sum(1).astype(float), axis = 0).plot(kind ='bar', stacked = True);plt.title('Stacked Bar Chart of Marital Status vs Purchase');plt.xlabel('Marital Status');plt.ylabel('Proportion of Customers')
# Marital status does not seem a strong predictor for the outcome variable 

table = pd.crosstab(dataset.education, dataset.y)
table.div(table.sum(1).astype(float), axis = 0).plot(kind ='bar', stacked = True);plt.title('Stacked Barchart of Education & Customers');plt.xlabel('Education');plt.ylabel('Proportion of Customers')
# Education seems a good predictor of the outcome variable
 
pd.crosstab(dataset.day, dataset.y).plot(kind='bar');plt.title('Purchase Frequency for Day of Week');plt.xlabel('Day of Week');plt.ylabel('Frequency of Purchase')
# Day of week may not be a good predictor for the outcome

pd.crosstab(dataset.month, dataset.y).plot(kind='bar');plt.title('Purchase Frequency for Month');plt.xlabel('Month');plt.ylabel('Frequency of Purchase')
# Month might be a good predictor of the outcome

'''pd.crosstab(dataset.balance, dataset.y).plot(kind='bar')
table = pd.crosstab(dataset.balance, dataset.y)
table.div(table.sum(1).astype(float), axis = 0).plot(kind ='bar', stacked = True) '''

dataset['poutcome'].unique()
pd.crosstab(dataset.poutcome, dataset.y).plot(kind='bar');plt.title('Purchase Frequency for Poutcome');plt.xlabel('poutcome');plt.ylabel('Frequency of purchase')
# Poutcome seems to be a good predictor of the outcome variable

# Normal Q-Q plot
plt.plot(num_data);plt.legend(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])

age = np.array(num_data['age'])
bal = np.array(num_data['balance'])
day = np.array(num_data['day'])
dur = np.array(num_data['duration'])
cam = np.array(num_data['campaign'])
pdays = np.array(num_data['pdays'])
prev = np.array(num_data['previous'])

from scipy import stats
stats.probplot(age, dist='norm',plot=plt);plt.title('Q-Q plot of Age')
stats.probplot(bal, dist='norm',plot=plt);plt.title('Q-Q plot of Balance')
stats.probplot(day, dist='norm',plot=plt);plt.title('Q-Q plot of Day')
stats.probplot(dur, dist='norm',plot=plt);plt.title('Q-Q plot of Duration')
stats.probplot(cam, dist='norm',plot=plt);plt.title('Q-Q plot of Campaign')
stats.probplot(pdays, dist='norm',plot=plt);plt.title('Q-Q plot of PDays')
stats.probplot(prev, dist='norm',plot=plt);plt.title('Q-Q plot of Previous')

# Normal probability distribution
x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.mean(x_age), np.std(x_age))
plt.plot(x_age, y_age, color='skyblue');plt.xlim(np.min(x_age), np.max(x_age));plt.title('Normal Probability Distribution of Age');plt.xlabel('Age');plt.ylabel('Probability')

x_bal = np.linspace(np.min(bal), np.max(bal))
y_bal = stats.norm.pdf(x_bal, np.mean(x_bal), np.std(x_bal))
plt.plot(x_bal, y_bal, color='coral');plt.xlim(np.min(x_bal), np.max(x_bal));plt.title('Normal Probability Distribution of Balance');plt.xlabel('Balance');plt.ylabel('Probability')

x_day = np.linspace(np.min(day), np.max(day))
y_day = stats.norm.pdf(x_day, np.mean(x_day), np.std(x_day))
plt.plot(x_day, y_day, color='blue');plt.xlim(np.min(x_day), np.max(x_day));plt.title('Normal Probability Distribution of Day');plt.xlabel('Day');plt.ylabel('Probability')

x_dur = np.linspace(np.min(dur), np.max(dur))
y_dur = stats.norm.pdf(x_dur, np.mean(x_dur), np.std(x_dur))
plt.plot(x_dur, y_dur, color='red');plt.xlim(np.min(x_dur), np.max(x_dur));plt.title('Normal Probability Distribution of Duration');plt.xlabel('Duration');plt.ylabel('Probability')

x_cam = np.linspace(np.min(cam), np.max(cam))
y_cam = stats.norm.pdf(x_cam, np.mean(x_cam), np.std(x_cam))
plt.plot(x_cam, y_cam, color='brown');plt.xlim(np.min(x_cam), np.max(x_cam));plt.title('Normal Probability Distribution of Campaign');plt.xlabel('Campaign');plt.ylabel('Probability')

x_pdays = np.linspace(np.min(pdays), np.max(pdays))
y_pdays = stats.norm.pdf(x_pdays, np.mean(x_pdays), np.std(x_pdays))
plt.plot(x_pdays, y_pdays, color='violet');plt.xlim(np.min(x_pdays), np.max(x_pdays));plt.title('Normal Probability Distribution of PDays');plt.xlabel('PDays');plt.ylabel('Probability')

x_prev = np.linspace(np.min(prev), np.max(prev))
y_prev = stats.norm.pdf(x_prev, np.mean(x_prev), np.std(x_prev))
plt.plot(x_prev, y_prev, color='green');plt.xlim(np.min(x_prev), np.max(x_prev));plt.title('Normal Probability Distribution of Previous');plt.xlabel('Previous');plt.ylabel('Probability')

# Boxplot
sns.boxplot(data=num_data).set_title('Boxplot of Numerical Variables')
# Boxplot of num_variables wrt each object 
sns.boxplot(x='job', y='age', data=dataset).set_title('Boxplot of Job & Age')
sns.boxplot(x='job', y='balance', data=dataset).set_title('Boxplot of Job & Balance')
sns.boxplot(x='job', y='day', data=dataset).set_title('Boxplot of Job & Day')
sns.boxplot(x='job', y='duration', data=dataset).set_title('Boxplot of Job & Duration')
sns.boxplot(x='job', y='campaign', data=dataset).set_title('Boxplot of Job & Campaign')
sns.boxplot(x='job', y='pdays', data=dataset).set_title('Boxplot of Job & PDays')
sns.boxplot(x='job', y='previous', data=dataset).set_title('Boxplot of Job & Previous')

sns.boxplot(x='marital', y='age', data=dataset).set_title('Boxplot of Marital & Age')
sns.boxplot(x='marital', y='balance', data=dataset).set_title('Boxplot of Marital & Balance')
sns.boxplot(x='marital', y='day', data=dataset).set_title('Boxplot of Marital & Day')
sns.boxplot(x='marital', y='duration', data=dataset).set_title('Boxplot of Marital & Duration')
sns.boxplot(x='marital', y='campaign', data=dataset).set_title('Boxplot of Marital & Campaign')
sns.boxplot(x='marital', y='pdays', data=dataset).set_title('Boxplot of Marital & PDays')
sns.boxplot(x='marital', y='previous', data=dataset).set_title('Boxplot of Marital & Previous')

sns.boxplot(x='education', y='age', data=dataset).set_title('Boxplot of Education & Age')
sns.boxplot(x='education', y='balance', data=dataset).set_title('Boxplot of Education & Balance')
sns.boxplot(x='education', y='day', data=dataset).set_title('Boxplot of Education & Day')
sns.boxplot(x='education', y='duration', data=dataset).set_title('Boxplot of Education & Duration')
sns.boxplot(x='education', y='campaign', data=dataset).set_title('Boxplot of Education & Campaign')
sns.boxplot(x='education', y='pdays', data=dataset).set_title('Boxplot of Education & PDays')
sns.boxplot(x='education', y='previous', data=dataset).set_title('Boxplot of Education & Previous')

sns.boxplot(x='default', y='age', data=dataset).set_title('Boxplot of Default & Age')
sns.boxplot(x='default', y='balance', data=dataset).set_title('Boxplot of Default & Balance')
sns.boxplot(x='default', y='day', data=dataset).set_title('Boxplot of Default & Day')
sns.boxplot(x='default', y='duration', data=dataset).set_title('Boxplot of Default & Duration')
sns.boxplot(x='default', y='campaign', data=dataset).set_title('Boxplot of Default & Campaign')
sns.boxplot(x='default', y='pdays', data=dataset).set_title('Boxplot of Default & PDays')
sns.boxplot(x='default', y='previous', data=dataset).set_title('Boxplot of Default & Previous')

sns.boxplot(x='housing', y='age', data=dataset).set_title('Boxplot of Housing & Age')
sns.boxplot(x='housing', y='balance', data=dataset).set_title('Boxplot of Housing & Balance')
sns.boxplot(x='housing', y='day', data=dataset).set_title('Boxplot of Housing & Day')
sns.boxplot(x='housing', y='duration', data=dataset).set_title('Boxplot of Housing & Duration')
sns.boxplot(x='housing', y='campaign', data=dataset).set_title('Boxplot of Housing & Campaign')
sns.boxplot(x='housing', y='pdays', data=dataset).set_title('Boxplot of Housing & PDays')
sns.boxplot(x='housing', y='previous', data=dataset).set_title('Boxplot of Housing & Previous')

sns.boxplot(x='loan', y='age', data=dataset).set_title('Boxplot of Loan & Age')
sns.boxplot(x='loan', y='balance', data=dataset).set_title('Boxplot of Loan & Balance')
sns.boxplot(x='loan', y='day', data=dataset).set_title('Boxplot of Loan & Day')
sns.boxplot(x='loan', y='duration', data=dataset).set_title('Boxplot of Loan & Duration')
sns.boxplot(x='loan', y='campaign', data=dataset).set_title('Boxplot of Loan & Campaign')
sns.boxplot(x='loan', y='pdays', data=dataset).set_title('Boxplot of Loan & PDays')
sns.boxplot(x='loan', y='previous', data=dataset).set_title('Boxplot of Loan & Previous')

sns.boxplot(x='contact', y='age', data=dataset).set_title('Boxplot of Contact & Age')
sns.boxplot(x='contact', y='balance', data=dataset).set_title('Boxplot of Contact & Balance')
sns.boxplot(x='contact', y='day', data=dataset).set_title('Boxplot of Contact & Day')
sns.boxplot(x='contact', y='duration', data=dataset).set_title('Boxplot of Contact & Duration')
sns.boxplot(x='contact', y='campaign', data=dataset).set_title('Boxplot of Contact & Campaign')
sns.boxplot(x='contact', y='pdays', data=dataset).set_title('Boxplot of Contact & PDays')
sns.boxplot(x='contact', y='previous', data=dataset).set_title('Boxplot of Contact & Previous')

sns.boxplot(x='month', y='age', data=dataset).set_title('Boxplot of Month & Age')
sns.boxplot(x='month', y='balance', data=dataset).set_title('Boxplot of Month & Balance')
sns.boxplot(x='month', y='day', data=dataset).set_title('Boxplot of Month & Day')
sns.boxplot(x='month', y='duration', data=dataset).set_title('Boxplot of Month & Duration')
sns.boxplot(x='month', y='campaign', data=dataset).set_title('Boxplot of Month & Campaign')
sns.boxplot(x='month', y='pdays', data=dataset).set_title('Boxplot of Month & PDays')
sns.boxplot(x='month', y='previous', data=dataset).set_title('Boxplot of Month & Previous')

sns.boxplot(x='poutcome', y='age', data=dataset).set_title('Boxplot of Poutcome & Age')
sns.boxplot(x='poutcome', y='balance', data=dataset).set_title('Boxplot of Poutcome & Balance')
sns.boxplot(x='poutcome', y='day', data=dataset).set_title('Boxplot of Poutcome & Day')
sns.boxplot(x='poutcome', y='duration', data=dataset).set_title('Boxplot of Poutcome & Duration')
sns.boxplot(x='poutcome', y='campaign', data=dataset).set_title('Boxplot of Poutcome & Campaign')
sns.boxplot(x='poutcome', y='pdays', data=dataset).set_title('Boxplot of Poutcome & PDays')
sns.boxplot(x='poutcome', y='previous', data=dataset).set_title('Boxplot of Poutcome & Previous')

sns.boxplot(x='y', y='age', data=dataset).set_title('Boxplot of Y & Age')
sns.boxplot(x='y', y='balance', data=dataset).set_title('Boxplot of Y & Balance')
sns.boxplot(x='y', y='day', data=dataset).set_title('Boxplot of Y & Day')
sns.boxplot(x='y', y='duration', data=dataset).set_title('Boxplot of Y & Duration')
sns.boxplot(x='y', y='campaign', data=dataset).set_title('Boxplot of Y & Campaign')
sns.boxplot(x='y', y='pdays', data=dataset).set_title('Boxplot of Y & PDays')
sns.boxplot(x='y', y='previous', data=dataset).set_title('Boxplot of Y & Previous')

# Boxplot of num_data
sns.boxplot(x='age',data=dataset, orient='v').set_title('Boxplot of Age')
sns.boxplot(x='balance', data=dataset, orient='v', color='coral').set_title('Boxplot of Balance')
sns.boxplot(x='day', data=dataset, orient='v', color='skyblue').set_title('Boxplot of Days')
sns.boxplot(x='duration', data=dataset, orient='v', color='orange').set_title('Boxplot of Duration')
sns.boxplot(x='campaign', data=dataset, orient='v', color='lightblue').set_title('Boxplot of Campaign')
sns.boxplot(x='pdays', data=dataset, orient='v', color='brown').set_title('Boxplot of PDays')
sns.boxplot(x='previous', data=dataset, orient='v', color='lightgreen').set_title('Boxplot of Previous')

# scatterplot
sns.scatterplot(x='age', y='balance', data=dataset).set_title('Scatterplot of Age & Balance')
sns.scatterplot(x='age', y='day', data=dataset).set_title('Scatterplot of Age & Day')
sns.scatterplot(x='age', y='duration', data=dataset).set_title('Scatterplot of Age & Balance')
sns.scatterplot(x='age', y='campaign', data=dataset).set_title('Scatterplot of Age & Campaign')
sns.scatterplot(x='age', y='pdays', data=dataset).set_title('Scatterplot of Age & PDays')

sns.scatterplot(x='balance', y='day', data=dataset).set_title('Scatterplot of Balance & Day')
sns.scatterplot(x='balance', y='duration', data=dataset).set_title('Scatterplot of Balance & duration')
sns.scatterplot(x='balance', y='campaign', data=dataset).set_title('Scatterplot of Balance & campaign')
sns.scatterplot(x='balance', y='pdays', data=dataset).set_title('Scatterplot of Balance & PDays')

sns.scatterplot(x='day', y='duration', data=dataset).set_title('Scatterplot of Day & Duration')
sns.scatterplot(x='day', y='campaign', data=dataset).set_title('Scatterplot of Day & Campaign')
sns.scatterplot(x='day', y='pdays', data=dataset).set_title('Scatterplot of Day & PDays')

sns.scatterplot(x='duration', y='campaign', data=dataset).set_title('Scatterplot of Duration & Campaign')
sns.scatterplot(x='duration', y='pdays', data=dataset).set_title('Scatterplot of Duration & PDays')

sns.scatterplot(x='campaign', y='pdays', data=dataset).set_title('Scatterplot of Campaign & PDays')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='y')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Create Dummy variables
cat_data.columns
cat_dummies = pd.get_dummies(cat_data, drop_first=True)

'''
cat_var = ['job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'poutcome', 'y']
for var in cat_var:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(dataset[var], prefix=var)
    dataset = dataset.join(cat_list)
    
dataset_col = dataset.columns.values.tolist()
to_keep = [i for i in dataset_col if i not in cat_var]

dataset = dataset[to_keep]
dataset.columns.values 
 
dataset_final.drop('y_no', axis=1)
dataset_final.rename(columns={'y_yes':'y'}, inplace =True) '''

# Outliers
# Z score
from scipy import stats
z = np.abs(stats.zscore(num_data))
threshold=3
print(np.where(z>3))
num_data.shape
df_out = num_data[(z<3).all(axis=1)] 
df_out.shape # 5002 rows are removed

final_data = pd.concat([df_out, cat_dummies], axis=1)
final_data.rename(columns={'y_yes':'y'}, inplace=True)

final_data.isnull().sum()
final_data.dropna(inplace=True)

# Metrics of features
X = final_data.iloc[:, 0:42].values
X_col = final_data.drop('y',axis=1).columns.tolist()
Y = final_data.iloc[:, -1].values

dataset['y'].value_counts() # Imbalanced

# ''' Over sampling using SMOTE '''
# SMOTE = Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)

os_data_X, os_data_Y = os.fit_sample(X, Y)
os_data_X = pd.DataFrame(data=os_data_X, columns=X_col) # Putting columns into dataframe of X
os_data_Y = pd.DataFrame(data=os_data_Y, columns=['y'])

# we can check the numbers of our data
print("length of oversampled data is", len(os_data_X))
print("Number of no-subscription in over sampled data", len(os_data_Y[os_data_Y['y']==0]))
print('Number of subscription', len(os_data_Y[os_data_Y['y']==1]))
print("Proportion of no subscription data in oversampled data is ", len(os_data_Y[os_data_Y['y']==0])/ len(os_data_X))
print("Proportion of subscription data in oversampled data is ", len(os_data_Y[os_data_Y['y']==1])/ len(os_data_X))

# Recursive Feature Elimination = RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

rfe = RFE(classifier)
rfe = rfe.fit(os_data_X, os_data_Y)
print(rfe.support_)
print(rfe.ranking_)
print(rfe.n_features_)

print("The Recursive Feature Elimination(RFE) has helped us select the following features:")
# Iterate from array de rfe.support_ and pick columns that are == True
fea_cols = rfe.get_support(1) # Most important features
X_final = os_data_X[os_data_X.columns[fea_cols]] # final features
#X = X.drop('y_no', axis=1)
# X = dataset_final[cols]
Y_final = os_data_Y['y']

# Implementing  logit the model
import statsmodels.api as sm
logit_model = sm.Logit(Y_final, X_final).fit()
logit_model.summary()

# Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.3, random_state=0)
model1 = LogisticRegression()
model1.fit(X_train, Y_train)

# Predicting the test results and calculating the accuracy
Y_pred = model1.predict(X_test)
print('Accuracy of Logistic Regression classifier on test set:{:.2f}'.format(model1.score(X_test, Y_test)))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)

# Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

model1_roc_auc = roc_auc_score(Y_test, Y_pred)
fpr, tpr, thresholds = roc_curve(Y_test, model1.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)'%model1_roc_auc)
plt.plot([0,1], [0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
