# Excel R solutions
%reset -f

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
dataset = pd.read_csv('glass.csv')
dataset.head()
dataset.dtypes
dataset.columns
dataset.Type.value_counts()
#dataset.columns = map(str.lower, dataset.columns)  # convert Columns into lower case

# Missing values
dataset.isnull().sum() #No missing values
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # One duplicate row
dataset.shape

# Statistical Description
dataset.describe()
# Measure of dispersion
np.var(dataset)
np.std(dataset)
# Skewness and kurtosis
from scipy.stats import skew, kurtosis
skew(dataset)
kurtosis(dataset)

# Histogram 
plt.hist(dataset['RI']);plt.title('Histogram of RI');plt.xlabel('RI');plt.ylabel('Frequency')
plt.hist(dataset['Na'], color='coral');plt.title('Histogram of Na');plt.xlabel('Na');plt.ylabel('Frequency')
plt.hist(dataset['K'], color='skyblue');plt.title('Histogram of Mg');plt.xlabel('Mg');plt.ylabel('Frequency')
plt.hist(dataset['Al'], color='orange');plt.title('Histogram of Al');plt.xlabel('Al');plt.ylabel('Frequency')
plt.hist(dataset['Si'], color='lightblue');plt.title('Histogram of Si');plt.xlabel('Si');plt.ylabel('Frequency')
plt.hist(dataset['K'], color='brown');plt.title('Histogram of K');plt.xlabel('K');plt.ylabel('Frequency')
plt.hist(dataset['Ca'], color='violet');plt.title('Histogram of Ca');plt.xlabel('Ca');plt.ylabel('Frequency')
plt.hist(dataset['Ba'], color='teal');plt.title('Histogram of Ba');plt.xlabel('Ba');plt.ylabel('Frequency')
plt.hist(dataset['Fe'], color='lightgreen');plt.title('Histogram of Fe');plt.xlabel('Fe');plt.ylabel('Frequency')

# barplot
import seaborn as sns
sns.countplot(dataset['Type']).set_title('Countplot of Type')

# Numerical data
num_data = dataset.iloc[:,:-1]
# Normal Q-Q plot
plt.plot(num_data);plt.legend(list(num_data))

RI = np.array(num_data['RI'])
Na = np.array(num_data['Na'])
Mg = np.array(num_data['Mg'])
Al = np.array(num_data['Al'])
Si = np.array(num_data['Si'])
K = np.array(num_data['K'])
Ca = np.array(num_data['Ca'])
Ba = np.array(num_data['Ba'])
Fe = np.array(num_data['Fe'])

from scipy import stats
stats.probplot(RI, dist='norm', plot=plt);plt.title('Q-Q plot of RI')
stats.probplot(Na, dist='norm', plot=plt);plt.title('Q-Q plot of Na')
stats.probplot(Mg, dist='norm', plot=plt);plt.title('Q-Q plot of Mg')
stats.probplot(Al, dist='norm', plot=plt);plt.title('Q-Q plot of Al')
stats.probplot(Si, dist='norm', plot=plt);plt.title('Q-Q plot of Si')
stats.probplot(K, dist='norm', plot=plt);plt.title('Q-Q plot of K')
stats.probplot(Ca, dist='norm', plot=plt);plt.title('Q-Q plot of Ca')
stats.probplot(Ba, dist='norm', plot=plt);plt.title('Q-Q plot of Ba')
stats.probplot(Fe, dist='norm', plot=plt);plt.title('Q-Q plot of Fe')

# Normal Probility Distribution
x_RI = np.linspace(np.min(RI), np.max(RI))
y_RI = stats.norm.pdf(x_RI, np.median(x_RI), np.std(x_RI))
plt.plot(x_RI, y_RI);plt.xlim(np.min(RI), np.max(RI));plt.title('Normal Probability Distribution of RI');plt.xlabel('RI');plt.ylabel('Probability')

x_Na = np.linspace(np.min(Na), np.max(Na))
y_Na = stats.norm.pdf(x_Na, np.median(x_Na), np.std(x_Na))
plt.plot(x_Na, y_Na);plt.xlim(np.min(Na), np.max(Na));plt.title('Normal Probability Distribution of Na');plt.xlabel('Na');plt.ylabel('Probability')

x_Mg = np.linspace(np.min(Mg), np.max(Mg))
y_Mg = stats.norm.pdf(x_Mg, np.median(x_Mg), np.std(x_Mg))
plt.plot(x_Mg, y_Mg);plt.xlim(np.min(Mg), np.max(Mg));plt.title('Normal Probability Distribution of Mg');plt.xlabel('Mg');plt.ylabel('Probability')

x_Al = np.linspace(np.min(Al), np.max(Al))
y_Al = stats.norm.pdf(x_Al, np.median(x_Al), np.std(x_Al))
plt.plot(x_Al, y_Al);plt.xlim(np.min(Al), np.max(Al));plt.title('Normal Probability Distribution of Al');plt.xlabel('Al');plt.ylabel('Probability')

x_Si = np.linspace(np.min(Si), np.max(Si))
y_Si = stats.norm.pdf(x_Si, np.median(x_Si), np.std(x_Si))
plt.plot(x_Si, y_Si);plt.xlim(np.min(Si), np.max(Si));plt.title('Normal Probability Distribution of Si');plt.xlabel('Si');plt.ylabel('Probability')

x_K = np.linspace(np.min(K), np.max(K))
y_K = stats.norm.pdf(x_K, np.median(x_K), np.std(x_K))
plt.plot(x_K, y_K);plt.xlim(np.min(K), np.max(K));plt.title('Normal Probability Distribution of K');plt.xlabel('K');plt.ylabel('Probability')

x_Ca = np.linspace(np.min(Ca), np.max(Ca))
y_Ca = stats.norm.pdf(x_Ca, np.median(x_Ca), np.std(x_Ca))
plt.plot(x_Ca, y_Ca);plt.xlim(np.min(Ca), np.max(Ca));plt.title('Normal Probability Distribution of Ca');plt.xlabel('Ca');plt.ylabel('Probability')

x_Ba = np.linspace(np.min(Ba), np.max(Ba))
y_Ba = stats.norm.pdf(x_Ba, np.median(x_Ba), np.std(x_Ba))
plt.plot(x_Ba, y_Ba);plt.xlim(np.min(Ba), np.max(Ba));plt.title('Normal Probability Distribution of Ba');plt.xlabel('Ba');plt.ylabel('Probability')

x_Fe = np.linspace(np.min(Fe), np.max(Fe))
y_Fe = stats.norm.pdf(x_Fe, np.median(x_Fe), np.std(x_Fe))
plt.plot(x_Fe, y_Fe);plt.xlim(np.min(Fe), np.max(Fe));plt.title('Normal Probability Distribution of Fe');plt.xlabel('Fe');plt.ylabel('Probability')
# all are follows the normal probability distribution

# Boxplot of numerical data
sns.boxplot(num_data['RI'], orient='v').set_title('Boxplot of RI')
sns.boxplot(num_data['Na'], color='coral', orient='v').set_title('Boxplot of Na')
sns.boxplot(num_data['Mg'], color='skyblue', orient='v').set_title('Boxplot of Mg')
sns.boxplot(num_data['Al'], color='orange', orient='v').set_title('Boxplot of Al')
sns.boxplot(num_data['Si'],color='lightblue', orient='v').set_title('Boxplot of Si')
sns.boxplot(num_data['K'],color='brown', orient='v').set_title('Boxplot of K')
sns.boxplot(num_data['Ca'],color='violet', orient='v').set_title('Boxplot of Ca')
sns.boxplot(num_data['Ba'],color='teal', orient='v').set_title('Boxplot of Ba')
sns.boxplot(num_data['Fe'],color='purple', orient='v').set_title('Boxplot of Fe')

# Boxplot of categorical data wrt numerical data
sns.boxplot(x='Type',y='RI', data=dataset).set_title('Boxplot of Type & RI')
sns.boxplot(x='Type',y='Na', color='coral', data=dataset).set_title('Boxplot of Type & Na')
sns.boxplot(x='Type',y='Mg', color='skyblue', data=dataset).set_title('Boxplot of Type & Mg')
sns.boxplot(x='Type',y='Al', color='orange', data=dataset).set_title('Boxplot of Type & Al')
sns.boxplot(x='Type',y='Si', color='lightblue', data=dataset).set_title('Boxplot of Type & Si')
sns.boxplot(x='Type',y='K', color='brown', data=dataset).set_title('Boxplot of Type & K')
sns.boxplot(x='Type',y='Ca', color='violet', data=dataset).set_title('Boxplot of Type & Ca')
sns.boxplot(x='Type',y='Ba', color='teal', data=dataset).set_title('Boxplot of Type & Ba')
sns.boxplot(x='Type',y='Fe', color='purple', data=dataset).set_title('Boxplot of Type & Fe')
# all are having outliers 

# Scatterplot
sns.scatterplot(x='RI', y='Na', data=num_data).set_title('Scatterplot of RI & Na')
sns.scatterplot(x='RI', y='Mg', data=num_data).set_title('Scatterplot of RI & Mg')
sns.scatterplot(x='RI', y='Al', data=num_data).set_title('Scatterplot of RI & Al')
sns.scatterplot(x='RI', y='Si', data=num_data).set_title('Scatterplot of RI & Si')
sns.scatterplot(x='RI', y='K', data=num_data).set_title('Scatterplot of RI & K')
sns.scatterplot(x='RI', y='Ca', data=num_data).set_title('Scatterplot of RI & Ca')
sns.scatterplot(x='RI', y='Ba', data=num_data).set_title('Scatterplot of RI & Ba')
sns.scatterplot(x='RI', y='Fe', data=num_data).set_title('Scatterplot of RI & Fe')

sns.scatterplot(x='Na', y='Mg', data=num_data).set_title('Scatterplot of Na & Mg')
sns.scatterplot(x='Na', y='Al', data=num_data).set_title('Scatterplot of Na & Al')
sns.scatterplot(x='Na', y='Si', data=num_data).set_title('Scatterplot of Na & Si')
sns.scatterplot(x='Na', y='K', data=num_data).set_title('Scatterplot of Na & K')
sns.scatterplot(x='Na', y='Ca', data=num_data).set_title('Scatterplot of Na & Ca')
sns.scatterplot(x='Na', y='Ba', data=num_data).set_title('Scatterplot of Na & Ba')
sns.scatterplot(x='Na', y='Fe', data=num_data).set_title('Scatterplot of Na & Fe')

sns.scatterplot(x='Mg', y='Al', data=num_data).set_title('Scatterplot of Mg & Al')
sns.scatterplot(x='Mg', y='Si', data=num_data).set_title('Scatterplot of Mg & Si')
sns.scatterplot(x='Mg', y='K', data=num_data).set_title('Scatterplot of Mg & K')
sns.scatterplot(x='Mg', y='Ca', data=num_data).set_title('Scatterplot of Mg & Ca')
sns.scatterplot(x='Mg', y='Ba', data=num_data).set_title('Scatterplot of Mg & Ba')
sns.scatterplot(x='Mg', y='Fe', data=num_data).set_title('Scatterplot of Mg & Fe')

sns.scatterplot(x='Al', y='Si', data=num_data).set_title('Scatterplot of Al & Si')
sns.scatterplot(x='Al', y='K', data=num_data).set_title('Scatterplot of Al & K')
sns.scatterplot(x='Al', y='Ca', data=num_data).set_title('Scatterplot of Al & Ca')
sns.scatterplot(x='Al', y='Ba', data=num_data).set_title('Scatterplot of Al & Ba')
sns.scatterplot(x='Al', y='Fe', data=num_data).set_title('Scatterplot of Al & Fe')

sns.scatterplot(x='Si', y='K', data=num_data).set_title('Scatterplot of Si & K')
sns.scatterplot(x='Si', y='Ca', data=num_data).set_title('Scatterplot of Si & Ca')
sns.scatterplot(x='Si', y='Ba', data=num_data).set_title('Scatterplot of Si & Ba')
sns.scatterplot(x='Si', y='Fe', data=num_data).set_title('Scatterplot of Si & Fe')

sns.scatterplot(x='K', y='Ca', data=num_data).set_title('Scatterplot of K & Ca')
sns.scatterplot(x='K', y='Ba', data=num_data).set_title('Scatterplot of K & Ba')
sns.scatterplot(x='K', y='Fe', data=num_data).set_title('Scatterplot of K & Fe')

sns.scatterplot(x='Ca', y='Ba', data=num_data).set_title('Scatterplot of Ca & Ba')
sns.scatterplot(x='Ca', y='Fe', data=num_data).set_title('Scatterplot of Ca & Fe')

sns.scatterplot(x='Ba', y='Fe', data=num_data).set_title('Scatterplot of Ba & Fe')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='Type')

# Heatmap
corr = dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

#Outliers
from scipy import stats
Z = np.abs(stats.zscore(dataset))
threshold=3
print(np.where(Z>3))
print(Z[105][1])
df_out = dataset[(Z<3).all(axis=1)]
dataset.shape
df_out.shape # 20 rows are removed
df_out.Type.value_counts()

# Normalize the data
from sklearn.preprocessing import normalize
norm_data = normalize(df_out.iloc[:,0:-1])
y_data = pd.Series(df_out['Type'])

# metric features
X = norm_data
Y = df_out.iloc[:,-1].values

''' Target variable have multiple classes. The classes are imbalanced.
I try to splite the data and apply SMOTE but geeting 
"ValueError: Expected n_neighbors <= n_samples,  but n_samples = 5, n_neighbors = 6" 
due to some classes contain less number of observations(rows) after splitting the data.'''

# First apply SMOTE and then Split the data

print("Before OverSampling, counts of label '1': {}".format(sum(Y==1)))
print("Before OverSampling, counts of label '2': {}".format(sum(Y==2)))
print("Before OverSampling, counts of label '3': {}".format(sum(Y==3)))
print("Before OverSampling, counts of label '5': {}".format(sum(Y==5)))
print("Before OverSampling, counts of label '6': {}".format(sum(Y==6)))
print("Before OverSampling, counts of label '7': {}".format(sum(Y==7)))

# correlation matrix before sampling
# Imbalanced DataFrame Correlation
corr = df_out.corr()
import seaborn as sns
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}).set_title("Imbalanced Correlation Matrix")

# SMOTE (Target variable is imbalanced)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
sm_X, sm_Y = sm.fit_sample(X, Y)

sm_X.shape
sm_Y.shape
print("After OverSampling, counts of label'1': {}".format(sum(sm_Y==1)))
print("After OverSampling, counts of label'2': {}".format(sum(sm_Y==2)))
print("After OverSampling, counts of label'3': {}".format(sum(sm_Y==3)))
print("After OverSampling, counts of label'5': {}".format(sum(sm_Y==5)))
print("After OverSampling, counts of label'6': {}".format(sum(sm_Y==6)))
print("After OverSampling, counts of label'7': {}".format(sum(sm_Y==7)))

# Correlation matrix after sampling
sm_data = pd.concat([pd.DataFrame(sm_X), pd.Series(sm_Y)],axis=1)
sm_data.columns = dataset.columns
sns.heatmap(sm_data.corr(), cmap='YlGnBu', annot_kws={'size':30}).set_title("Balanced Correlation Matrix")

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sm_X, sm_Y,test_size=0.25, random_state=0)

# Hyperparameter tunning using GridSearchCV
from sklearn.model_selection import GridSearchCV
# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# parameters
parameters = {'n_neighbors':[3,5,7,9,11,13,15,17,19,21],'weights': ['uniform', 'distance'],'metric':['euclidean', 'manhattan']}
# Classifier
neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters, verbose=1, cv=3, n_jobs=-1) #cv=5
clf.fit(X_train, Y_train)
clf.best_params_
clf.best_estimator_
clf.best_score_

# Predicting test set results
Y_pred = clf.predict(X_test)
confusion_matrix(Y_test,Y_pred)
accuracy_score(Y_test, Y_pred)

# Another method without Hyperparameter tunning

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
accuracy = []
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(X_train,Y_train)
    train_acc = np.mean(neigh.predict(X_train)==Y_train)
    test_acc = np.mean(neigh.predict(X_test)==Y_test)
    accuracy.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in accuracy],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in accuracy],"ro-")
plt.legend(["train","test"])

# In both the methods n_neighbors = 3 is the best and observed from second method k=23 looks good but accuracy decreases.
# Build the KNN classifier with k=3
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3) # [23] & [11]

# Fitting with training data 
neigh.fit(X_train,Y_train)

# train accuracy 
train_acc = np.mean(neigh.predict(X_train)==Y_train)
train_acc # 92% [78%] & [83]
# test accuracy
test_acc = np.mean(neigh.predict(X_test)==Y_test)
test_acc # 86% [77%] & [81]
