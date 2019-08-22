# Reset the Conole
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv('crime_data.csv', index_col=0)
dataset.head()
dataset.shape
dataset.info()
dataset.columns
dataset.isnull().sum() # No missing values
dataset.dtypes

dataset.describe()

# Measures of Dispersion
np.var(dataset)
np.std(dataset)

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset)
kurtosis(dataset)

# Histogram
plt.hist(dataset['Murder']);plt.title('Histogram of Murder');plt.xlabel('Murder');plt.ylabel('Frequency')
plt.hist(dataset['Assault'], color='skyblue');plt.title('Histogram of Assault');plt.xlabel('Assault');plt.ylabel('Frequency')
plt.hist(dataset['UrbanPop'], color='coral');plt.title('Histogram of UrbanPop');plt.xlabel('UrbanPop');plt.ylabel('Frequency')
plt.hist(dataset['Rape'], color='orange');plt.title('Histogram of Rape');plt.xlabel('Rape');plt.ylabel('Frequency')

# Normal Q-Q plot
plt.plot(dataset);plt.legend(['Murder', 'Assault', 'UrbanPop', 'Rape'])

mur = np.array(dataset['Murder'])
ass = np.array(dataset['Assault'])
ubp = np.array(dataset['UrbanPop'])
rape = np.array(dataset['Rape'])

from scipy import stats
stats.probplot(mur, dist='norm', plot=plt);plt.title('Q-Q plot of Murder')
stats.probplot(ass, dist='norm', plot=plt);plt.title('Q-Q plot of Assault')
stats.probplot(ubp, dist='norm', plot=plt);plt.title('Q-Q plot of UrbanPop')
stats.probplot(rape, dist='norm', plot=plt);plt.title('Q-Q plot of Rape')

# Normal Probability Distribution

x_mur = np.linspace(np.min(mur), np.max(mur))
y_mur = stats.norm.pdf(x_mur, np.mean(x_mur), np.std(x_mur))
plt.plot(x_mur, y_mur);plt.xlim(np.min(mur), np.max(mur));plt.title('Normal Probability Distribution of Murder');plt.xlabel('Murder');plt.ylabel('Probability')

x_ass = np.linspace(np.min(ass), np.max(ass))
y_ass = stats.norm.pdf(x_ass, np.mean(x_ass), np.std(x_ass))
plt.plot(x_ass, y_ass);plt.xlim(np.min(ass), np.max(ass));plt.title('Normal Probability Distribution of Assault');plt.xlabel('Assault');plt.ylabel('Probability')

x_ubp = np.linspace(np.min(ubp), np.max(ubp))
y_ubp = stats.norm.pdf(x_ubp, np.mean(x_ubp), np.std(x_ubp))
plt.plot(x_ubp, y_ubp);plt.xlim(np.min(ubp), np.max(ubp));plt.title('Normal Probability Distribution of UrbanPop');plt.xlabel('UrbanPop');plt.ylabel('Probability')

x_rape = np.linspace(np.min(rape), np.max(rape))
y_rape = stats.norm.pdf(x_rape, np.mean(x_rape), np.std(x_rape))
plt.plot(x_rape, y_rape);plt.xlim(np.min(rape), np.max(rape));plt.title('Normal Probability Distribution of Rape');plt.xlabel('Rape');plt.ylabel('Probability')

# Boxplot
import seaborn as sns
sns.boxplot(dataset, orient='v').set_title('Boxplot of Independent Variables')
sns.boxplot(dataset.Murder, orient='v', color='coral').set_title('Boxplot of Murder')
sns.boxplot(dataset.Assault, orient='v', color='skyblue').set_title('Boxplot of Assault')
sns.boxplot(dataset.UrbanPop, orient='v', color='orange').set_title('Boxplot of UrbanPop')
sns.boxplot(dataset.Rape, orient='v', color='brown').set_title('Boxplot of Rape')

# Scatterplot
sns.scatterplot(x='Murder', y='Assault', data=dataset).set_title('Scatterplot of Murder & Assault')
sns.scatterplot(x='Murder', y='UrbanPop', data=dataset).set_title('Scatterplot of Murder & UrbanPop')
sns.scatterplot(x='Murder', y='Rape', data=dataset).set_title('Scatterplot of Murder & Rape')

sns.scatterplot(x ='Assault', y ='UrbanPop', data=dataset).set_title('Scatterplot of Assault & UrbanPop')
sns.scatterplot(x ='Assault', y ='Rape', data = dataset).set_title('Scatterplot of Assault & Rape')

sns.scatterplot(x='UrbanPop', y='Rape', data=dataset).set_title('Scatterplot of UrbanPop & Rape')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Find outliers using Z_Score
from scipy import stats
Z = np.abs(stats.zscore(dataset))
threshold =3
print(np.where(Z>3)) # No outliers

'''# Find Outliers Tukey IQR
def find_outlier_tukey(x):
	Q1 = np.percentile(x, 25)
	Q3 = np.percentile(x, 75)
	IQR = Q3-Q1
	floor = Q1-1.5*IQR
	ceiling = Q3+1.5*IQR
	outlier_indices = list(x.index[(x<floor) | (x>ceiling)])
	outlier_values = list(x[outlier_indices])
	return outlier_indices, outlier_values

outlier_indices, outlier_values = find_outlier_tukey(dataset['Assault'])
print(np.sort(outlier_values)) '''

# Normalize the data
from sklearn.preprocessing import normalize
norm_data = normalize(dataset)

# Using Dendrogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(norm_data, method='ward'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distance')

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(norm_data)

# Getting Labels of Clusters assigned to each row
hc.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
dataset['clusters'] = hc.labels_ # creating a new column and assigning it to new column
dataset.head()

dataset = dataset.iloc[:,[4,0,1,2,3]]

groups = dataset.iloc[:,1:].groupby(dataset.clusters).mean()

dataset.to_csv("kmeans_cluster_crime.csv")
groups.to_csv('final_kmeans_crime.csv')
