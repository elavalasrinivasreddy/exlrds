#Reseting the Console
%reset -f

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
file = 'EastWestAirlines.xlsx'  # Assign spreadsheet filename to file
xl = pd.ExcelFile(file)  # Load spreadsheet
print(xl.sheet_names)    # Print the sheet names
dataset = xl.parse('data')
data_description = xl.parse('Description')

dataset.head()
dataset.shape
dataset.info()
dataset.columns

dataset.describe()

dataset.isnull().sum() # No missing values
dataset.drop_duplicates(keep='first', inplace=True)

dataset.drop('ID#', axis=1, inplace=True)
dataset['Award?'].value_counts()  # 1 = Not Null and 0 = Null
dataset.rename(columns={'Award?':'Award'}, inplace=True)

# Measures of Dispersion
np.var(dataset)
np.std(dataset)

# skewness and kurtosis
from scipy.stats import skew, kurtosis
skew(dataset)
kurtosis(dataset)

# Histogram
plt.hist(dataset.Balance);plt.title('Histogram of Balance');plt.xlabel('Balance');plt.ylabel('Frequency')
plt.hist(dataset.Qual_miles, color='coral');plt.title('Histogram of Qual_miles');plt.xlabel('Qual_miles');plt.ylabel('Frequency')
plt.hist(dataset.cc1_miles, color='skyblue');plt.title('Histogram of cc1_miles');plt.xlabel('cc1_miles');plt.ylabel('Frequency')
plt.hist(dataset.cc2_miles, color='orange');plt.title('Histogram of cc2_miles');plt.xlabel('cc2_miles');plt.ylabel('Frequency')
plt.hist(dataset.cc3_miles, color='lightblue');plt.title('Histogram of cc3_miles');plt.xlabel('cc3_miles');plt.ylabel('Frequency')
plt.hist(dataset.Bonus_miles, color='brown');plt.title('Histogram of Bonus_miles');plt.xlabel('Bonus_miles');plt.ylabel('Frequency')
plt.hist(dataset.Bonus_trans, color='violet');plt.title('Histogram of Bonus_trans');plt.xlabel('Bonus_trans');plt.ylabel('Frequency')
plt.hist(dataset.Flight_miles_12mo, color='lightgreen');plt.title('Histogram of Flight_miles_12mo');plt.xlabel('Flight_miles_12mo');plt.ylabel('Frequency')
plt.hist(dataset.Flight_trans_12, color='purple');plt.title('Histogram of Flight_trans_12');plt.xlabel('Flight_trans_12');plt.ylabel('Frequency')
plt.hist(dataset.Days_since_enroll, color='teal');plt.title('Histogram of Days_since_enroll');plt.xlabel('Days_since_enroll');plt.ylabel('Frequency')

# Barplot
import seaborn as sns
sns.countplot(dataset['Award']).set_title('Countplot of Award')

# Normal Q-Q plot
plt.plot(dataset);plt.legend(['Balance','Qual_miles','cc1_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12','Days_since_enroll','Award'])

bal = np.array(dataset['Balance'])
qm = np.array(dataset['Qual_miles'])
c1m = np.array(dataset['cc1_miles'])
c2m = np.array(dataset['cc2_miles'])
c3m = np.array(dataset['cc3_miles'])
bm = np.array(dataset['Bonus_miles'])
bt = np.array(dataset['Bonus_trans'])
fm = np.array(dataset['Flight_miles_12mo'])
ft = np.array(dataset['Flight_trans_12'])
dse = np.array(dataset['Days_since_enroll'])

from scipy import stats
stats.probplot(bal, dist='norm', plot=plt);plt.title('Q-Q plot of Balance')
stats.probplot(qm, dist='norm', plot=plt);plt.title('Q-Q plot of Qual_miles')
stats.probplot(c1m, dist='norm', plot=plt);plt.title('Q-Q plot of cc1_miles')
stats.probplot(c2m, dist='norm', plot=plt);plt.title('Q-Q plot of cc2_miles')
stats.probplot(c3m, dist='norm', plot=plt);plt.title('Q-Q plot of cc3_miles')
stats.probplot(bm, dist='norm', plot=plt);plt.title('Q-Q plot of Bonus_miles')
stats.probplot(bt, dist='norm', plot=plt);plt.title('Q-Q plot of Bonus_trans')
stats.probplot(fm, dist='norm', plot=plt);plt.title('Q-Q plot of Flight_miles_12mo')
stats.probplot(ft, dist='norm', plot=plt);plt.title('Q-Q plot of Flight_trans_12')
stats.probplot(dse, dist='norm', plot=plt);plt.title('Q-Q plot of Days_since_enroll')

# Normal Probability Distribution
x_bal = np.linspace(np.min(bal), np.max(bal))
y_bal = stats.norm.pdf(x_bal, np.mean(x_bal), np.std(x_bal))
plt.plot(x_bal, y_bal);plt.xlim(np.min(bal), np.max(bal));plt.title('Normal Probability Distribution of Balance');plt.xlabel('Balance');plt.ylabel('Probability')

x_qm = np.linspace(np.min(qm), np.max(qm))
y_qm = stats.norm.pdf(x_qm, np.mean(x_qm), np.std(x_qm))
plt.plot(x_qm, y_qm);plt.xlim(np.min(qm), np.max(qm));plt.title('Normal Probability Distribution of Qual_miles');plt.xlabel('Qual_miles');plt.ylabel('Probability')

x_c1m = np.linspace(np.min(c1m), np.max(c1m))
y_c1m = stats.norm.pdf(x_c1m, np.mean(x_c1m), np.std(x_c1m))
plt.plot(x_c1m, y_c1m);plt.xlim(np.min(c1m), np.max(c1m));plt.title('Normal Probability Distribution of cc1_miles');plt.xlabel('cc1_miles');plt.ylabel('Probability')

x_c2m = np.linspace(np.min(c2m), np.max(c2m))
y_c2m = stats.norm.pdf(x_c2m, np.mean(x_c2m), np.std(x_c2m))
plt.plot(x_c2m, y_c2m);plt.xlim(np.min(c2m), np.max(c2m));plt.title('Normal Probability Distribution of cc2_miles');plt.xlabel('cc2_miles');plt.ylabel('Probability')

x_c3m = np.linspace(np.min(c3m), np.max(c3m))
y_c3m = stats.norm.pdf(x_c3m, np.mean(x_c3m), np.std(x_c3m))
plt.plot(x_c3m, y_c3m);plt.xlim(np.min(c3m), np.max(c3m));plt.title('Normal Probability Distribution of cc3_miles');plt.xlabel('cc3_miles');plt.ylabel('Probability')

x_bm = np.linspace(np.min(bm), np.max(bm))
y_bm = stats.norm.pdf(x_bm, np.mean(x_bm), np.std(x_bm))
plt.plot(x_bm, y_bm);plt.xlim(np.min(bm), np.max(bm));plt.title('Normal Probability Distribution of Bonus_miles');plt.xlabel('Bonus_miles');plt.ylabel('Probability')

x_bt = np.linspace(np.min(bt), np.max(bt))
y_bt = stats.norm.pdf(x_bt, np.mean(x_bt), np.std(x_bt))
plt.plot(x_bt, y_bt);plt.xlim(np.min(bt), np.max(bt));plt.title('Normal Probability Distribution of Bonus_trans');plt.xlabel('Bonus_trans');plt.ylabel('Probability')

x_fm = np.linspace(np.min(fm), np.max(fm))
y_fm = stats.norm.pdf(x_fm, np.mean(x_fm), np.std(x_fm))
plt.plot(x_fm, y_fm);plt.xlim(np.min(fm), np.max(fm));plt.title('Normal Probability Distribution of Flight_miles_12mo');plt.xlabel('Flight_miles_12mo');plt.ylabel('Probability')

x_ft = np.linspace(np.min(ft), np.max(ft))
y_ft = stats.norm.pdf(x_ft, np.mean(x_ft), np.std(x_ft))
plt.plot(x_ft, y_ft);plt.xlim(np.min(ft), np.max(ft));plt.title('Normal Probability Distribution of Flight_trans_12');plt.xlabel('Flight_trans_12');plt.ylabel('Probability')

x_dse = np.linspace(np.min(x_dse), np.max(x_dse))
y_dse = stats.norm.pdf(x_dse, np.mean(x_dse), np.std(x_dse))
plt.plot(x_dse, y_dse);plt.xlim(np.min(dse), np.max(dse));plt.title('Normal Probability Distribution of Days_since_enroll');plt.xlabel('Days_since_enroll');plt.ylabel('Probability')

# Boxplot
import seaborn as sns
sns.boxplot(dataset.Balance, orient='v').set_title('Boxplot of Balance')
sns.boxplot(dataset.Qual_miles, orient='v', color='coral').set_title('Boxplot of Qual_miles')
sns.boxplot(dataset.cc1_miles, orient='v', color='skyblue').set_title('Boxplot of cc1_miles') # No Outliers
sns.boxplot(dataset.cc2_miles, orient='v', color='brown').set_title('Boxplot of cc2_miles')
sns.boxplot(dataset.cc3_miles, orient='v', color='violet').set_title('Boxplot of cc3_miles')
sns.boxplot(dataset.Bonus_miles, orient='v', color='lightblue').set_title('Boxplot of Bonus_miles')
sns.boxplot(dataset.Bonus_trans, orient='v', color='orange').set_title('Boxplot of Bonus_trans')
sns.boxplot(dataset.Flight_miles_12mo, orient='v', color='lightgreen').set_title('Boxplot of Flight_miles_12mo')
sns.boxplot(dataset.Flight_trans_12, orient='v', color='teal').set_title('Boxplot of Flight_trans_12')
sns.boxplot(dataset.Days_since_enroll, orient='v', color='olive').set_title('Boxplot of Days_since_enroll') # No Outliers

# Scatterplot
sns.scatterplot(x='Balance', y='Qual_miles', data=dataset).set_title('Scatterplot of Balance & Qual_miles')
sns.scatterplot(x='Balance', y='cc1_miles', data=dataset).set_title('Scatterplot of Balance & cc1_miles')
sns.scatterplot(x='Balance', y='cc2_miles', data=dataset).set_title('Scatterplot of Balance & cc2_miles')
sns.scatterplot(x='Balance', y='cc3_miles', data=dataset).set_title('Scatterplot of Balance & cc3_miles')
sns.scatterplot(x='Balance', y='Bonus_miles', data=dataset).set_title('Scatterplot of Balance & Bonus_miles')
sns.scatterplot(x='Balance', y='Bonus_trans', data=dataset).set_title('Scatterplot of Balance & Bonus_trans')
sns.scatterplot(x='Balance', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of Balance & Flight_miles_12mo')
sns.scatterplot(x='Balance', y='Flight_trans_12', data=dataset).set_title('Scatterplot of Balance & Flight_trans_12')
sns.scatterplot(x='Balance', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Balance & Days_since_enroll')

sns.scatterplot(x='Qual_miles', y='cc1_miles', data=dataset).set_title('Scatterplot of Qual_miles & cc1_miles')
sns.scatterplot(x='Qual_miles', y='cc2_miles', data=dataset).set_title('Scatterplot of Qual_miles & cc2_miles')
sns.scatterplot(x='Qual_miles', y='cc3_miles', data=dataset).set_title('Scatterplot of Qual_miles & cc3_miles')
sns.scatterplot(x='Qual_miles', y='Bonus_miles', data=dataset).set_title('Scatterplot of Qual_miles & Bonus_miles')
sns.scatterplot(x='Qual_miles', y='Bonus_trans', data=dataset).set_title('Scatterplot of Qual_miles & Bonus_trans')
sns.scatterplot(x='Qual_miles', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of Qual_miles & Flight_miles_12mo')
sns.scatterplot(x='Qual_miles', y='Flight_trans_12', data=dataset).set_title('Scatterplot of Qual_miles & Flight_trans_12')
sns.scatterplot(x='Qual_miles', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Qual_miles & Days_since_enroll')

sns.scatterplot(x='cc1_miles', y='cc2_miles', data=dataset).set_title('Scatterplot of cc1_miles & cc2_miles')
sns.scatterplot(x='cc1_miles', y='cc3_miles', data=dataset).set_title('Scatterplot of cc1_miles & cc3_miles')
sns.scatterplot(x='cc1_miles', y='Bonus_miles', data=dataset).set_title('Scatterplot of cc1_miles & Bonus_miles')
sns.scatterplot(x='cc1_miles', y='Bonus_trans', data=dataset).set_title('Scatterplot of cc1_miles & Bonus_trans')
sns.scatterplot(x='cc1_miles', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of cc1_miles & Flight_miles_12mo')
sns.scatterplot(x='cc1_miles', y='Flight_trans_12', data=dataset).set_title('Scatterplot of cc1_miles & Flight_trans_12')
sns.scatterplot(x='cc1_miles', y='Days_since_enroll', data=dataset).set_title('Scatterplot of cc1_miles & Days_since_enroll')

sns.scatterplot(x='cc2_miles', y='cc3_miles', data=dataset).set_title('Scatterplot of cc2_miles & cc3_miles')
sns.scatterplot(x='cc2_miles', y='Bonus_miles', data=dataset).set_title('Scatterplot of cc2_miles & Bonus_miles')
sns.scatterplot(x='cc2_miles', y='Bonus_trans', data=dataset).set_title('Scatterplot of cc2_miles & Bonus_trans')
sns.scatterplot(x='cc2_miles', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of cc2_miles & Flight_miles_12mo')
sns.scatterplot(x='cc2_miles', y='Flight_trans_12', data=dataset).set_title('Scatterplot of cc2_miles & Flight_trans_12')
sns.scatterplot(x='cc2_miles', y='Days_since_enroll', data=dataset).set_title('Scatterplot of cc2_miles & Days_since_enroll')

sns.scatterplot(x='cc3_miles', y='Bonus_miles', data=dataset).set_title('Scatterplot of cc3_miles & Bonus_miles')
sns.scatterplot(x='cc3_miles', y='Bonus_trans', data=dataset).set_title('Scatterplot of cc3_miles & Bonus_trans')
sns.scatterplot(x='cc3_miles', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of cc3_miles & Flight_miles_12mo')
sns.scatterplot(x='cc3_miles', y='Flight_trans_12', data=dataset).set_title('Scatterplot of cc3_miles & Flight_trans_12')
sns.scatterplot(x='cc3_miles', y='Days_since_enroll', data=dataset).set_title('Scatterplot of cc3_miles & Days_since_enroll')

sns.scatterplot(x='Bonus_miles', y='Bonus_trans', data=dataset).set_title('Scatterplot of Bonus_miles & Bonus_trans')
sns.scatterplot(x='Bonus_miles', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of Bonus_miles & Flight_miles_12mo')
sns.scatterplot(x='Bonus_miles', y='Flight_trans_12', data=dataset).set_title('Scatterplot of Bonus_miles & Flight_trans_12')
sns.scatterplot(x='Bonus_miles', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Bonus_miles & Days_since_enroll')

sns.scatterplot(x='Bonus_trans', y='Flight_miles_12mo', data=dataset).set_title('Scatterplot of Bonus_trans & Flight_miles_12mo')
sns.scatterplot(x='Bonus_trans', y='Flight_trans_12', data=dataset).set_title('Scatterplot of Bonus_trans & Flight_trans_12')
sns.scatterplot(x='Bonus_trans', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Bonus_trans & Days_since_enroll')

sns.scatterplot(x='Flight_miles_12mo', y='Flight_trans_12', data=dataset).set_title('Scatterplot of Flight_miles_12mo & Flight_trans_12')
sns.scatterplot(x='Flight_miles_12mo', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Flight_miles_12mo & Days_since_enroll')

sns.scatterplot(x='Flight_trans_12', y='Days_since_enroll', data=dataset).set_title('Scatterplot of Flight_trans_12 & Days_since_enroll')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='Award')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Find the Outliers Using Z_Score
from scipy import stats
Z = np.abs(stats.zscore(dataset))
threshold = 3
print(np.where(Z>3))
print(Z[3897][1])

# Removing Outliers
df_out = dataset[(Z<3).all(axis=1)]
dataset.shape
df_out.shape

# Normalize the data
from sklearn.preprocessing import normalize
norm_data = normalize(df_out)

# Using Dendrogram to find the optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(df_out, method = 'ward'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distances')

# Fitting Hierarchical Clustering to the Airlines dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=6, affinity = 'euclidean', linkage='ward')
Y_hc = hc.fit_predict(df_out)

#cluster_labels=pd.Series(hc.labels_)
df_out['clust']=hc.labels_ # creating a  new column and assigning it to new column 
df_out = df_out.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
df_out.head()
df_out.clust.value_counts()

# getting aggregate mean of each cluster
aggregate = dataset.iloc[:,2:].groupby(df_out.clust).median()

# creating a csv file 
dataset.to_csv("EastWestAirline_HC.csv",encoding="utf-8")
aggregate.to_csv('EastWestAirline_Agg.csv', encoding='utf-8')
