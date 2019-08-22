# Reset the console
%reset -f

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('wine.csv')
dataset.shape
dataset.head()

dataset.isnull().sum() # No missing values
dataset.Type.value_counts()

dataset.describe()
# Measures of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset)
kurtosis(dataset)

# Histogram
plt.hist(dataset['Alcohol']);plt.title('Histogram of Alcohol');plt.xlabel('Alcohol');plt.ylabel('Frequency')
plt.hist(dataset['Malic'], color='Coral');plt.title('Histogram of Malic');plt.xlabel('Malic');plt.ylabel('Frequency')
plt.hist(dataset['Ash'], color='skyblue');plt.title('Histogram of Ash');plt.xlabel('Ash');plt.ylabel('Frequency')
plt.hist(dataset['Alcalinity'], color='orange');plt.title('Histogram of Alcalinity');plt.xlabel('Alcalinity');plt.ylabel('Frequency')
plt.hist(dataset['Magnesium'], color='brown');plt.title('Histogram of Magnesium');plt.xlabel('Magnesium');plt.ylabel('Frequency')
plt.hist(dataset['Phenols'], color='violet');plt.title('Histogram of Phenols');plt.xlabel('Phenols');plt.ylabel('Frequency')
plt.hist(dataset['Flavanoids'], color='lightgreen');plt.title('Histogram of Flavanoids');plt.xlabel('Flavanoids');plt.ylabel('Frequency')
plt.hist(dataset['Nonflavanoids'], color='lightblue');plt.title('Histogram of Nonflavanoids');plt.xlabel('Nonflavanoids');plt.ylabel('Frequency')
plt.hist(dataset['Proanthocyanins'], color='teal');plt.title('Histogram of Proanthocyanins');plt.xlabel('Proanthocyanins');plt.ylabel('Frequency')
plt.hist(dataset['Color'], color='navy');plt.title('Histogram of Color');plt.xlabel('Color');plt.ylabel('Frequency')
plt.hist(dataset['Hue'], color='gold');plt.title('Histogram of Hue');plt.xlabel('Hue');plt.ylabel('Frequency')
plt.hist(dataset['Dilution'], color='black');plt.title('Histogram of Dilution');plt.xlabel('Dilution');plt.ylabel('Frequency')
plt.hist(dataset['Proline'], color='purple');plt.title('Histogram of Proline');plt.xlabel('Proline');plt.ylabel('Frequency')

# Normal Q-Q plot
plt.plot(dataset);plt.legend(['Type','Alcohol','Malic','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline'])

alchol = np.array(dataset['Alcohol'])
malic = np.array(dataset['Malic'])
ash = np.array(dataset['Ash'])
alcalinity = np.array(dataset['Alcalinity'])
magnesium = np.array(dataset['Magnesium'])
phenols = np.array(dataset['Phenols'])
flavanoids = np.array(dataset['Flavanoids'])
nonflavanoids = np.array(dataset['Nonflavanoids'])
proantho = np.array(dataset['Proanthocyanins'])
color = np.array(dataset['Color'])
hue = np.array(dataset['Hue'])
dilution = np.array(dataset['Dilution'])
proline = np.array(dataset['Proline'])

from scipy import stats
stats.probplot(alchol, dist='norm', plot=plt);plt.title('Q-Q plot of Alcohol')
stats.probplot(malic, dist='norm', plot=plt);plt.title('Q-Q plot of Malic')
stats.probplot(ash, dist='norm', plot=plt);plt.title('Q-Q plot of Ash')
stats.probplot(alcalinity, dist='norm', plot=plt);plt.title('Q-Q plot of Alcalinity')
stats.probplot(magnesium, dist='norm', plot=plt);plt.title('Q-Q plot of Magnesium')
stats.probplot(phenols, dist='norm', plot=plt);plt.title('Q-Q plot of Phenols')
stats.probplot(flavanoids, dist='norm', plot=plt);plt.title('Q-Q plot of Flavanoids')
stats.probplot(nonflavanoids, dist='norm', plot=plt);plt.title('Q-Q plot of Nonflavanoids')
stats.probplot(proantho, dist='norm', plot=plt);plt.title('Q-Q plot of Proanthocyanins')
stats.probplot(color, dist='norm', plot=plt);plt.title('Q-Q plot of Color')
stats.probplot(hue, dist='norm', plot=plt);plt.title('Q-Q plot of Hue')
stats.probplot(dilution, dist='norm', plot=plt);plt.title('Q-Q plot of Dilution')
stats.probplot(proline, dist='norm', plot=plt);plt.title('Q-Q plot of Proline')

# Normal Probability Distribution
x_alchol = np.linspace(np.min(alchol), np.max(alchol))
y_alchol = stats.norm.pdf(x_alchol, np.median(alchol), np.std(alchol))
plt.plot(x_alchol, y_alchol);plt.xlim(np.min(alchol), np.max(alchol));plt.title('Normal Probability Distribution of Alcohol');plt.xlabel('Alcohol');plt.ylabel('Probability')

x_malic = np.linspace(np.min(malic), np.max(malic))
y_malic = stats.norm.pdf(x_malic, np.median(malic), np.std(malic))
plt.plot(x_malic, y_malic);plt.xlim(np.min(malic), np.max(malic));plt.title('Normal Probability Distribution of Malic');plt.xlabel('Malic');plt.ylabel('Probability')

x_ash = np.linspace(np.min(ash), np.max(ash))
y_ash = stats.norm.pdf(x_ash, np.median(ash), np.std(ash))
plt.plot(x_ash, y_ash);plt.xlim(np.min(ash), np.max(ash));plt.title('Normal Probability Distribution of Ash');plt.xlabel('Ash');plt.ylabel('Probability')

x_alcalinity = np.linspace(np.min(alcalinity), np.max(alcalinity))
y_alcalinity = stats.norm.pdf(x_alcalinity, np.median(alcalinity), np.std(alcalinity))
plt.plot(x_alcalinity, y_alcalinity);plt.xlim(np.min(alcalinity), np.max(alcalinity));plt.title('Normal Probability Distribution of Alcalinity');plt.xlabel('Alcalinity');plt.ylabel('Probability')

x_magnesium = np.linspace(np.min(magnesium), np.max(magnesium))
y_magnesium = stats.norm.pdf(x_magnesium, np.median(magnesium), np.std(magnesium))
plt.plot(x_magnesium, y_magnesium);plt.xlim(np.min(magnesium), np.max(magnesium));plt.title('Normal Probability Distribution of Magnesium');plt.xlabel('Magnesium');plt.ylabel('Probability')

x_phenols = np.linspace(np.min(phenols), np.max(phenols))
y_phenols = stats.norm.pdf(x_phenols, np.median(phenols), np.std(phenols))
plt.plot(x_phenols, y_phenols);plt.xlim(np.min(phenols), np.max(phenols));plt.title('Normal Probability Distribution of Phenols');plt.xlabel('Phenols');plt.ylabel('Probability')

x_flavanoids = np.linspace(np.min(flavanoids), np.max(flavanoids))
y_flavanoids = stats.norm.pdf(x_flavanoids, np.median(flavanoids), np.std(flavanoids))
plt.plot(x_flavanoids, y_flavanoids);plt.xlim(np.min(flavanoids), np.max(flavanoids));plt.title('Normal Probability Distribution of Flavanoids');plt.xlabel('Flavanoids');plt.ylabel('Probability')

x_nonflavanoids = np.linspace(np.min(nonflavanoids), np.max(nonflavanoids))
y_nonflavanoids = stats.norm.pdf(x_nonflavanoids, np.median(nonflavanoids), np.std(nonflavanoids))
plt.plot(x_nonflavanoids, y_nonflavanoids);plt.xlim(np.min(nonflavanoids), np.max(nonflavanoids));plt.title('Normal Probability Distribution of Nonflavanoids');plt.xlabel('Nonflavanoids');plt.ylabel('Probability')

x_proantho = np.linspace(np.min(proantho), np.max(proantho))
y_proantho = stats.norm.pdf(x_proantho, np.median(proantho), np.std(proantho))
plt.plot(x_proantho, y_proantho);plt.xlim(np.min(proantho), np.max(proantho));plt.title('Normal Probability Distribution of Proanthocyanins');plt.xlabel('Proanthocyanins');plt.ylabel('Probability')

x_color = np.linspace(np.min(color), np.max(color))
y_color = stats.norm.pdf(x_color, np.median(color), np.std(color))
plt.plot(x_color, y_color);plt.xlim(np.min(color), np.max(color));plt.title('Normal Probability Distribution of Color');plt.xlabel('Color');plt.ylabel('Probability')

x_hue = np.linspace(np.min(hue), np.max(hue))
y_hue = stats.norm.pdf(x_hue, np.median(hue), np.std(hue))
plt.plot(x_hue, y_hue);plt.xlim(np.min(hue), np.max(hue));plt.title('Normal Probability Distribution of Hue');plt.xlabel('Hue');plt.ylabel('Probability')

x_dilution = np.linspace(np.min(dilution), np.max(dilution))
y_dilution = stats.norm.pdf(x_dilution, np.median(dilution), np.std(dilution))
plt.plot(x_dilution, y_dilution);plt.xlim(np.min(dilution), np.max(dilution));plt.title('Normal Probability Distribution of Dilution');plt.xlabel('Dilution');plt.ylabel('Probability')

x_proline = np.linspace(np.min(proline), np.max(proline))
y_proline = stats.norm.pdf(x_proline, np.median(proline), np.std(proline))
plt.plot(x_proline, y_proline);plt.xlim(np.min(proline), np.max(proline));plt.title('Normal Probability Distribution of Proline');plt.xlabel('Proline');plt.ylabel('Probability')

# scatterplot of numerical data wrt categorical data
import seaborn as sns
sns.boxplot(x='Type', y='Alcohol', data=dataset).set_title('scatterplot of Type & Alcohol')
sns.boxplot(x='Type', y='Malic', data=dataset).set_title('scatterplot of Type & Malic')
sns.boxplot(x='Type', y='Ash', data=dataset).set_title('Boxplot of Type & Ash')
sns.boxplot(x='Type', y='Alcalinity', data=dataset).set_title('Boxplot of Type & Alcalinity')
sns.boxplot(x='Type', y='Magnesium', data=dataset).set_title('Boxplot of Type & Magnesium')
sns.boxplot(x='Type', y='Phenols', data=dataset).set_title('Boxplot of Type & Phenols')
sns.boxplot(x='Type', y='Flavanoids', data=dataset).set_title('Boxplot of Type & Flavanoids')
sns.boxplot(x='Type', y='Nonflavanoids', data=dataset).set_title('Boxplot of Type & Nonflavanoids')
sns.boxplot(x='Type', y='Proanthocyanins', data=dataset).set_title('Boxplot of Type & Proanthocyanins')
sns.boxplot(x='Type', y='Color', data=dataset).set_title('Boxplot of Type & Color')
sns.boxplot(x='Type', y='Hue', data=dataset).set_title('Boxplot of Type & Hue')
sns.boxplot(x='Type', y='Dilution', data=dataset).set_title('Boxplot of Type & Dilution')
sns.boxplot(x='Type', y='Proline', data=dataset).set_title('Boxplot of Type & Proline')

# Boxplot of numerical data
sns.boxplot(x='Alcohol', y='Malic', data=dataset).set_title('Boxplot of Alcohol & Malic')
sns.boxplot(x='Alcohol', y='Ash', data=dataset).set_title('Boxplot of Alcohol & Ash')
sns.boxplot(x='Alcohol', y='Alcalinity', data=dataset).set_title('Boxplot of Alcohol & Alcalinity')
sns.boxplot(x='Alcohol', y='Magnesium', data=dataset).set_title('Boxplot of Alcohol & Magnesium')
sns.boxplot(x='Alcohol', y='Phenols', data=dataset).set_title('Boxplot of Alcohol & Phenols')
sns.boxplot(x='Alcohol', y='Flavanoids', data=dataset).set_title('Boxplot of Alcohol & Flavanoids')
sns.boxplot(x='Alcohol', y='Nonflavanoids', data=dataset).set_title('Boxplot of Alcohol & Nonflavanoids')
sns.boxplot(x='Alcohol', y='Proanthocyanins', data=dataset).set_title('Boxplot of Alcohol & Proanthocyanins')
sns.boxplot(x='Alcohol', y='Color', data=dataset).set_title('Boxplot of Alcohol & Color')
sns.boxplot(x='Alcohol', y='Hue', data=dataset).set_title('Boxplot of Alcohol & Hue')
sns.boxplot(x='Alcohol', y='Dilution', data=dataset).set_title('Boxplot of Alcohol & Dilution')
sns.boxplot(x='Alcohol', y='Proline', data=dataset).set_title('Boxplot of Alcohol & Proline')

sns.boxplot(x='Malic', y='Ash', data=dataset).set_title('Boxplot of Malic & Ash')
sns.boxplot(x='Malic', y='Alcalinity', data=dataset).set_title('Boxplot of Malic & Alcalinity')
sns.boxplot(x='Malic', y='Magnesium', data=dataset).set_title('Boxplot of Malic & Magnesium')
sns.boxplot(x='Malic', y='Phenols', data=dataset).set_title('Boxplot of Malic & Phenols')
sns.boxplot(x='Malic', y='Flavanoids', data=dataset).set_title('Boxplot of Malic & Flavanoids')
sns.boxplot(x='Malic', y='Nonflavanoids', data=dataset).set_title('Boxplot of Malic & Nonflavanoids')
sns.boxplot(x='Malic', y='Proanthocyanins', data=dataset).set_title('Boxplot of Malic & Proanthocyanins')
sns.boxplot(x='Malic', y='Color', data=dataset).set_title('Boxplot of Malic & Color')
sns.boxplot(x='Malic', y='Hue', data=dataset).set_title('Boxplot of Malic & Hue')
sns.boxplot(x='Malic', y='Dilution', data=dataset).set_title('Boxplot of Malic & Dilution')
sns.boxplot(x='Malic', y='Proline', data=dataset).set_title('Boxplot of Malic & Proline')

sns.boxplot(x='Ash', y='Alcalinity', data=dataset).set_title('Boxplot of Ash & Alcalinity')
sns.boxplot(x='Ash', y='Magnesium', data=dataset).set_title('Boxplot of Ash & Magnesium')
sns.boxplot(x='Ash', y='Phenols', data=dataset).set_title('Boxplot of Ash & Phenols')
sns.boxplot(x='Ash', y='Flavanoids', data=dataset).set_title('Boxplot of Ash & Flavanoids')
sns.boxplot(x='Ash', y='Nonflavanoids', data=dataset).set_title('Boxplot of Ash & Nonflavanoids')
sns.boxplot(x='Ash', y='Proanthocyanins', data=dataset).set_title('Boxplot of Ash & Proanthocyanins')
sns.boxplot(x='Ash', y='Color', data=dataset).set_title('Boxplot of Ash & Color')
sns.boxplot(x='Ash', y='Hue', data=dataset).set_title('Boxplot of Ash & Hue')
sns.boxplot(x='Ash', y='Dilution', data=dataset).set_title('Boxplot of Ash & Dilution')
sns.boxplot(x='Ash', y='Proline', data=dataset).set_title('Boxplot of Ash & Proline')

sns.boxplot(x='Alcalinity', y='Magnesium', data=dataset).set_title('Boxplot of Alcalinity & Magnesium')
sns.boxplot(x='Alcalinity', y='Phenols', data=dataset).set_title('Boxplot of Alcalinity & Phenols')
sns.boxplot(x='Alcalinity', y='Flavanoids', data=dataset).set_title('Boxplot of Alcalinity & Flavanoids')
sns.boxplot(x='Alcalinity', y='Nonflavanoids', data=dataset).set_title('Boxplot of Alcalinity & Nonflavanoids')
sns.boxplot(x='Alcalinity', y='Proanthocyanins', data=dataset).set_title('Boxplot of Alcalinity & Proanthocyanins')
sns.boxplot(x='Alcalinity', y='Color', data=dataset).set_title('Boxplot of Alcalinity & Color')
sns.boxplot(x='Alcalinity', y='Hue', data=dataset).set_title('Boxplot of Alcalinity & Hue')
sns.boxplot(x='Alcalinity', y='Dilution', data=dataset).set_title('Boxplot of Alcalinity & Dilution')
sns.boxplot(x='Alcalinity', y='Proline', data=dataset).set_title('Boxplot of Alcalinity & Proline')

sns.boxplot(x='Magnesium', y='Phenols', data=dataset).set_title('Boxplot of Magnesium & Phenols')
sns.boxplot(x='Magnesium', y='Flavanoids', data=dataset).set_title('Boxplot of Magnesium & Flavanoids')
sns.boxplot(x='Magnesium', y='Nonflavanoids', data=dataset).set_title('Boxplot of Magnesium & Nonflavanoids')
sns.boxplot(x='Magnesium', y='Proanthocyanins', data=dataset).set_title('Boxplot of Magnesium & Proanthocyanins')
sns.boxplot(x='Magnesium', y='Color', data=dataset).set_title('Boxplot of Magnesium & Color')
sns.boxplot(x='Magnesium', y='Hue', data=dataset).set_title('Boxplot of Magnesium & Hue')
sns.boxplot(x='Magnesium', y='Dilution', data=dataset).set_title('Boxplot of Magnesium & Dilution')
sns.boxplot(x='Magnesium', y='Proline', data=dataset).set_title('Boxplot of Magnesium & Proline')

sns.boxplot(x='Phenols', y='Flavanoids', data=dataset).set_title('Boxplot of Ash & Flavanoids')
sns.boxplot(x='Phenols', y='Nonflavanoids', data=dataset).set_title('Boxplot of Phenols & Nonflavanoids')
sns.boxplot(x='Phenols', y='Proanthocyanins', data=dataset).set_title('Boxplot of Phenols & Proanthocyanins')
sns.boxplot(x='Phenols', y='Color', data=dataset).set_title('Boxplot of Phenols & Color')
sns.boxplot(x='Phenols', y='Hue', data=dataset).set_title('Boxplot of Phenols & Hue')
sns.boxplot(x='Phenols', y='Dilution', data=dataset).set_title('Boxplot of Phenols & Dilution')
sns.boxplot(x='Phenols', y='Proline', data=dataset).set_title('Boxplot of Phenols & Proline')

sns.boxplot(x='Flavanoids', y='Nonflavanoids', data=dataset).set_title('Boxplot of Flavanoids & Nonflavanoids')
sns.boxplot(x='Flavanoids', y='Proanthocyanins', data=dataset).set_title('Boxplot of Flavanoids & Proanthocyanins')
sns.boxplot(x='Flavanoids', y='Color', data=dataset).set_title('Boxplot of Flavanoids & Color')
sns.boxplot(x='Flavanoids', y='Hue', data=dataset).set_title('Boxplot of Flavanoids & Hue')
sns.boxplot(x='Flavanoids', y='Dilution', data=dataset).set_title('Boxplot of Flavanoids & Dilution')
sns.boxplot(x='Flavanoids', y='Proline', data=dataset).set_title('Boxplot of Flavanoids & Proline')

sns.boxplot(x='Nonflavanoids', y='Proanthocyanins', data=dataset).set_title('Boxplot of Nonflavanoids & Proanthocyanins')
sns.boxplot(x='Nonflavanoids', y='Color', data=dataset).set_title('Boxplot of Nonflavanoids & Color')
sns.boxplot(x='Nonflavanoids', y='Hue', data=dataset).set_title('Boxplot of Nonflavanoids & Hue')
sns.boxplot(x='Nonflavanoids', y='Dilution', data=dataset).set_title('Boxplot of Nonflavanoids & Dilution')
sns.boxplot(x='Nonflavanoids', y='Proline', data=dataset).set_title('Boxplot of Nonflavanoids & Proline')

sns.boxplot(x='Proanthocyanins', y='Color', data=dataset).set_title('Boxplot of Proanthocyanins & Color')
sns.boxplot(x='Proanthocyanins', y='Hue', data=dataset).set_title('Boxplot of Proanthocyanins & Hue')
sns.boxplot(x='Proanthocyanins', y='Dilution', data=dataset).set_title('Boxplot of Proanthocyanins & Dilution')
sns.boxplot(x='Proanthocyanins', y='Proline', data=dataset).set_title('Boxplot of Proanthocyanins & Proline')

sns.boxplot(x='Color', y='Hue', data=dataset).set_title('Boxplot of Color & Hue')
sns.boxplot(x='Color', y='Dilution', data=dataset).set_title('Boxplot of Color & Dilution')
sns.boxplot(x='Color', y='Proline', data=dataset).set_title('Boxplot of Color & Proline')

sns.boxplot(x='Hue', y='Dilution', data=dataset).set_title('Boxplot of Hue & Dilution')
sns.boxplot(x='Hue', y='Proline', data=dataset).set_title('Boxplot of Hue & Proline')

sns.boxplot(x='Dilution', y='Proline', data=dataset).set_title('Boxplot of Dilution & Proline')

# Scatterplot

sns.scatterplot(x='Alcohol', y='Malic', data=dataset).set_title('scatterplot of Alcohol & Malic')
sns.scatterplot(x='Alcohol', y='Ash', data=dataset).set_title('scatterplot of Alcohol & Ash')
sns.scatterplot(x='Alcohol', y='Alcalinity', data=dataset).set_title('scatterplot of Alcohol & Alcalinity')
sns.scatterplot(x='Alcohol', y='Magnesium', data=dataset).set_title('scatterplot of Alcohol & Magnesium')
sns.scatterplot(x='Alcohol', y='Phenols', data=dataset).set_title('scatterplot of Alcohol & Phenols')
sns.scatterplot(x='Alcohol', y='Flavanoids', data=dataset).set_title('scatterplot of Alcohol & Flavanoids')
sns.scatterplot(x='Alcohol', y='Nonflavanoids', data=dataset).set_title('scatterplot of Alcohol & Nonflavanoids')
sns.scatterplot(x='Alcohol', y='Proanthocyanins', data=dataset).set_title('scatterplot of Alcohol & Proanthocyanins')
sns.scatterplot(x='Alcohol', y='Color', data=dataset).set_title('scatterplot of Alcohol & Color')
sns.scatterplot(x='Alcohol', y='Hue', data=dataset).set_title('scatterplot of Alcohol & Hue')
sns.scatterplot(x='Alcohol', y='Dilution', data=dataset).set_title('scatterplot of Alcohol & Dilution')
sns.scatterplot(x='Alcohol', y='Proline', data=dataset).set_title('scatterplot of Alcohol & Proline')

sns.scatterplot(x='Malic', y='Ash', data=dataset).set_title('scatterplot of Malic & Ash')
sns.scatterplot(x='Malic', y='Alcalinity', data=dataset).set_title('scatterplot of Malic & Alcalinity')
sns.scatterplot(x='Malic', y='Magnesium', data=dataset).set_title('scatterplot of Malic & Magnesium')
sns.scatterplot(x='Malic', y='Phenols', data=dataset).set_title('scatterplot of Malic & Phenols')
sns.scatterplot(x='Malic', y='Flavanoids', data=dataset).set_title('scatterplot of Malic & Flavanoids')
sns.scatterplot(x='Malic', y='Nonflavanoids', data=dataset).set_title('scatterplot of Malic & Nonflavanoids')
sns.scatterplot(x='Malic', y='Proanthocyanins', data=dataset).set_title('scatterplot of Malic & Proanthocyanins')
sns.scatterplot(x='Malic', y='Color', data=dataset).set_title('scatterplot of Malic & Color')
sns.scatterplot(x='Malic', y='Hue', data=dataset).set_title('scatterplot of Malic & Hue')
sns.scatterplot(x='Malic', y='Dilution', data=dataset).set_title('scatterplot of Malic & Dilution')
sns.scatterplot(x='Malic', y='Proline', data=dataset).set_title('scatterplot of Malic & Proline')

sns.scatterplot(x='Ash', y='Alcalinity', data=dataset).set_title('scatterplot of Ash & Alcalinity')
sns.scatterplot(x='Ash', y='Magnesium', data=dataset).set_title('scatterplot of Ash & Magnesium')
sns.scatterplot(x='Ash', y='Phenols', data=dataset).set_title('scatterplot of Ash & Phenols')
sns.scatterplot(x='Ash', y='Flavanoids', data=dataset).set_title('scatterplot of Ash & Flavanoids')
sns.scatterplot(x='Ash', y='Nonflavanoids', data=dataset).set_title('scatterplot of Ash & Nonflavanoids')
sns.scatterplot(x='Ash', y='Proanthocyanins', data=dataset).set_title('scatterplot of Ash & Proanthocyanins')
sns.scatterplot(x='Ash', y='Color', data=dataset).set_title('scatterplot of Ash & Color')
sns.scatterplot(x='Ash', y='Hue', data=dataset).set_title('scatterplot of Ash & Hue')
sns.scatterplot(x='Ash', y='Dilution', data=dataset).set_title('scatterplot of Ash & Dilution')
sns.scatterplot(x='Ash', y='Proline', data=dataset).set_title('scatterplot of Ash & Proline')

sns.scatterplot(x='Alcalinity', y='Magnesium', data=dataset).set_title('scatterplot of Alcalinity & Magnesium')
sns.scatterplot(x='Alcalinity', y='Phenols', data=dataset).set_title('scatterplot of Alcalinity & Phenols')
sns.scatterplot(x='Alcalinity', y='Flavanoids', data=dataset).set_title('scatterplot of Alcalinity & Flavanoids')
sns.scatterplot(x='Alcalinity', y='Nonflavanoids', data=dataset).set_title('scatterplot of Alcalinity & Nonflavanoids')
sns.scatterplot(x='Alcalinity', y='Proanthocyanins', data=dataset).set_title('scatterplot of Alcalinity & Proanthocyanins')
sns.scatterplot(x='Alcalinity', y='Color', data=dataset).set_title('scatterplot of Alcalinity & Color')
sns.scatterplot(x='Alcalinity', y='Hue', data=dataset).set_title('scatterplot of Alcalinity & Hue')
sns.scatterplot(x='Alcalinity', y='Dilution', data=dataset).set_title('scatterplot of Alcalinity & Dilution')
sns.scatterplot(x='Alcalinity', y='Proline', data=dataset).set_title('scatterplot of Alcalinity & Proline')

sns.scatterplot(x='Magnesium', y='Phenols', data=dataset).set_title('scatterplot of Magnesium & Phenols')
sns.scatterplot(x='Magnesium', y='Flavanoids', data=dataset).set_title('scatterplot of Magnesium & Flavanoids')
sns.scatterplot(x='Magnesium', y='Nonflavanoids', data=dataset).set_title('scatterplot of Magnesium & Nonflavanoids')
sns.scatterplot(x='Magnesium', y='Proanthocyanins', data=dataset).set_title('scatterplot of Magnesium & Proanthocyanins')
sns.scatterplot(x='Magnesium', y='Color', data=dataset).set_title('scatterplot of Magnesium & Color')
sns.scatterplot(x='Magnesium', y='Hue', data=dataset).set_title('scatterplot of Magnesium & Hue')
sns.scatterplot(x='Magnesium', y='Dilution', data=dataset).set_title('scatterplot of Magnesium & Dilution')
sns.scatterplot(x='Magnesium', y='Proline', data=dataset).set_title('scatterplot of Magnesium & Proline')

sns.scatterplot(x='Phenols', y='Flavanoids', data=dataset).set_title('scatterplot of Ash & Flavanoids')
sns.scatterplot(x='Phenols', y='Nonflavanoids', data=dataset).set_title('scatterplot of Phenols & Nonflavanoids')
sns.scatterplot(x='Phenols', y='Proanthocyanins', data=dataset).set_title('scatterplot of Phenols & Proanthocyanins')
sns.scatterplot(x='Phenols', y='Color', data=dataset).set_title('scatterplot of Phenols & Color')
sns.scatterplot(x='Phenols', y='Hue', data=dataset).set_title('scatterplot of Phenols & Hue')
sns.scatterplot(x='Phenols', y='Dilution', data=dataset).set_title('scatterplot of Phenols & Dilution')
sns.scatterplot(x='Phenols', y='Proline', data=dataset).set_title('scatterplot of Phenols & Proline')

sns.scatterplot(x='Flavanoids', y='Nonflavanoids', data=dataset).set_title('scatterplot of Flavanoids & Nonflavanoids')
sns.scatterplot(x='Flavanoids', y='Proanthocyanins', data=dataset).set_title('scatterplot of Flavanoids & Proanthocyanins')
sns.scatterplot(x='Flavanoids', y='Color', data=dataset).set_title('scatterplot of Flavanoids & Color')
sns.scatterplot(x='Flavanoids', y='Hue', data=dataset).set_title('scatterplot of Flavanoids & Hue')
sns.scatterplot(x='Flavanoids', y='Dilution', data=dataset).set_title('scatterplot of Flavanoids & Dilution')
sns.scatterplot(x='Flavanoids', y='Proline', data=dataset).set_title('scatterplot of Flavanoids & Proline')

sns.scatterplot(x='Nonflavanoids', y='Proanthocyanins', data=dataset).set_title('scatterplot of Nonflavanoids & Proanthocyanins')
sns.scatterplot(x='Nonflavanoids', y='Color', data=dataset).set_title('scatterplot of Nonflavanoids & Color')
sns.scatterplot(x='Nonflavanoids', y='Hue', data=dataset).set_title('scatterplot of Nonflavanoids & Hue')
sns.scatterplot(x='Nonflavanoids', y='Dilution', data=dataset).set_title('scatterplot of Nonflavanoids & Dilution')
sns.scatterplot(x='Nonflavanoids', y='Proline', data=dataset).set_title('scatterplot of Nonflavanoids & Proline')

sns.scatterplot(x='Proanthocyanins', y='Color', data=dataset).set_title('scatterplot of Proanthocyanins & Color')
sns.scatterplot(x='Proanthocyanins', y='Hue', data=dataset).set_title('scatterplot of Proanthocyanins & Hue')
sns.scatterplot(x='Proanthocyanins', y='Dilution', data=dataset).set_title('scatterplot of Proanthocyanins & Dilution')
sns.scatterplot(x='Proanthocyanins', y='Proline', data=dataset).set_title('scatterplot of Proanthocyanins & Proline')

sns.scatterplot(x='Color', y='Hue', data=dataset).set_title('scatterplot of Color & Hue')
sns.scatterplot(x='Color', y='Dilution', data=dataset).set_title('scatterplot of Color & Dilution')
sns.scatterplot(x='Color', y='Proline', data=dataset).set_title('scatterplot of Color & Proline')

sns.scatterplot(x='Hue', y='Dilution', data=dataset).set_title('scatterplot of Hue & Dilution')
sns.scatterplot(x='Hue', y='Proline', data=dataset).set_title('scatterplot of Hue & Proline')

sns.scatterplot(x='Dilution', y='Proline', data=dataset).set_title('scatterplot of Dilution & Proline')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='Type')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Consider only numerical data
X = dataset.iloc[:,1:]

#Outliers
from scipy import stats
Z = np.abs(stats.zscore(X))
threshold=3
print(np.where(Z>3))
print(Z[59][2])
df_out = X[(Z<3).all(axis=1)]
X.shape
df_out.shape

# Normalization
from sklearn.preprocessing import normalize
norm_data = normalize(df_out)
norm_data.shape

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 13) # 13 variables in dataset
pca_values = pca.fit_transform(norm_data) # PCA scores

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# Cumulative variance sum
var1 = np.cumsum(np.round(var,decimals=4)*100)
var1

# Plot between PCA1 and PCA2
x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
z = np.array(pca_values[:,2])
plt.scatter(x,y)

# Clustering
k_df = pd.DataFrame(pca_values[:,0:3])

# K-means clustering
# Using the Elbow Method to find optimal no.of clusters
wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
	kmeans.fit(k_df)
	wcss.append(kmeans.inertia_)
# scree plot
plt.plot(range(1,11), wcss, 'ro-');plt.title('The Elbow Method');plt.xlabel('No.of Clusters');plt.ylabel('wcss')

# Apply K-Means to the crime dataset [4 clusters i get]
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(k_df)

# Getting Labels of Clusters assigned to each row
kmeans.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
k_df['clusters'] = kmeans.labels_ # creating a new column and assigning it to new column
k_df.head()

k_df = k_df.iloc[:,[3,0,1,2]]
k_df.head()

k_groups = k_df.iloc[:,1:].groupby(k_df.clusters).median()
k_groups

# H-clustering
h_df = pd.DataFrame(pca_values[:,0:3])

# Using Dendrogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(h_df, method='ward'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distance')
# i get 5 clusters
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(h_df)

# Getting Labels of Clusters assigned to each row
hc.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
h_df['clusters'] = hc.labels_ # creating a new column and assigning it to new column
h_df.head()

h_df = h_df.iloc[:,[3,0,1,2]]

h_groups = h_df.iloc[:,1:].groupby(h_df.clusters).median()
h_groups

# clustering on original data
knorm_data = pd.DataFrame(norm_data)
knorm_data.shape

# K-means clustering
# Using the Elbow Method to find optimal no.of clusters
wcss = []
from sklearn.cluster import KMeans
for i in range(1,11):
	kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
	kmeans.fit(knorm_data)
	wcss.append(kmeans.inertia_)
# scree plot
plt.plot(range(1,11), wcss, 'ro-');plt.title('The Elbow Method');plt.xlabel('No.of Clusters');plt.ylabel('wcss')

# Apply K-Means to the crime dataset [4 clusters i get]
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y1_kmeans = kmeans.fit_predict(knorm_data)

# Getting Labels of Clusters assigned to each row
kmeans.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
knorm_data['clusters'] = kmeans.labels_ # creating a new column and assigning it to new column
knorm_data.head()

knorm_data = knorm_data.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
knorm_data.head()

knorm_groups = knorm_data.iloc[:,1:].groupby(knorm_data.clusters).median()
knorm_groups

# H-clustering
hnorm_data = pd.DataFrame(norm_data)

# Using Dendrogram to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(hnorm_data, method='complete'));plt.title('Dendrogram');plt.xlabel('Observations');plt.ylabel('Euclidean Distance')
# i get 4 clusters
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
y1_hc = hc.fit_predict(hnorm_data)

# Getting Labels of Clusters assigned to each row
hc.labels_

# md = pd.Series(kmeans.labels_) # converting numpy array into pandas series object
hnorm_data['clusters'] = hc.labels_ # creating a new column and assigning it to new column
hnorm_data.head()

#h_df = h_df.iloc[:,[3,0,1,2]]

h_groups = hnorm_data.iloc[:,1:].groupby(hnorm_data.clusters).median()
h_groups
