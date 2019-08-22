# Reset the console
%reset -f

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Import dataset
dataset = pd.read_csv('my_movies.csv')
# dataset have transaction format data for association rules
X = dataset.loc[:,['Sixth Sense', 'Gladiator', 'LOTR1','Harry Potter1','Patriot',
				'LOTR2', 'Harry Potter2', 'LOTR','Braveheart', 'Green Mile']]

'''   Support = 0.005   '''

frequent_itemsets = apriori(X, min_support=0.005, use_colnames=True) # 53 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot for frequent itemsets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 238 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)

'''   Support = 0.05   '''

frequent_itemsets = apriori(X, min_support=0.05, use_colnames=True) # 53 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot for frequent itemsets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 238 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)

'''   Support = 0.1   '''

frequent_itemsets = apriori(X, min_support=0.1, use_colnames=True) # 53 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot for frequent itemsets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 238 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
