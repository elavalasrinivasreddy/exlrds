# Reset console
%reset -f

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Import dataset
dataset = []
# As the file is in transaction data we will be reading data directly
with open('groceries.csv') as file: dataset = file.read()

# Splitting the data into seperate transaction using seperator '\n'
dataset = dataset.split('\n')

dataset_gro = []
for i in dataset:dataset_gro.append(i.split(','))

# Creating Dataframe for the transactions data

# Purpose of converting all list into Series object because to treat each list element 
dataset_series = pd.DataFrame(pd.Series(dataset_gro))
dataset_series = dataset_series.iloc[:9834, :] # Removing last empty transaction

dataset_series.columns = ['transactions']

# Creating a dummy columns for the each item in each transactions...using column names as index
X = dataset_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

'''   Support = 0.001 and max_len = 3   '''

frequent_itemsets = apriori(X, min_support=0.001, max_len=3, use_colnames=True) # 9968 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 45712 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.001 and max_len = 2   '''

frequent_itemsets = apriori(X, min_support=0.001, max_len=2, use_colnames=True) # 3138 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 5426 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.004 and max_len=3   '''

frequent_itemsets = apriori(X, min_support=0.004, max_len=3, use_colnames=True) # 1368 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 4066 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.004 and max_len=2   '''

frequent_itemsets = apriori(X, min_support=0.004, max_len=2, use_colnames=True) # 936 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 1498 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.005 and  max_len= 3   '''

frequent_itemsets = apriori(X, min_support=0.005, max_len=3, use_colnames=True) # 989 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 2698 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.005 and max_len=2   '''

frequent_itemsets = apriori(X, min_support=0.005, max_len=2, use_colnames=True) # 725 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 1130 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.007 and max_len=3   '''

frequent_itemsets = apriori(X, min_support=0.007, max_len=3, use_colnames=True) # 591 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 1348 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.007 and max_len=2   '''

frequent_itemsets = apriori(X, min_support=0.007, max_len=2, use_colnames=True) # 485 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 720 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.009 and max_len=3   '''

frequent_itemsets = apriori(X, min_support=0.009, max_len=3, use_colnames=True) # 404 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 796 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.009 and max_len=2   '''

frequent_itemsets = apriori(X, min_support=0.009, max_len=2, use_colnames=True) # 354 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 496 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.04 and max_len = 3   '''

frequent_itemsets = apriori(X, min_support=0.04, max_len=3, use_colnames=True) # 41 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 16 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.04 and max_len= 2   '''

frequent_itemsets = apriori(X, min_support=0.04, max_len=2, use_colnames=True) # 41 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 16 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.05 and max_len = 3   '''

frequent_itemsets = apriori(X, min_support=0.05, max_len=3, use_colnames=True) # 31 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 6 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.05 and max_len=2   '''

frequent_itemsets = apriori(X, min_support=0.05, max_len=2, use_colnames=True) # 31 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 6 rules are generated
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)