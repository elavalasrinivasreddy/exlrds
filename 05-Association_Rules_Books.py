# Reset console
%reset -f

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Import Dataset
dataset = pd.read_csv('book.csv') # dataset has transaction format data for association rules

X = dataset.copy()

'''   Support = 0.001   '''

frequent_itemsets = apriori(X, min_support=0.001, use_colnames=True) # 2047 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 173042 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.004   '''

frequent_itemsets = apriori(X, min_support=0.004, use_colnames=True) # 1243 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 51518 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.005   '''

frequent_itemsets = apriori(X, min_support=0.005, use_colnames=True) # 1062 itemsets
# Most frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending =False, inplace=True)
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 35298 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.007   '''

frequent_itemsets = apriori(X, min_support=0.007, use_colnames=True) # 830 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 21118 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.009   '''

frequent_itemsets = apriori(X, min_support=0.009, use_colnames=True) # 677 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 14080 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.04   '''

frequent_itemsets = apriori(X, min_support=0.04, use_colnames=True) # 133 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 1066 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.05   '''

frequent_itemsets = apriori(X, min_support=0.05, use_colnames=True) # 100 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 662 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.07   '''

frequent_itemsets = apriori(X, min_support=0.07, use_colnames=True) # 66 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 306 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)

'''   Support = 0.09   '''

frequent_itemsets = apriori(X, min_support=0.09, use_colnames=True) # 47 itemsets
# Most frequent item sets based on support
frequent_itemsets.sort_values('support', ascending=False, inplace=True) 
# Barplot of frequent item sets
plt.bar(x=list(range(1,11)), height=frequent_itemsets.support[1:11], color='rgmyk')
plt.xticks(list(range(1,11)), frequent_itemsets.itemsets[1:11], rotation=45);plt.xlabel('item_sets');plt.ylabel('support')
# Rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1) # 144 rules
rules.head(10)
rules.sort_values('confidence',ascending =False, inplace=True)
rules.head(10)
