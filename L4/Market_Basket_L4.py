# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:38:26 2021

@author: tmb1
"""

import pandas as pd

pd. set_option('max_columns', None)
#数据加载
dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)
print(dataset)
print(dataset.shape)#(7501,20)

#Step1:把数据整理成id=>item形式，转换成transaction
transactions = []
#按照行进行遍历
for i in range(0, dataset.shape[0]):
    #记录一行的Transaction
    temp = []
    # 按照列进行遍历
    for j in range(0, dataset.shape[1]):
        if str(dataset.values[i,j]) != 'nan':
            temp.append(dataset.values[i,j])
    print(temp)
    transactions.append(temp)

from efficient_apriori import apriori
	
# Step2:设定关联规则的参数（support,confident）挖掘关联规则,挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.03,  min_confidence=0.3)
print('频繁项集：', itemsets)
print('关联规则：', rules)


# 使用Mlxtend探索购物篮数据的频繁项集和关联规则
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 数据加载后，使用空字节填充NaN
dataset = dataset.fillna('')
dataset.head(5)

# 将每一行数据组合成一列，用逗号分隔
dataset_combined = dataset[0]
for i in range(1, len(dataset.columns)):
    dataset_combined = dataset_combined + '/' + dataset[i]
dataset_combined = pd.DataFrame(dataset_combined)
dataset_combined.columns = ['items']
dataset_combined.head(10)

# 将购物篮数据进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
dataset = dataset_combined['items'].str.get_dummies('/')
#dataset = transaction.str.get_dummies(',')
pd.options.display.max_columns=100
print(dataset.head())


# 挖掘频繁项集，最小支持度为0.02
itemsets = apriori(dataset,use_colnames=True, min_support=0.015)
# 按照支持度从大到小进行时候粗
itemsets = itemsets.sort_values(by="support" , ascending=False) 
print('-'*20, '频繁项集', '-'*20)
print(itemsets)
# 根据频繁项集计算关联规则，设置最小提升度为2
rules =  association_rules(itemsets, metric='lift', min_threshold=1.5)
# 按照提升度从大到小进行排序
rules = rules.sort_values(by="lift" , ascending=False) 
print('-'*30, '关联规则', '-'*30)
print(rules)

'''
#尝试使用FPGrowth算法,后期探索
import fptools as fp

'''
