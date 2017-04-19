# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:20:59 2017

@author: zhouyonglong
"""

from __future__ import division
from pyomo.environ import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#result = pd.read_csv('result_40_20_display.csv',header=0,names=['state','strategy'])
result = pd.read_csv('result_100_40_30_20.csv',header=0,names=['state','strategy'])

result = result[result['strategy']=='优惠券']

rewards_summary = pd.read_csv('rewards_summary_100_40_30_20.csv')
#rewards_summary = pd.read_csv('rewards_summary.csv')

rewards_summary = rewards_summary[rewards_summary['action']=='groupon']

rewards = rewards_summary.merge(result[['state']],
                              on='state',
                              how='inner')
rewards = rewards[['state','premium_count','premium_mean',
                   'premium_sum','premium_std']]


rewards_count_dict = dict()
rewards_mean_dict = dict()
rewards_var_dict = dict()


for i in range(0,len(rewards)):
    rewards_count_dict[i+1]=rewards.iloc[i]['premium_count']
    rewards_mean_dict[i+1]=rewards.iloc[i]['premium_mean']
    rewards_var_dict[i+1]=np.square(rewards.iloc[i]['premium_std'])


rewards_sum = rewards['premium_sum'].sum()
rewards_count_sum = rewards['premium_count'].sum()

budget = 20000
cost = 100
targets_total = budget/cost

targets_distribution=dict()
risks = np.arange(0.05,0.55,0.05)
total_returns = []


for risk in risks:
    _lambda = rewards_sum*targets_total/rewards_count_sum*risk
    
    model = ConcreteModel()
    #
    states = np.arange(1,len(rewards)+1)
    model.states = Set(initialize=states)
    
    
    model.rewards_count = Param(model.states,
                                initialize=rewards_count_dict)
    model.rewards_mean = Param(model.states,
                                initialize=rewards_mean_dict)
    model.rewards_var = Param(model.states,
                                initialize=rewards_var_dict)
    
    #
    model.targets = Var(model.states,within=NonNegativeIntegers)
    
    # - _lambda*model.targets[i]*model.rewards_var[i]
    def obj_rule(model):
        return sum(model.targets[i]*model.rewards_mean[i] 
                       for i in states)
    model.obj = Objective(rule=obj_rule,sense=maximize)
    
    
    def targets_rule1(model):
        return sum(model.targets[i] for i in states) <= targets_total
    model.con1 = Constraint(rule=targets_rule1)
    
    
    def targets_rule2(model, state):
        if state == len(rewards)+1:
            return ConstraintList.End
        return model.targets[state] <= model.rewards_count[state]
    model.con2 = ConstraintList(rule=targets_rule2)
    
    
    def targets_rule3(model):
        return sum(model.targets[i]*model.targets[i]*model.rewards_var[i]
                       for i in states)/np.square(targets_total) <= _lambda
    model.con3 = Constraint(rule=targets_rule3)
    
    
    # gurobi
    opt = SolverFactory("gurobi")
    results = opt.solve(model)
    #results.write()
    
    targets = []
    total_return=0
    for (key,value) in model.targets.get_values().items():
        targets.append((rewards.iloc[key-1]['state'],
                        value))
        total_return+=rewards.iloc[key-1]['premium_mean']*value                
    
    total_returns.append(total_return)
    targets_distribution[risk]=targets


plt.plot(risks,total_returns)

plt.axvline(0.10,linestyle='--')
plt.axvline(0.20,linestyle='--')

plt.axhline(total_returns[1],linestyle='--')
plt.axhline(total_returns[3],linestyle='--')

plt.title('风险-回报曲线')
plt.xlabel('风险（标准差）')
plt.ylabel('回报')


targets_1 = targets_distribution[0.1]
targets_1 = pd.DataFrame(targets_1,columns=['客户状态','目标人数'])
targets_1['客户状态'] = targets_1['客户状态'].apply(
    lambda x:'历史保费'+x+'天之内到期')

targets_2 = targets_distribution[0.2]
targets_2 = pd.DataFrame(targets_2,columns=['客户状态','目标人数'])
targets_2['客户状态'] = targets_2['客户状态'].apply(
    lambda x:'历史保费'+x+'天之内到期')

targets = pd.DataFrame()
for item in targets_distribution.items():
    key = str(item[0])
    value = []
    for x in item[1]:
        value.append(x[1])
    targets[key] = value

plt.figure()
for risk in risks:
    #plot rows of data as if they were series data
    row = targets[str(risk)]
    labelColor = (risk-min(risks))/(max(risks)-min(risks))
    row.plot(color=plt.cm.RdYlBu(labelColor), 
             alpha=0.95)

plt.xlabel("客户状态")
plt.ylabel(("目标人数"))
plt.show()

plt.legend(risks)
plt.xticks(range(len(rewards['state'])),
           rewards['state'],rotation=15)













