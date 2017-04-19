

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:31:57 2017

@author: zhouyonglong
"""
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType,StructType,IntegerType,DateType,DoubleType
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import UserDefinedFunction

import pandas as pd
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import itertools
import mdptoolbox
from statsmodels import robust


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mean_deviation(data, axis=None):
    return np.mean(data - np.mean(data, axis), axis)

def mean_deviation2(data, axis=None):
    return np.sum(data - np.mean(data, axis), axis)    
    

def get_transition_rewards(x):
    last_expiry_date = x[1]
    response_date = x[6]
    history_value = x[2]
    groupon = x[4]
    premium = x[5]

    if groupon > 0:
        action='groupon'
    else:
        action='nothing'

    if history_value >= 10000:
        prior_state = '>10000'
    elif history_value >= 4000 and history_value < 10000:
        prior_state = '4000-10000'
    elif history_value >= 3000 and history_value < 4000:    
        prior_state = '3000-4000' 
    elif history_value >= 2000 and history_value < 3000:
        prior_state = '2000-3000'
    else:
        prior_state ='<2000'
         
    
    if response_date:
        advance = (last_expiry_date - response_date).days
    else:
        advance = 0
        
    if advance > 30:
        prior_state += ',30-90'
    elif advance > 20:
        prior_state += ',20-30'
    elif advance > 10:
        prior_state += ',10-20'
    else:
        prior_state += ',1-10'
        
    return (prior_state,action,premium)    
    

    

def create_transition_sequence(x):
    
    last_expiry_date = x[1]
    response_date = x[6]
    history_value = x[2]
    groupon = x[4]
    premium = x[5]
    

    if response_date:
        advance = (last_expiry_date - response_date).days
    else:
        advance = 0

    
    if groupon > 0:
        action='groupon'
        reward = premium
    else:
        action='nothing'
        reward= premium
    
    total = history_value + premium
    default_action = 'nothing'
    reward_before_conversion = -history_value*0.1
    #reward_after_conversion = 0
    churn_reward = -2000
    result = []


    if history_value >= 10000:
        prior_state = '>10000'
        current_state = '>10000'
        
    elif history_value >= 4000 and history_value < 10000 and total > 10000:
        prior_state = '4000-10000'
        current_state = '>10000'
    elif history_value >= 4000 and history_value<10000 and total <= 10000:    
        prior_state = '4000-10000'
        current_state = '4000-10000'
        
    elif history_value>=3000 and history_value<4000 and total > 10000:    
        prior_state = '3000-4000' 
        current_state = '>10000'       
    elif history_value>=3000 and history_value<4000 and total >= 4000:   
        prior_state = '3000-4000'
        current_state = '4000-10000'
    elif history_value>=3000 and history_value<4000 and total < 4000:    
        prior_state = '3000-4000'
        current_state = '3000-4000'
        
    elif history_value>=2000 and history_value<3000 and total > 10000:
        prior_state = '2000-3000'
        current_state = '>10000'
    elif history_value>=2000 and history_value<3000 and total >= 4000:
        prior_state = '2000-3000'
        current_state = '4000-10000'
    elif history_value>=2000 and history_value<3000 and total >= 3000:
        prior_state = '2000-3000'
        current_state = '3000-4000'
    elif history_value>=2000 and history_value<3000 and total < 3000:
        prior_state = '2000-3000'
        current_state = '2000-3000'  
        
    elif history_value<2000 and total >= 10000:
        prior_state = '<2000'
        current_state = '>10000'
    elif history_value<2000 and total >= 4000:
        prior_state = '<2000'
        current_state = '4000-10000'
    elif history_value<2000 and total >= 3000:
        prior_state = '<2000'
        current_state = '3000-4000'
    elif history_value<2000 and total >= 2000:
        prior_state = '<2000'
        current_state = '2000-3000'
    elif history_value<2000 and total<2000:
        prior_state = '<2000'
        current_state = '<2000'
    else:
        prior_state = str(history_value)
        current_state = str(total)
       
        
    if advance > 30:
        result.append((prior_state+',30-90',action,
                       current_state+',0',reward))
#        result.append((current_state+',20-30',default_action,
#                       current_state+',10-20',reward_after_conversion))
#        result.append((current_state+',10-20',default_action,
#                       current_state+',1-10',reward_after_conversion))        
#        result.append((current_state+',1-10',default_action,
#                       current_state+',0',reward_after_conversion))
    elif advance > 20:
        result.append((prior_state+',30-90',default_action,
                       prior_state+',20-30',reward_before_conversion))
        result.append((prior_state+',20-30',action,
                       current_state+',0',reward))
#        result.append((current_state+',10-20',default_action,
#                       current_state+',1-10',reward_after_conversion))         
#        result.append((current_state+',1-10',default_action,
#                       current_state+',0',reward_after_conversion))    
    elif advance > 10:
        result.append((prior_state+',30-90',default_action,
                       prior_state+',20-30',reward_before_conversion))
        result.append((prior_state+',20-30',default_action,
                       prior_state+',10-20',reward_before_conversion))
        result.append((prior_state+',10-20',action,
                       current_state+',0',reward))
#        result.append((current_state+',1-10',default_action,
#                       current_state+',0',reward_after_conversion))
    elif advance > 0:
        result.append((prior_state+',30-90',default_action,
                       prior_state+',20-30',reward_before_conversion))
        result.append((prior_state+',20-30',default_action,
                       prior_state+',10-20',reward_before_conversion))        
        result.append((prior_state+',10-20',default_action,
                       prior_state+',1-10',reward_before_conversion))
        result.append((prior_state+',1-10',action,
                       current_state+',0',reward))
    #
    else:
        result.append((prior_state+',30-90',default_action,
                       prior_state+',20-30',reward_before_conversion))
        result.append((prior_state+',20-30',default_action,
                       prior_state+',10-20',reward_before_conversion))        
        result.append((prior_state+',10-20',default_action,
                       prior_state+',1-10',reward_before_conversion))
        result.append((prior_state+',1-10',default_action,
                       prior_state+',0',churn_reward))

        
    return result
   
        
        
states = []

states.append('<2000,30-90')
states.append('<2000,20-30')
states.append('<2000,10-20')
states.append('<2000,1-10')
#states.append('<2000,0')

states.append('2000-3000,30-90')
states.append('2000-3000,20-30')
states.append('2000-3000,10-20')
states.append('2000-3000,1-10')
#states.append('2000-3000,0')

states.append('3000-4000,30-90')
states.append('3000-4000,20-30')
states.append('3000-4000,10-20')
states.append('3000-4000,1-10')
#states.append('3000-4000,0')

states.append('4000-10000,30-90')
states.append('4000-10000,20-30')
states.append('4000-10000,10-20')
states.append('4000-10000,1-10')
#states.append('4000-10000,0')

states.append('>10000,30-90')
states.append('>10000,20-30')
states.append('>10000,10-20')
states.append('>10000,1-10')
#states.append('>10000,0')



def create_transition_matrix(states_pair):    
    
    num_states = len(states)
    groupon_table = pd.DataFrame(np.zeros((num_states,num_states)),
                                 columns=states,
                                 index=states)
    no_groupon_table = pd.DataFrame(np.zeros((num_states,num_states)),
                                    columns=states,
                                    index=states)
    reward_table = pd.DataFrame(np.zeros((num_states,2)),
                                columns=['groupon_reward',
                                         'no_groupon_reward'],
                                index=states)

    
    for item in states_pair:
        from_state = item[0][0]
        to_state = item[0][1]

        count = item[1][0]
        reward = item[1][1]
        groupon_table.loc[from_state][to_state] = count
        reward = 0 if count==0 else reward/count
        reward_table.loc[from_state]['groupon_reward']+=reward

        count = item[1][2]
        reward = item[1][3]
        no_groupon_table.loc[from_state][to_state] = count
        reward = 0 if count==0 else reward/count
        reward_table.loc[from_state]['no_groupon_reward']+=reward


    #normalize
    for r in range(0,num_states):
        current_state = states[r]

        row_sum = sum([x for x in groupon_table.iloc[r]])
        if row_sum>0:
            for c in range(0,num_states):
                groupon_table.iloc[r][c] = groupon_table.iloc[r][c] / row_sum
        else:
            groupon_table.loc[current_state][current_state] = 1            


        row_sum = sum([x for x in no_groupon_table.iloc[r]])
        if row_sum>0:
            for c in range(0,num_states):
                no_groupon_table.iloc[r][c] = no_groupon_table.iloc[r][c] / row_sum
        else:
            no_groupon_table.loc[current_state][current_state] = 1 

    groupon_table = groupon_table.fillna(0)
    no_groupon_table = no_groupon_table.fillna(0)
    
    return (groupon_table,no_groupon_table,reward_table)   



get_year = lambda x:x.year
get_campaign = lambda x:0 if x=='None' else 1



conf = SparkConf()
conf.set("spark.app.name", "markov_decision_process")
conf.set("spark.master", "local[4]")


session = SparkSession.builder.config(conf=conf).getOrCreate()


schema = StructType()
schema.add('identity', StringType(), True)
schema.add('product_id', StringType(), True)
schema.add('underwrote_date', DateType(), True)
schema.add('start_date', DateType(), True)
schema.add('end_date', DateType(), True)
schema.add('offset', IntegerType(), True)
schema.add('premium', DoubleType(), True)
schema.add('discount', DoubleType(), True)
schema.add('groupon', StringType(), True)
schema.add('campaign', StringType(), True)

           
transactions = session.read.csv("transactions.csv",
                           schema=schema,
                           sep=',')

transactions = transactions.rdd.map(lambda x:(x[0],x[1],x[2],x[3],x[4],x[5],
                                              x[6],x[7],x[8],get_campaign(x[9]),
                                              get_year(x[3])))

schema.add('start_year',IntegerType())
transactions = session.createDataFrame(transactions,
                                       schema=schema)


'''
groupon_transactions = transactions.filter(transactions.groupon=='1')
groupon_discount = pd.DataFrame(groupon_transactions.rdd.map(lambda x:x[7]).collect())
print(groupon_discount.describe())

groupon_discount = groupon_discount[groupon_discount[0]<1000]
groupon_discount = groupon_discount[groupon_discount[0]>0]
plt.hist(groupon_discount,bins=5)


no_groupon_transactions = transactions.filter(transactions.groupon!='1')
no_groupon_discount = pd.DataFrame(no_groupon_transactions.rdd.map(lambda x:x[7]).collect())
print(no_groupon_discount.describe())

no_groupon_discount = no_groupon_discount[no_groupon_discount[0]<1000]
no_groupon_discount = no_groupon_discount[no_groupon_discount[0]>0]
plt.figure()
plt.hist(no_groupon_discount,bins=5)
'''



history_transactions = transactions.select('identity','start_year','premium')
history_transactions = history_transactions.filter(history_transactions.start_year!=2017)

history_transactions = history_transactions.select('identity','premium')
history_value = history_transactions.groupby('identity').agg({'premium':'sum'})
history_value = history_value.withColumnRenamed(
    'sum(premium)','history_value')


last_transactions = transactions.select('identity','start_year','end_date')
last_transactions = last_transactions.filter(last_transactions.start_year==2016)

last_transactions = last_transactions.select('identity','end_date')


last_transaction_count = last_transactions.groupby('identity').count()
last_transaction_count = last_transaction_count.filter('count<=2')
last_transaction_count = last_transaction_count.select('identity')
last_transactions = last_transactions.join(last_transaction_count,
                                           on='identity',
                                           how='inner')


last_transaction_date = last_transactions.groupby('identity').agg({'end_date':'max'})

last_transaction_date = last_transaction_date.withColumnRenamed(
    'max(end_date)','last_expiry_date')


current_transactions = transactions.select('identity',
                                           'start_year',
                                           'underwrote_date',
                                           'premium',
                                           'campaign',
                                           'groupon')

current_transactions = current_transactions.filter(current_transactions.start_year==2017)
current_transactions = current_transactions.select('identity',
                                                   'underwrote_date',
                                                   'premium',
                                                   'campaign',
                                                   'groupon')

current_transactions = current_transactions.groupby('identity')\
    .agg({'groupon':'sum',
          'campaign':'sum',
          'premium':'avg',
          'underwrote_date':'min'})


current_transactions = current_transactions\
    .withColumnRenamed('sum(groupon)','groupon')\
    .withColumnRenamed('sum(campaign)','campaign')\
    .withColumnRenamed('avg(premium)','premium')\
    .withColumnRenamed('min(underwrote_date)','response_date')

current_transactions = current_transactions.select('identity',
                                                   'groupon',
                                                   'campaign',
                                                   'premium',
                                                   'response_date')

customers = last_transaction_date.join(history_value,
                                       on='identity',
                                       how='left_outer')

customers = customers.join(current_transactions,
                           on='identity',
                           how='left_outer')

customers = customers.fillna(0)

customers = customers.filter(customers.history_value<=20000)


'''
customers.cache()
customers.collect()
mean_absolute_errors = []
for i in range(5):
    ###################################################### train model 
    
    (train_customers,test_customers) = customers.randomSplit([0.7,0.3])
    
    train_rewards = train_customers.rdd.map(lambda x:get_transition_rewards(x)).collect()
    
    train_rewards = pd.DataFrame(train_rewards,
                                 columns=['prior_state','action','premium'])
    
    train_rewards_summary = pd.pivot_table(train_rewards[['prior_state','action','premium']],
                                     index=['prior_state','action'],
                                     aggfunc=[np.mean,np.std,len,mean_deviation])
    
    train_rewards_summary = train_rewards_summary.fillna(0)
    train_rewards_summary = train_rewards_summary.reset_index(level=['prior_state',
                                                         'action'])
    train_rewards_summary.columns = ['state',
                                     'action',
                                     'premium_mean',
                                     'premium_std',
                                     'premium_count',
                                     'premium_mean_deviation']
    
           
    
    
    ###################################################### test model 
    
    test_rewards = test_customers.rdd.map(lambda x:get_transition_rewards(x)).collect()
    
    test_rewards = pd.DataFrame(test_rewards,
                                 columns=['prior_state','action','premium'])
    
    
    test_rewards_summary = pd.pivot_table(test_rewards[['prior_state','action','premium']],
                                     index=['prior_state','action'],
                                     aggfunc=[np.mean,np.std,len,mean_deviation])
    
    test_rewards_summary = test_rewards_summary.fillna(0)
    test_rewards_summary = test_rewards_summary.reset_index(level=['prior_state',
                                                         'action'])
    test_rewards_summary.columns = ['state',
                                     'action',
                                     'premium_mean',
                                     'premium_std',
                                     'premium_count',
                                     'premium_mean_deviation']
                            
    premium_mean = \
        train_rewards_summary[['state','action','premium_mean']].merge(
            test_rewards_summary[['state','action','premium_mean']],
            on=('state','action'),
            how='left')
        
    premium_mean = premium_mean.fillna(0)
    premium_mean['error'] =  \
        premium_mean.apply(lambda x:abs(x['premium_mean_x']-x['premium_mean_y']),
                           axis=1)  
                                  
    mean_absolute_error = premium_mean['error'].mean()
    mean_absolute_errors.append(mean_absolute_error)

avg_mean_absolute_error = np.mean(mean_absolute_errors)
'''



rewards = customers.rdd.map(lambda x:get_transition_rewards(x)).collect()

rewards = pd.DataFrame(rewards,
                       columns=['prior_state','action','premium'])

rewards_summary = pd.pivot_table(rewards[['prior_state','action','premium']],
                                 index=['prior_state','action'],
                                 aggfunc=[sum,np.mean,np.std,len,mean_deviation])

rewards_summary = rewards_summary.fillna(0)
rewards_summary = rewards_summary.reset_index(level=['prior_state',
                                                     'action'])
rewards_summary.columns = ['state',
                             'action',
                             'premium_sum',
                             'premium_mean',
                             'premium_std',
                             'premium_count',
                             'premium_mean_deviation']

rewards_summary.to_csv('rewards_summary_100_40_30_20.csv',index=False)                           


             
sequences = customers.rdd.flatMap(lambda x:create_transition_sequence(x))

states_transition = sequences.map(lambda x:((x[0],x[2]),
                                             (1 if x[1]=='groupon' else 0,
                                              x[3] if x[1]=='groupon' else 0,
                                              1 if x[1]!='groupon' else 0,
                                              x[3] if x[1]!='groupon' else 0)))

states_summary = states_transition.reduceByKey(lambda x,y:(x[0]+y[0],
                             x[1]+y[1],
                             x[2]+y[2],
                             x[3]+y[3])).collect()


transition_matrix = create_transition_matrix(states_summary)

groupon_df= transition_matrix[0].sort_index()
no_groupon_df = transition_matrix[1].sort_index()
reward_df = transition_matrix[2].sort_index()


transition = np.array(
     [np.asmatrix(groupon_df),
      np.asmatrix(no_groupon_df)])

reward = np.array(np.asmatrix(reward_df))


mdp = mdptoolbox.mdp.ValueIteration(transition, 
                                    reward, 
                                    discount=0.95)
mdp.setVerbose()
mdp.run()
value = mdp.V

result = pd.DataFrame(list(mdp.policy),
                      columns=['策略'],
                      index=reward_df.index)
 

result = result.replace(0,'无动作')
result = result.replace(1,'优惠券')
result.to_csv('result_100_40_30_20.csv')

result.index = ['历史保费'+i+'天内到期' for i in result.index]
result.to_csv('result_100_40_30_20_display.csv')
# 历史保费 天内到期







