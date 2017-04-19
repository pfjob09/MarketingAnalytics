#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:50:01 2017

@author: zhouyonglong
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import re


from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import unix_timestamp
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType,StructType,IntegerType,TimestampType
from pyspark.sql.functions import UserDefinedFunction




def create_path(element):
    path=[]
    path.append('start')
    
    visits = element[1]
    visits = sorted(visits,key=lambda x:x[3])
    
    channels = [x[1] for x in visits]
    for channel in channels:
        path.append(channel)
    
    conversion = sum([x[2] for x in visits])>0
    if conversion:
        path.append('conversion')
    else:
        path.append('null')
        
    return path



def create_path_with_value(element):
    path=[]
    path.append('start')
    
    visits = element[1]
    visits = sorted(visits,key=lambda x:x[3])
    
    channels = [x[1] for x in visits]
    for channel in channels:
        path.append(channel)
    
    conversion = sum([x[2] for x in visits])>0
    if conversion:
        path.append('conversion')
    else:
        path.append('null')
    
    value = sum([x[4] for x in visits])
    path.append(str(value))
    
        
    return path   
    
    
    
def create_last_click_stats_pair(element):
    visits = element[1]
    visits = sorted(visits,key=lambda x:x[3],reverse=True)
    
    conversion = sum(x[2] for x in visits)>0
    if conversion:
        result = []
        channel = visits[0][1]
        value = sum(x[4] for x in visits)
        result.append((channel,value))
        return result
    else:
        return []


def create_linear_click_stats_pair(element):
    visits = element[1]
    visits = sorted(visits,key=lambda x:x[3])
    
    conversion = sum(x[2] for x in visits)>0
    if conversion:
        result = []
        value = sum(x[4] for x in visits)
        for channel in [x[1] for x in visits]:
            result.append((channel,value/len(visits)))
        return result
    else:
        return []    
    


def create_first_order_states_pair(path):
    if len(path)<3:
        return []
    else:
        result = []
        
        for i in range(0,len(path)-1):
            from_state = path[i]
            to_state = path[i+1]
            result.append(((from_state,to_state),1))
        
        return result


def create_first_order_states_pair_with_value(path):
    if len(path)<3:
        return []
    else:
        result = []
        value = float(path[-1])
        for i in range(0,len(path)-2):
            from_state = path[i]
            to_state = path[i+1]
            result.append(((from_state,to_state),value))
        
        return result        
        
        
        
channels = ["banner","text","keyword","link","video","mobile",'unknown']

channels_dict = {'01':'banner',
                 '02':'text',
                 '06':'keyword',
                 '07':'link',
                 '08':'video',
                 '09':'mobile',
                 '10':'mobile',
                 '11':'mobile'}

columns = channels.copy()
columns.insert(0,"start")
columns.append("null")
columns.append("conversion")
      


def create_first_order_markov_model(states_pair):    
    states = defaultdict()
    states["start"]=0

    channel_index=0
    for channel in channels:
        channel_index=channel_index+1
        states[channel]=channel_index
        
    states["null"]=len(channels)+1
    states["conversion"]=len(channels)+2
    num_states = len(states)

    table = pd.DataFrame(np.zeros((num_states,num_states)),
                         columns=columns,
                         index=columns)

    for item in states_pair:
        from_state = item[0][0]
        to_state = item[0][1]
        value = item[1]

        row = states[from_state] 
        column = states[to_state] 
        table.iloc[row][column] = value

    # remove same-state transitions
    for r in range(0,num_states):
        prior_state = columns[r]
        for c in range(0,num_states):
            current_state = columns[c]
            if prior_state in channels and prior_state!=current_state and current_state!='start':
                table.iloc[r][c] = table.iloc[r][c]
            elif prior_state=='start' and current_state in channels:
                table.iloc[r][c] = table.iloc[r][c]
            else:
                table.iloc[r][c] = 0
    
    #normalize
    for r in range(0,num_states):
        prior_state = columns[r]
        if prior_state in channels or prior_state=='start':
            row_sum = sum([x for x in table.iloc[r]])
            for c in range(0,num_states):
                table.iloc[r][c] = table.iloc[r][c] / row_sum
    
    table = table.fillna(0)
    
    return table        
        
        
# calculate orders count
get_orders = UserDefinedFunction(lambda x:0 if x=='' else int(str.replace(x,',','')), 
                                 IntegerType())        

# calculate orders count
get_revenue = UserDefinedFunction(lambda x:0 if x=='' else int(str.replace(x,',','')), 
                                 IntegerType())


def get_channel(campaign):
    channel_type = campaign[-2:]
    if channel_type in channels_dict.keys():
        return channels_dict[channel_type]
    else:
        return 'unknown'
    
map_campaign = UserDefinedFunction(lambda x:get_channel(x),
                                   StringType())



conf = SparkConf()
conf.set("spark.app.name", "channel_attribution_spark")
#conf.set("spark.master", "spark://192.168.42.141:7077")
conf.set("spark.master", "local[4]")


session = SparkSession.builder.config(conf=conf).getOrCreate()

           
schema = StructType().add("TrackingID", StringType(), True)\
    .add("IdentityNumber", StringType(), True, None)\
    .add("VisitNumber", StringType(), True, None)\
    .add("HitTimestring", StringType(), True, None)\
    .add("Campaign", StringType(), True, None)\
    .add("Type", StringType(), True, None)\
    .add("PhoneNumber", StringType(), True, None)\
    .add("Email", StringType(), True, None)\
    .add("ReferrerDomain", StringType(), True, None)\
    .add("Orders", StringType(), True, None)\
    .add("Revenue", StringType(), True, None)

   
# clicks_test_users.txt
# raw_clicks.csv
raw_clicks = session.read.csv("./web/*.txt",
                              schema=schema,
                              sep='\t')        
           
raw_clicks = raw_clicks.filter(raw_clicks.TrackingID != 'None')

raw_clicks = raw_clicks.filter(raw_clicks.HitTimestring!='None')
raw_clicks = raw_clicks.filter(raw_clicks.HitTimestring!='0')
raw_clicks = raw_clicks.dropna(subset='HitTimestring')

raw_clicks = raw_clicks.filter(raw_clicks.ReferrerDomain != 'None')
raw_clicks = raw_clicks.filter(raw_clicks.ReferrerDomain != 'Internal Domain')


raw_clicks = raw_clicks.filter(raw_clicks.Revenue != 'None')
raw_clicks = raw_clicks.filter(raw_clicks.Campaign != 'None')

    
clicks = raw_clicks.select('TrackingID','VisitNumber','HitTimestring',
                         'Campaign','Orders','Revenue')


clicks = clicks.select(*[map_campaign(column).alias('Campaign')
    if column == 'Campaign' else column for column in clicks.columns])

clicks = clicks.select(*[unix_timestamp(column,'yyyy-MM-dd HH:mm:ss.SSS').alias('VisitDatetime')
    if column == 'HitTimestring' else column for column in clicks.columns])

clicks = clicks.select(*[get_orders(column).alias('Orders')
    if column == 'Orders' else column for column in clicks.columns])

clicks = clicks.select(*[get_revenue(column).alias('Revenue')
    if column == 'Revenue' else column for column in clicks.columns])


visits = clicks.groupBy('TrackingID','VisitNumber','Campaign')\
    .agg({'Orders':'max','VisitDatetime':'max','Revenue':'sum'})



visits = visits.rdd.map(lambda x:((x[0]),(x[1],x[2],x[3],x[4],x[5])))

visits = visits.groupByKey().mapValues(list)



last_click_states_pair = visits.flatMap(lambda x:create_last_click_stats_pair(x))\
    .reduceByKey(lambda x,y:x+y).collect()

total_conversion = sum([x[1] for x in last_click_states_pair])

last_click_conversion_rate = \
    [(x[0],x[1]/total_conversion) for x in last_click_states_pair]


last_click_channels = [x[0] for x in last_click_conversion_rate]
for channel in channels:
    if channel not in last_click_channels:
        last_click_conversion_rate.append((channel,0))

last_click_conversion_rate.sort(key=lambda x:x[0])
     

linear_states_pair = visits.flatMap(lambda x:create_linear_click_stats_pair(x))\
    .reduceByKey(lambda x,y:x+y).collect()

total_conversion = sum([x[1] for x in linear_states_pair])

linear_click_conversion_rate = \
    [(x[0],x[1]/total_conversion) for x in linear_states_pair]
     

linear_click_channels = [x[0] for x in linear_click_conversion_rate]
for channel in channels:
    if channel not in linear_click_channels:
        linear_click_conversion_rate.append((channel,0))

linear_click_conversion_rate.sort(key=lambda x:x[0])        
        
        
#path = visits.map(lambda x:create_path(x))

#first_order_states_pair = path.flatMap(lambda x:create_first_order_states_pair(x))\
#    .reduceByKey(lambda x,y:x+y)

path = visits.map(lambda x:create_path_with_value(x))

first_order_states_pair = path.flatMap(
    lambda x:create_first_order_states_pair_with_value(x))\
    .reduceByKey(lambda x,y:x+y)
    
first_order_markov_model = \
    create_first_order_markov_model(first_order_states_pair.collect())



DG=nx.DiGraph()

for prior in columns:
    for current in columns:
        weight = first_order_markov_model[current][prior]
        if weight > 0:
            DG.add_edge(prior,current,weight=weight)

            
pos = nx.spring_layout(DG)

nx.draw_networkx_edges(DG,pos,edgelist=DG.edges(),width=1,arrows=True)
nx.draw_networkx_labels(DG,pos,font_size=20,font_family='sans-serif')

edge_labels=dict([((u,v,),round(d['weight'],5)) for u,v,d in DG.edges(data=True)])
nx.draw_networkx_edge_labels(DG,pos,label_pos=0.3,edge_labels=edge_labels)



total_conversion = 0
all_simple_paths = list(nx.all_simple_paths(DG, source='start', 
                                            target='conversion'))
for path in all_simple_paths:
    
    conversion = 1
    for i in range(0,len(path)-1):
        antecesor = path[i]
        successor = path[i+1]
        weight = DG.edge[antecesor][successor]['weight']
        conversion = conversion * weight
    
    total_conversion = total_conversion + conversion

    
nodes = DG.nodes()
removal_effects = defaultdict()


for node in nodes:
    if node in channels:

        removal_effects[node] = 0
        graph = DG.copy()
        graph.remove_node(node)
        
        conversion_after_removal = 0
        for path in nx.all_simple_paths(graph, source='start', 
                                        target='conversion'):
        
            conversion = 1
            for i in range(0,len(path)-1):
                antecesor = path[i]
                successor = path[i+1]
                weight = DG.edge[antecesor][successor]['weight']
                conversion = conversion * weight
            
            conversion_after_removal = conversion_after_removal + conversion
               
        removal_effect = 1 - conversion_after_removal / total_conversion
        removal_effects[node] = removal_effect
    

total_removal_effects = sum([item[1] for item in removal_effects.items()]) 


first_order_markov_model_conversion_rate = \
    [(item[0],item[1]/total_removal_effects)
        for item in removal_effects.items()]


first_markov_channels = [x[0] for x in first_order_markov_model_conversion_rate]
for channel in channels:
    if channel not in first_markov_channels:
        first_order_markov_model_conversion_rate.append((channel,0))

first_order_markov_model_conversion_rate.sort(key=lambda x:x[0]) 




result = pd.DataFrame(last_click_conversion_rate,
                      columns=['channel','last_conversion_rate'])

result = result.merge(pd.DataFrame(linear_click_conversion_rate,
                                   columns=['channel','linear_conversion_rate']),
                      on='channel')

result = result.merge(pd.DataFrame(first_order_markov_model_conversion_rate,
                                   columns=['channel','markov_conversion_rate']),
                      on='channel')



fig = plt.figure()
ax = fig.add_subplot(111)
index = np.arange(len(channels))
width = 0.3

rects1 = ax.bar(index, result['last_conversion_rate'], width,
                color='red',
                error_kw=dict(elinewidth=2,ecolor='red'))

rects2 = ax.bar(index+width, result['linear_conversion_rate'], width,
                color='black',
                error_kw=dict(elinewidth=2,ecolor='black'))

rects3 = ax.bar(index+2*width, result['markov_conversion_rate'], width,
                color='blue',
                error_kw=dict(elinewidth=2,ecolor='black'))

# axes and labels
ax.set_xlim(-width,len(index)+width)
ax.set_ylim(0,0.7)
ax.set_ylabel('转化率')
ax.set_title('不同渠道来源转化率')

ax.set_xticks(index+1.5*width)
xtickNames = ax.set_xticklabels(result['channel'])
plt.setp(xtickNames, rotation=0, fontsize=10)

ax.legend((rects1[0],rects2[0],rects3[0]), 
          (u'最后一次点击模型',u'线性归因模型',u'数据驱动模型'))





