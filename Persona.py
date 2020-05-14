import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
plt.rcParams['font.sans-serif']=['SimHei']     #中文显示问题
plt.rcParams['axes.unicode_minus']=False   #符号显示问题

#显示所有列
pd.set_option('display.max_columns', None)

path = os.getcwd()
data = pd.read_csv(path + '/data/LCIS.csv')
# print(data.describe(),data.head())
print(data.info())

# 进行数据预处理及用户画像分析
columns = {'ListingId':'列表序号','recorddate':'记录日期'}
data.rename(columns = columns,inplace=True)
# print(data.info())
# 缺失率分析
miss_rate = pd.DataFrame(data.apply( lambda x : sum(x.isnull())/len(x),axis=0),columns = ['缺失率'])
# print(miss_rate)
miss_columns = miss_rate[miss_rate['缺失率']>0]['缺失率'].apply(lambda x: format(x,'.3%'))
# print(miss_columns)
# print(data[data['下次计划还款日期'].isnull()].head())
# print(data[data['上次还款日期'].isnull()].head())
# print(data[data['历史成功借款金额'].isnull()].head())
# print(data[data['记录日期'].isnull()].head())
data.dropna(subset = ['记录日期'], how='any' ,inplace=True)
# miss_rate = pd.DataFrame(data.apply( lambda x : sum(x.isnull())/len(x),axis=0),columns = ['缺失率'])
# print(miss_rate)
# 重复值删除
# print(data[data.duplicated()])
data.drop_duplicates(inplace=True)
# print(data[data.duplicated()])
# 异常值
# sns.countplot(data['手机认证'])
data = data[(data['手机认证'] =='成功认证')|(data['手机认证'] == '未成功认证')]
# sns.countplot(data['是否首标'])
# plt.show()

data.to_csv(path + '/data/LCIS_clean.csv',encoding='gbk')
# 用户画像

# 年龄与放贷金额
df_age = data.groupby(['年龄'])['借款金额'].sum()
df_age = pd.DataFrame(df_age)
df_age['借款金额累计'] = df_age['借款金额'].cumsum()
df_age['借款金额累计占比'] = df_age['借款金额累计']/df_age['借款金额'].sum()
# print(df_age)
index_num = df_age[df_age['借款金额累计占比'] > 0.8].index[0]
cum_percent = df_age.loc[index_num,'借款金额累计占比']
# print(cum_percent)
plt.figure(figsize=(16,9))
plt.bar(x = df_age.index,height=df_age['借款金额'],color = 'steelblue',alpha = 0.5,linewidth=3)
plt.xlabel( xlabel='年龄',fontsize=10)
plt.axvline(x = index_num,color = 'orange',linestyle = '--',alpha = 0.8)
df_age['借款金额累计占比'].plot(style = '--ob',secondary_y = True)
plt.text(index_num+1,cum_percent,'累计占比达到：%.3f%%'%(cum_percent*100),color = 'black')
plt.title('年龄借款金额')
plt.show()
data['年龄分段'] = pd.cut(data['年龄'],[17,24,30,36,42,48,54,65],right=True)
df_age = pd.pivot_table(data=data,index= '年龄分段',values='列表序号',columns = '标当前状态',aggfunc=np.size)
# print(df_age)
df_age['借款笔数'] = df_age.sum(axis = 1)
df_age['借款笔数分布'] = df_age['借款笔数']/df_age['借款笔数'].sum()
df_age['逾期占比'] = df_age['逾期中']/df_age['借款笔数']
# print(df_age)
df_age['借款笔数分布%'] = df_age['借款笔数分布'].apply(lambda x:format(x,'.3%'))
df_age['逾期占比%'] = df_age['逾期占比'].apply(lambda x:format(x,'.3%'))
print(df_age)
plt.figure(figsize=(16,9))
df_age['借款笔数分布'].plot(kind = 'bar',color = 'steelblue',alpha = 0.5)
plt.ylabel('借款笔数分布')
df_age['逾期占比'].plot(color = 'steelblue',alpha = 0.5,secondary_y = True)
plt.ylabel('逾期占比')
plt.grid(True)
plt.title('年龄与逾期情况')
plt.show()

# 其他特征与逾期情况关系
def trans(data,col,ind):
    df = pd.pivot_table(data=data, index=ind, values='列表序号', columns=col, aggfunc=np.size)
    df['借款笔数'] = df.apply(np.sum,axis = 1)
    df['借款笔数分布'] = df['借款笔数'] / df['借款笔数'].sum()
    df['逾期占比'] = df['逾期中'] / df['借款笔数']
    print(df)

    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.pie(x=df['借款笔数分布'], labels=df.index, colors=['red', 'yellow'],
            autopct='%.1f%%', pctdistance=0.5, labeldistance=1.1)
    plt.title('%s比例'% ind)
    plt.subplot(122)
    plt.bar(x=df.index, height=df['逾期占比'], color=['orange', 'g'])
    plt.title('不同%s的人逾期情况'% ind)
    plt.suptitle('%s用户画像'% ind)
    plt.show()
    return df

col = ['初始评级', '借款类型', '是否首标', '性别', '手机认证', '户口认证',
       '视频认证', '学历认证', '征信认证', '淘宝认证']
for c in col:
    print(c)
    trans(data, '标当前状态', c)
