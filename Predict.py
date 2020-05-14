import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

sns.set(style='darkgrid')
plt.rcParams['font.sans-serif']=['SimHei']     #中文显示问题
plt.rcParams['axes.unicode_minus']=False   #符号显示问题

#显示所有列
pd.set_option('display.max_columns', None)

path = os.getcwd()
data = pd.read_csv(path + '/data/LCIS_clean.csv',encoding='gbk')
# print(data.describe(),data.head())
print(data.info())

# 尝试进行特征选择并通过特征预测用户逾期风险

# col = data.columns
# print(col)
# 构造目标特征
# print(data['标当前状态'].unique())
data['是否逾期'] = data['标当前状态'].map({'已还清':'否' ,'逾期中':'是', '正常还款中':'否'})
# print(data.head(10))
# pdb.set_trace()
# breakpoint()
col = ['借款金额', '借款期限', '借款利率',  '初始评级', '借款类型',
       '是否首标', '年龄', '性别', '手机认证', '户口认证', '视频认证', '学历认证', '征信认证', '淘宝认证',
       '历史成功借款次数', '历史成功借款金额',  '历史正常还款期数', '历史逾期还款期数', '是否逾期']
df = data[col]

from sklearn.preprocessing import LabelEncoder
gle = LabelEncoder()
# print(bool(df['初始评级'].dtype == 'object'))
for c in col:
    if df[c].dtype == 'object':
        num  = gle.fit_transform(df[c])
        df[c] = num
    pass
# print(df.head())
# print(df[df['标当前逾期天数']>0].head(10))
# 计算相关性系数矩阵
# corr = df.corr(method='pearson')
# # 创建一个 Mask 来隐藏相关矩阵的上三角形
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# # 绘制图像
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corr, mask=mask, vmax=1, center=0, annot=True, fmt='.1f',
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.title('借款特征与是否逾期相关性')
# plt.show()

#尝试用k近邻进行预测
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

def pre(df):
    # df = df.iloc[:30000, :]
    # print(df)
    y = df['是否逾期']
    x = df.drop(['是否逾期'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    # 特征工程 （标准化）
    std = StandardScaler()
    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    return x_train, x_test, y_train, y_test

def knn(x_train, x_test, y_train, y_test):

    # 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=7)
    # 超参数验证:经验证，n_neighbors=7最佳
    # param={'n_neighbors':[4,5,6,7,8]}
    # knn=GridSearchCV(knn,param_grid=param,cv=2)
    # fit,predict,score
    knn.fit(x_train, y_train)
    train_socre = knn.score(x_train, y_train)
    test_socre = knn.score(x_test, y_test)
    # print(train_socre, test_socre)
    # print(knn.best_score_,knn.best_estimator_)
    return test_socre

def tree(x_train, x_test, y_train, y_test ):
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)
    test_sore = dec.score(x_test, y_test)
    return test_sore

def forest(x_train, x_test, y_train, y_test ):

    # rfc = RandomForestClassifier(random_state=2, n_estimators=30, min_samples_split=4, min_samples_leaf=2)
    # rfc_s = cross_val_score(rfc, x_train, y_train, cv=10)
    # clf = DecisionTreeClassifier()
    # clf_s = cross_val_score(clf, x_train, y_train, cv=10)
    # plt.plot(range(1, 11), rfc_s, label="RandomForest")
    # plt.plot(range(1, 11), clf_s, label="Decision Tree")
    # plt.title('决策树与随机森林准确率比较')
    # plt.legend()
    # plt.show()

    # kf = KFold(n_splits=3)
    # tree_param_grid = {'min_samples_split': list((2, 3, 4)),
    #                    'n_estimators': list((3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))}
    # grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=kf)  # (算法，调节参数（用字典形式），交叉验证次数)
    # grid.fit(x_train, y_train)  # 训练集
    # print(grid.cv_results_, grid.best_params_, grid.best_score_ ) # 得分，最优参数，最优得分

    rf = RandomForestClassifier(random_state=2, n_estimators=30, min_samples_split=4, min_samples_leaf=2)
    rf.fit(x_train, y_train)
    scores_test = rf.score(x_test,y_test)
    # print(scores_test)
    return scores_test

socre = []
x_train, x_test, y_train, y_test = pre(df.sample(3000))
socre.append(knn(x_train, x_test, y_train, y_test))
x_train, x_test, y_train, y_test = pre(df)
socre.append(tree(x_train, x_test, y_train, y_test))
socre.append(forest(x_train, x_test, y_train, y_test))
print(socre)
plt.figure(figsize=(16,9))
plt.bar(x = ['k近邻','决策树','随机森林'],height = socre,color=['orange', 'g','c'])
plt.ylabel('准确率',fontsize=14)
plt.ylim(0.9,1.0)
plt.title('三种分类预测算法准确率对比')
plt.show()
