import pandas as pd


'''
# 数据预处理
'''

'''
## 读取数据
dataDns = pd.read_csv(r'.\learn-master\dns\iodine\2018-03-19-19-06-24.csv',sep = ',')


##  缺失值处理：删除
# print(dataDns.isnull().any(axis=0))  # 打印缺失值所在属性
dataDnsNew = dataDns.drop(labels=dataDns.index[dataDns['qd_qname_shannon'].isnull()],axis=0)
# print(dataDnsNew.isnull().any(axis=0))
X = dataDnsNew[['label','qdcount','ancount','arcount','nscount','qd_qname_len','qd_qname_shannon','qd_qtype','an_rrname_len','an_rrname_shannon','an_type','an_ttl','an_rdata_len','an_rdata_shannon','ar_rrname_len','ar_rrname_shanonn','ar_type','ar_rdata_len','ar_rdata_shannon']]

## 数据去重
X.drop_duplicates(['qd_qname_shannon'],keep='first',inplace=True)
X.to_csv(r'.\learn-master\dns\iodine\duplicates.csv',sep = ',',index=False)
'''

'''
# 为节约时间，读取去重后的数据开始后面的步骤
dataDNS = pd.read_csv(r'.\learn-master\dns\iodine\duplicates.csv',sep = ',')
X = dataDNS[['label','qdcount','ancount','arcount','nscount','qd_qname_len','qd_qname_shannon','qd_qtype','an_rrname_len','an_rrname_shannon','an_type','an_ttl','an_rdata_len','an_rdata_shannon','ar_rrname_len','ar_rrname_shanonn','ar_type','ar_rdata_len','ar_rdata_shannon']]
# print(dataDns)
'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
colors = np.array(['red','green','black','blue','purple','pink','tan','brown','orange'])



'''
# 主成分分析,此时带有噪声点
'''


'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
import pandas as pd
csvGo = pd.DataFrame(X_pca)
csvGo.to_csv('dataWithNoise.csv',sep = ',',index=False)
plt.scatter(X_pca[:,0],X_pca[:,1],marker='o')
plt.show()
'''


# plt.scatter(dataDnsNew['qd_qname_len'],dataDnsNew['qd_qname_shannon'],c=colors[dataDnsNew['label']])
# plt.show()
# plt.scatter(dataDnsNew.qd_qname_len,dataDnsNew.arcount,marker = '+',c='black')


'''
# 噪声处理
# pd.cut()，按照数据值的大小（从最大值到最小值进行等距划分）进行划分。每个间隔段里的间隔区间都是相同的，统一区间。
# pd.qcut()，按照数据出现频率百分比划分，比如要把数据分为四份，则四段分别是数据的0-25%，25%-50%，50%-75%，75%-100%，每个间隔段里的元素个数都是相同的，统一权重。
# df_cut = dataDnsNew.copy()
df_cut['cut_group'] = pd.cut(df_cut['qd_qname_shannon'],4)
# print(df_cut)
'''

'''
# 读取噪声处理之后的数据
dataDNS = pd.read_csv(r'.\dataNoNoise.csv',sep = ',')
X = np.asarray(dataDNS)
'''


'''
# 聚类
'''

'''
## KMeans聚类
from sklearn.cluster import KMeans
kn3 = KMeans(n_clusters=3).fit(X)
kn5 = KMeans(n_clusters=5).fit(X)
dataDNS['cluster3'] = kn3.labels_
dataDNS['cluster5'] = kn5.labels_
# print(dataDnsNew.sort_values('cluster'))
plt.scatter(X[:,0],X[:,1],c=kn5.labels_)
plt.show()
'''
'''
# DBSCAN-聚类
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=8, min_samples=2).fit(X)
dataDNS['clusterdb'] = db.labels_
plt.scatter(X[:,0],X[:,1],c=colors[dataDNS['clusterdb']])
plt.show()
'''

'''
# 基于Meanshift-聚类的SMOTE的过采样处理
from sklearn.cluster import MeanShift
ms = MeanShift().fit(X)
dataDNS['clusterms'] = ms.labels_

from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=20)
X,Y = smo.fit_sample(X,ms.labels_)

csvGo = pd.DataFrame(X)
csvGo.to_csv('overSample.csv',sep = ',',index=False)

plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
'''


'''
# 打标签
'''

'''
# 读取SMOTE处理之后的数据
dataDNS = pd.read_csv(r'.\overSample.csv',sep = ',')
X = np.asarray(dataDNS)
# 基于Meanshift-聚类打标签
from sklearn.cluster import MeanShift
db = MeanShift().fit(X)
dataDNS['clusterms'] = db.labels_
print(db.labels_)
Y = db.labels_
Y[np.where(Y==0)] = 0
Y[np.where(Y==1)] = 1
Y[np.where(Y==2)] = 0
Y[np.where(Y==3)] = 1
Y[np.where(Y==4)] = 1
# Y[np.where(Y==5)] = 1
# Y = pd.DataFrame(Y)
# X = pd.DataFrame(X)
label = pd.DataFrame(X)
# label = label.drop(['0','1'])
# label['x'] = X[:,0]
# label['y'] = X[:,1]
label['class'] = Y
label.to_csv('labelNoNoise.csv',index=False)
plt.scatter(X[:,0],X[:,1],c=db.labels_)
plt.show()
'''


'''
# 基于Kmean 5 打标签,此时标签含有噪声样本
noise = pd.read_csv('noise.csv',sep = ',')
noise = np.asarray(noise)
# print(noise[:,0])
from sklearn.cluster import KMeans
kn5 = KMeans(n_clusters=5).fit(X)
dataDNS['cluster5'] = kn5.labels_
Y = kn5.labels_
Y = np.asarray(Y)
# print(np.where(Y==0))
# print(Y.shape)
Y[np.where(Y==0)] = 0
Y[np.where(Y==1)] = 1
Y[np.where(Y==2)] = 0
Y[np.where(Y==3)] = 0
Y[np.where(Y==4)] = 1
# print(np.where(Y==0))
Y[np.where(noise[:,0]==1)] = 1
# Y[np.where(Y==5)] = 1
Y = pd.DataFrame(Y)
Y.to_csv('labelWithNoise.csv',index=False)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
# print(dataDnsNew.sort_values('cluster'))
'''

'''
# ROC曲线绘制函数
'''

from sklearn.metrics import roc_curve, auc
def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


# 读取标签数据集
labelDNS = pd.read_csv(r'.\labelNoNoise.csv',sep = ',')
label = np.asarray(labelDNS)
Y = label[:,2]
dataDNS = pd.read_csv(r'.\overSample.csv',sep = ',')
X = np.asarray(dataDNS)
# 数据分成训练集和测试集
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)



'''
# 分类算法
'''

'''
# 使用带交叉验证的网格搜索寻找最优参数
# 决策树分类
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
# decisionTree = tree.DecisionTreeClassifier()
params = {'max_leaf_nodes':list(range(2,100)),'min_samples_split':[2,3,4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1,verbose=1,cv=3)
grid_search_cv.fit(X_train,Y_train)

# 模型评估
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
Y_pred = grid_search_cv.predict(X_test)
# print(Y_test)
score = accuracy_score(Y_test,Y_pred)
print('accuracy_score:',score)

f_score = f1_score(Y_test,Y_pred,average='micro')
print('f1_score: {0}'.format(f_score))
from sklearn.metrics import recall_score
r_score = recall_score(Y_test,Y_pred,average='micro')
print('recall_score: {0}'.format(r_score))
plot_roc(Y_test,Y_pred)
'''



'''
# 随机森林分类
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# params = {'n_estimators':range(10,71,10),'min_samples_leaf':list(range(10,60,10)),'min_samples_split':[2,3,4],'max_depth':range(3,14,2),'max_features':range(3,11,2)}
# grid_search_cv = GridSearchCV(estimator = RandomForestClassifier(random_state=10,criterion='gini'),param_grid = params,verbose=1,iid=False,cv=3)
# grid_search_cv.fit(X_train,Y_train)
# print(Y_train.shape)
model = RandomForestClassifier(oob_score=True, random_state=30).fit(X_train,Y_train.ravel())
# 模型评估
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
Y_pred = model.predict(X_test)
# print(Y_test)
score = accuracy_score(Y_test,Y_pred)
print('accuracy_score:',score)

f_score = f1_score(Y_test,Y_pred,average='micro')
print('f1_score: {0}'.format(f_score))
# randomForest = ensemble.RandomForestClassifier()
from sklearn.metrics import recall_score
r_score = recall_score(Y_test,Y_pred,average='micro')
print('recall_score: {0}'.format(r_score))
plot_roc(Y_test,Y_pred)
'''




# 朴素贝叶斯分类算法
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# print(Y_train)
model = GaussianNB().fit(X_train,Y_train.ravel())
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


Y_pred = model.predict(X_test)
# print(Y_test)
score = accuracy_score(Y_test,Y_pred)
print('accuracy_score:',score)
f_score = f1_score(Y_test,Y_pred,average='micro')
print('f1_score: {0}'.format(f_score))

from sklearn.metrics import recall_score
r_score = recall_score(Y_test,Y_pred,average='micro')
print('recall_score: {0}'.format(r_score))
plot_roc(Y_test,Y_pred)













