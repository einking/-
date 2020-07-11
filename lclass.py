import numpy as np 
# from sklearn.neighbors import NearestNeighbors
# from collections import Counter
# from collections import deque
from scipy.stats import norm 
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from annoy import AnnoyIndex
import pandas as pd
from matplotlib import pyplot
import sys
import math
import pandas as pd

# return_distance = True

def dataProcess(dataCircle):
    model = AnnoyIndex(dataCircle.shape[1], "euclidean")
    for i in range(dataCircle.shape[0]):
        model.add_item(i, dataCircle[i])
    model.build(10)

    # print("print nn in ",sys._getframe().f_lineno,dataCircle)
    dataAfterProcess = []
    asnn=[]
    for i in range(dataCircle.shape[0]):
        asnn.append(model.get_nns_by_item(i, 10, search_k=-1, include_distances=True))
        dataAfterProcess = np.asarray(asnn)
    return dataAfterProcess

def neighborsWithoutDistance(nn):
    indptr = np.asarray(nn[:, 0,:][:,1:],dtype=np.int64)
    return indptr

def neighborsWithDistance(nn):
    indptr = np.asarray(nn[:, 0,:][:,1:],dtype=np.int64)
    # print('indptr in ',sys._getframe().f_lineno,indptr)
    distance = nn[:,1,:][:,1:]
    return (indptr,distance)

def neighborsByRadius(radius,KneighborsWithDistance,):
    # KneighborsWithDistance = np.asarray(KneighborsWithDistance)
    # print('print KneighborsWithDistance in ', sys._getframe().f_lineno,radius.shape[0])
    neighborsbyRadius = [np.extract(KneighborsWithDistance[1][i]<=radius[i],KneighborsWithDistance[0][i]) for i in range(radius.shape[0])]
    
    # print('print neighbors in ', sys._getframe().f_lineno,neighbors)
    return neighborsbyRadius

def densityGet(KneighborsDistance,radiusTrue=False):
    # print("print neighborsWithDistance in ",sys._getframe().f_lineno,KneighborsWithDistance[1][:,0:])
    # np.log:  以e为低
    if radiusTrue:
        density = np.mean(KneighborsDistance[:,0],axis=1)
        density = np.log(density)
    else : 
        # 此处噪声与k近邻的距离大，因此计算出来的密度也大   
        density = np.mean(KneighborsDistance[1][:,0:],axis=1)
        densityMean = np.mean(density)
        density[np.where(density==0)] = densityMean
        density = np.log(density)
        # print(density)
    return density

def noiseFind(density,noisePercent,noisePoint,dataCircle):
    densityCopy = density.copy()
    print(densityCopy)
    # noise = np.asarray(dataCircle.shape[0], dtype = np.int64)
    noiseCount = dataCircle.shape[0] * noisePercent
    minDensity = np.min(densityCopy, 0)
    while noiseCount >= 0:
        # 此处噪声与k近邻的距离大，因此计算出来的密度也大 
        # 密度最大的即被认为是噪声
        maxIndex = np.argmax(densityCopy, 0)
        print(np.argmax(densityCopy, 0))
        print(maxIndex)
        noisePoint[maxIndex] = 1
        densityCopy[maxIndex] = minDensity-1
        noiseCount = noiseCount-1
    return noisePoint

def noiseDelete(noisePoint,dataCircle):
    noiseIndex = np.where(noisePoint == 1)
    # 在原数据集上按照噪声所在的行删除噪声样本
    dataWithoutNoise = np.delete(dataCircle,noiseIndex,axis=0)
    labelWithoutNoise = np.delete(labelCircle,noiseIndex)
    return dataWithoutNoise,labelWithoutNoise
    # print()

def radiusGet(KneighborsNoNoiseWithDistance,dataCircle):
    radius = np.zeros(dataCircle.shape[0])
    # print(radius.shape[0])
    # for i in range(dataWithoutNoise.shape[0]):
        # radius[i] = (1+1/(1+np.exp(chain_len.max()-chain_len[core[i]]))) * np.mean(KneighborsNoNoiseWithDistance[1][:,0:],axis=1)
    # 1+1/(1+np.exp(1)) *
    radius =  np.mean(KneighborsNoNoiseWithDistance[1][:,0:],axis=1)
    # print("print dataCircle.shape[0] in ",sys._getframe().f_lineno,KneighborsNoNoiseWithDistance.shape[0])
    return radius
    # print()

def borderDgreeGet(densityNoNoise,KneighborsNoNoiseWithDistance,KneighborsByRadius):
    densityWithoutNoise = densityNoNoise.copy()
    KneighborsNoNoiseWithDistance = np.asarray(KneighborsNoNoiseWithDistance)
    # print(KneighborsNoNoiseWithDistance.shape[1])
    '''
    for i in range(densityNoNoise.shape[0]):
        for j in range(0,KneighborsByRadius[i].shape[0]):
            print("print KneighborsByRadius in ",sys._getframe().f_lineno,KneighborsNoNoiseWithDistance[1][i][j])
        # print("print KneighborsByRadius in ",sys._getframe().f_lineno,KneighborsByRadius[i])
    '''
    # print("print KneighborsNoNoiseWithDistance in ",sys._getframe().f_lineno,KneighborsNoNoiseWithDistance)
    for i in range(densityWithoutNoise.shape[0]):
        for j in range(0,KneighborsByRadius[i].shape[0]):
            # print("print KneighborsByRadius[i][j] in ",sys._getframe().f_lineno,KneighborsByRadius[i][j])
            # print("print KneighborsNoNoiseWithDistance[1][i][j] in ",sys._getframe().f_lineno,KneighborsNoNoiseWithDistance[1][i][j])
            densityWithoutNoise[i] += densityWithoutNoise[KneighborsByRadius[i][j]] * KneighborsNoNoiseWithDistance[1][i][j]
    return densityWithoutNoise

def borderThresholdGet(borderDgree,borderPoint):
    # x = np.linspace(-math.pi/2,math.pi/2,borderDgree.shape[0])
    # bDgree = np.sort(borderDgree)
    # slope,intercept = np.polyfit(x,bDgree,1)
    slope = np.mean(borderDgree)+norm.ppf(0.8)*np.std(borderDgree,ddof=1)
    for i in range(borderPoint.shape[0]):
        if borderDgree[i] > slope:
            # print(borderDgree[i])
            borderPoint[i] = 1
    print(slope)
    return borderPoint

def borderPointFind(density,borderPercent,borderPoint,dataCircle):
    densityCopy = density.copy()
    # noise = np.asarray(dataCircle.shape[0], dtype = np.int64)
    borderCount = dataCircle.shape[0] * borderPercent
    minDensity = np.min(densityCopy, 0)
    while borderCount != 0:
        # 此处噪声与k近邻的距离大，因此计算出来的密度也大 
        # 密度最大的即被认为是噪声
        maxIndex = np.argmax(densityCopy, 0)
        borderPoint[maxIndex] = 1
        densityCopy[maxIndex] = minDensity-1
        borderCount = borderCount-1
    return borderPoint


if __name__ == "__main__":
    noisePercent = 0.005
    from sklearn.decomposition import PCA

    dataDNS = pd.read_csv(r'.\learn-master\dns\iodine\duplicates.csv',sep = ',')
    X = dataDNS[['label','qdcount','ancount','arcount','nscount','qd_qname_len','qd_qname_shannon','qd_qtype','an_rrname_len','an_rrname_shannon','an_type','an_ttl','an_rdata_len','an_rdata_shannon','ar_rrname_len','ar_rrname_shanonn','ar_type','ar_rdata_len','ar_rdata_shannon']]
    pca = PCA(n_components=2)
    dataCircle = pca.fit_transform(X)
    labelCircle = np.ones(dataCircle.shape[0], dtype=np.int64)
    dataAfterProcess = dataProcess(dataCircle)
    dataCircle=np.asarray(dataCircle)
    # Quadrant = np.zeros((dataCircle.shape[0],5),dtype=np.int64)
    # noisePoint = labelCircle.copy()
    noisePoint = np.zeros(dataCircle.shape[0],dtype=np.int64)
    # noisePoint = np.zeros(dataCircle.shape[0],dtype=np.int64)
    # KneighborsWithoutDistance = neighborsWithoutDistance(dataAfterProcess)
    KneighborsWithDistance = neighborsWithDistance(dataAfterProcess)

    density = densityGet(KneighborsWithDistance,False)
    # densityMean = radiusGet(KneighborsWithDistance,noisePercent)
    noise = noiseFind(density,noisePercent,noisePoint,dataCircle)
    import pandas as pd
    csvGo = pd.DataFrame(noise)
    csvGo.to_csv('noise.csv',sep = ',',index=False)


    #### 新的开始
    dataWithoutNoise,labelWithoutNoise = noiseDelete(noise,dataCircle)

    fig=pyplot.figure(1)
    pyplot.subplot(131)
    pyplot.title('make circle')
    pyplot.scatter(dataCircle[:,0],dataCircle[:,1],marker='o',c=labelCircle)

    pyplot.subplot(132)
    pyplot.title('cluster circle')
    pyplot.scatter(dataCircle[:,0],dataCircle[:,1],marker='o',c=noisePoint)

    pyplot.subplot(133)
    pyplot.title('noise  circle')
    # pyplot.scatter(dataWithoutNoise[:,0],dataWithoutNoise[:,1],marker='o',c=labelWithoutNoise)
    pyplot.scatter(dataWithoutNoise[:,0],dataWithoutNoise[:,1],marker='o')

    pyplot.show()
    