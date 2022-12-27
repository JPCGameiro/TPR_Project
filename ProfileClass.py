import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
warnings.filterwarnings('ignore')


def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r', 'y']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()

def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


########### Main Code #############
Classes={0:'Client1',1:'Client2',2:'Client3',3:"Attacker"}
#plt.ion()
nfig=1


#Load data from text file
features_c1=np.loadtxt("Captures/client1_d1_obs_features.dat")
features_c2=np.loadtxt("Captures/client2_d1_obs_features.dat")
features_c3=np.loadtxt("Captures/client3_d1_obs_features.dat")
features_attacker=np.loadtxt("Captures/attacker_d1_obs_features.dat")
#Returning arrays with ones of the size of the features extracted
oClass_c1=np.ones((len(features_c1),1))*0
oClass_c2=np.ones((len(features_c2),1))*1
oClass_c3=np.ones((len(features_c3),1))*2
oClass_attacker=np.ones((len(features_attacker),1))*3

#Stack arrays of features and classes vertically
features=np.vstack((features_c1,features_c2,features_c3,features_attacker))
oClass=np.vstack((oClass_c1,oClass_c2,oClass_c3,oClass_attacker))
print('Train Stats Features Size:',features.shape)
print('Classes Size: ', oClass.shape)

#Plot features
# plt.figure(4)
# plotFeatures(features,oClass,2,3) #0,27




#############----Feature Training----#############
#:1
#TrainFeatures with first 50% of clients behavior only
percentage=0.5
pC1=int(len(features_c1)*percentage)
trainFeatures_c1=features_c1[:pC1,:]
pC2=int(len(features_c2)*percentage)
trainFeatures_c2=features_c2[:pC2,:]
pC3=int(len(features_c3)*percentage)
trainFeatures_c3=features_c3[:pC3,:]


#1 Build train features of normal behavior
#   o2trainClass -> first 50% of good behavior only
trainFeatures=np.vstack((trainFeatures_c1,trainFeatures_c2,trainFeatures_c3))
o2trainClass=np.vstack((oClass_c1[:pC1],oClass_c2[:pC2],oClass_c3[:pC3]))
i2trainFeatures=trainFeatures


#2 Build train features of normal + bad behavior
#   o3trainClass -> first 50% of good and bad behavior
pAt=int(len(features_attacker)*percentage)
trainFeatures_at=features_attacker[:pAt,:]
trainFeatures=np.vstack((trainFeatures_c1,trainFeatures_c2,trainFeatures_c3,trainFeatures_at))
o3trainClass=np.vstack((oClass_c1[:pC1],oClass_c2[:pC2],oClass_c3[:pC3],oClass_attacker[:pAt]))
i3trainFeatures=trainFeatures


#:3 Build test features of normal + bad behavior
#   o3testClass -> second 50% of good and bad behavior
testFeatures_c1=features_c1[pC1:,:]
testFeatures_c2=features_c1[pC1:,:]
testFeatures_c3=features_c1[pC1:,:]
testFeatures_at=features_attacker[pAt:,:]
testFeatures=np.vstack((testFeatures_c1, testFeatures_c2, testFeatures_c3, testFeatures_at))

o3testClass=np.vstack((oClass_c1[pC1:],oClass_c2[pC2:],oClass_c3[pC3:],oClass_attacker[pAt:]))
#i3testFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))
#i3testFeatures=np.hstack((testFeatures,testFeaturesS))
i3testFeatures=testFeatures






#############----Feature Normalization----#############
from sklearn.preprocessing import MaxAbsScaler

#Normalize i2 train class
i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

#Normalize i3 train class
i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

##Normalize i3 test class with i2 and i3 train Scalers
i3AtestFeaturesN=i2trainScaler.transform(i3testFeatures)
i3CtestFeaturesN=i3trainScaler.transform(i3testFeatures)


'''
print("Mean of i2 TrainFeatures")
print(np.mean(i2trainFeaturesN,axis=0))
print("Standard deviation of i2 TrainFeatures")
print(np.std(i2trainFeaturesN,axis=0))

print("Mean of i3 TrainFeatures")
print(np.mean(i3trainFeaturesN,axis=0))
print("Standard deviation of i3 TrainFeatures")
print(np.std(i3trainFeaturesN,axis=0))

print("Mean of i2 scaled TestFeatures")
print(np.mean(i3AtestFeaturesN,axis=0))
print("Standard deviation of i2 scaled  TestFeatures")
print(np.std(i3AtestFeaturesN,axis=0))

print("Mean of i3 scaled TestFeatures")
print(np.mean(i3CtestFeaturesN,axis=0))
print("Standard deviation of i3 scaled  TestFeatures")
print(np.std(i3CtestFeaturesN,axis=0))
'''



#############----Principal Components Analysis----#############
from sklearn.decomposition import PCA

pca2 = PCA(n_components=3, svd_solver='full')
#Add to the PCA train model the i2 training data
i2trainPCA=pca2.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

# pca3 = PCA(n_components=4, svd_solver='full')
#Add to the PCA train model the i3 training data
i3trainPCA=pca2.fit(i3trainFeaturesN)
i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

#Obtain test features from i2 PCA training
i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)


#Plot train PCA features
print(i2trainFeaturesNPCA.shape,o2trainClass.shape)
plt.figure(8)
plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)

#Plot train PCA features
print(i3trainFeaturesNPCA.shape,o3trainClass.shape)
plt.figure(8)
plotFeatures(i3trainFeaturesNPCA,o3trainClass,0,1)

#Plot test PCA fratures
print(i3AtestFeaturesNPCA.shape,o3testClass.shape)
plt.figure(8)
plotFeatures(i3AtestFeaturesNPCA,o3testClass,0,1)


exit(0)


#############----Anomaly Detection based on centroids distances----#############
from sklearn.preprocessing import MaxAbsScaler
#11
# centroids={}
# for c in range(3):  # Only the first three classes
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2trainFeaturesN[pClass,:],axis=0)})
# print('All Features Centroids:\n',centroids)

# AnomalyThreshold=1.2
# print('\n-- Anomaly Detection based on Centroids Distances --')
# nObsTest,nFea=i3trainFeaturesN.shape
# for i in range(nObsTest):
#     x=i3trainFeaturesN[i]
#     dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
#     if min(dists)>AnomalyThreshold:
#         result="Anomaly"
#     else:
#         result="OK"
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3trainClass[i][0]],*dists,result))

#12
# centroids={}
# for c in range(3):  # Only the first two classes
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
# print('All Features Centroids:\n',centroids)

# AnomalyThreshold=1.2
# print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
# nObsTest,nFea=i3AtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x=i3AtestFeaturesNPCA[i]
#     dists=[distance(x,centroids[0]),distance(x,centroids[1])]
#     if min(dists)>AnomalyThreshold:
#         result="Anomaly"
#     else:
#         result="OK"
       
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))

#13 ---- Este aqui é o que parece dar melhor
from scipy.stats import multivariate_normal
print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
means={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    means.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
#print(means)

covs={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    covs.update({c:np.cov(i2trainFeaturesNPCA[pClass,:],rowvar=0)})
#print(covs)

AnomalyThreshold=0.05
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3AtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
    
    print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*probs,result))