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
plt.figure(4)
plotFeatures(features,oClass,0,1)




#############----Feature Training----#############
#TrainFeatures with clients behavior only
percentage=0.5
pC1=int(len(features_c1)*percentage)
trainFeatures_c1=features_c1[:pC1,:]
pC2=int(len(features_c2)*percentage)
trainFeatures_c2=features_c2[:pC2,:]
pC3=int(len(features_c3)*percentage)
trainFeatures_c3=features_c3[:pC3,:]
pCAt=int(len(features_attacker)*percentage)
trainFeatures_at=features_attacker[:pCAt,:]

#Build features of normal behavior
trainFeatures=np.vstack((trainFeatures_c1,trainFeatures_c2,trainFeatures_c3))
o2trainClass=np.vstack((oClass_c1[:pC1],oClass_c2[:pC2],oClass_c3[:pC3]))
i2trainFeatures=trainFeatures

#Build features of normal and attacker behavior
trainFeatures=np.vstack((trainFeatures_c1,trainFeatures_c2,trainFeatures_c3, trainFeatures_at))
o3trainClass=np.vstack((oClass_c1[:pC1],oClass_c2[:pC2],oClass_c3[:pC3], oClass_attacker[:pCAt]))
i3trainFeatures=trainFeatures




#############----Feature Normalization----#############
from sklearn.preprocessing import MaxAbsScaler

#Fit and Transform to normalize data
i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

print("Mean of TrainFeatures")
print(np.mean(i2trainFeaturesN,axis=0))
print("Standard deviation of TrainFeatures")
print(np.std(i2trainFeaturesN,axis=0))


print("Mean of TrainFeatures")
print(np.mean(i3trainFeaturesN,axis=0))
print("Standard deviation of TrainFeatures")
print(np.std(i3trainFeaturesN,axis=0))





#############----Principal Components Analysis----#############
from sklearn.decomposition import PCA

pca2 = PCA(n_components=3, svd_solver='full')
#Add to the model training data
i2trainPCA=pca2.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

pca3 = PCA(n_components=4, svd_solver='full')
#Add to the model training data
i3trainPCA=pca3.fit(i3trainFeaturesN)
i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

#Plot PCA features
print(i2trainFeaturesNPCA.shape,o2trainClass.shape)
plt.figure(8)
plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)

#Plot PCA features
print(i3trainFeaturesNPCA.shape,o3trainClass.shape)
plt.figure(8)
plotFeatures(i3trainFeaturesNPCA,o3trainClass,0,1)





#############----Anomaly Detection based on centroids distances----#############
from sklearn.preprocessing import MaxAbsScaler
centroids={}
for c in range(3):  # Only the first three classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2trainFeaturesN[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

AnomalyThreshold=1.2
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=i3trainFeaturesN.shape
for i in range(nObsTest):
    x=i3trainFeaturesN[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3trainClass[i][0]],*dists,result))