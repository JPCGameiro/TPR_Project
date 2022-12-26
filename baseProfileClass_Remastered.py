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
            
## -- 4 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
# ## -- 11 -- ##
# def distance(c,p):
#     return(np.sqrt(np.sum(np.square(p-c))))

########### Main Code #############
Classes={0:'Client1',1:'Client2',2:'Client3',3:"Attacker"}
plt.ion()
nfig=1

## -- 3 -- ##
#Load data from text file
features_c1=np.loadtxt("client1_d1_obs_features.dat")
features_c2=np.loadtxt("client2_d1_obs_features.dat")
features_c3=np.loadtxt("client3_d1_obs_features.dat")
features_attacker=np.loadtxt("goodAndBad_d1_obs_features.dat")      #temporario so pa testar

#Returning arrays with ones of the size of the features extracted

oClass_c1=np.ones((len(features_c1),1))*0
oClass_c2=np.ones((len(features_c2),1))*1
oClass_c3=np.ones((len(features_c3),1))*2
oClass_attacker=np.ones((len(features_attacker),1))*3

#Stack arrays of features and classes vertically
features=np.vstack((features_c1,features_c2,features_c3,features_attacker))
oClass=np.vstack((oClass_c1,oClass_c2,oClass_c3,oClass_attacker))

#NAO CONSEGUI DESENHAR O PLOT
# print('Train Stats Features Size:',features.shape)

# # ## -- 4 -- ##
# plt.figure(4)
# plotFeatures(features,oClass,0,1)#0,8

#ISTO AQUI É A PARTE DOS SILENCIOS
# ## -- 5 -- ##
# features_browsingS=np.loadtxt("Browsing_obs_sil_features.dat")
# features_ytS=np.loadtxt("YouTube_obs_sil_features.dat")
# features_miningS=np.loadtxt("Mining_obs_sil_features.dat")

# featuresS=np.vstack((features_ytS,features_browsingS,features_miningS))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Silence Features Size:',featuresS.shape)
# plt.figure(5)
# plotFeatures(featuresS,oClass,0,2)


#ISTO AQUI É A PARTE DO WAVELET
# ## -- 7 -- ##
# features_browsingW=np.loadtxt("Browsing_obs_per_features.dat")
# features_ytW=np.loadtxt("YouTube_obs_per_features.dat")
# features_miningW=np.loadtxt("Mining_obs_per_features.dat")

# featuresW=np.vstack((features_ytW,features_browsingW,features_miningW))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Wavelet Features Size:',featuresW.shape)
# plt.figure(7)
# plotFeatures(featuresW,oClass,3,6)


#ESTE AQUI ESTA +-
# ## -- 8 -- ##
# #:1

percentage=0.5
pC1=int(len(features_c1)*percentage)
trainFeatures_c1=features_c1[:pC1,:]
pC2=int(len(features_c2)*percentage)
trainFeatures_c2=features_c2[:pC2,:]
pC3=int(len(features_c3)*percentage)
trainFeatures_c3=features_c3[:pC3,:]

trainFeatures=np.vstack((trainFeatures_c1,trainFeatures_c2,trainFeatures_c3))

# trainFeatures_c1S=features_c1S[:pC1,:]
# trainFeatures_c2S=features_c2S[:pC2,:]
# trainFeatures_c3S=features_c3S[:pC3,:]

# trainFeaturesS=np.vstack((trainFeatures_c1S,trainFeatures_c2S,trainFeatures_c3S))

# trainFeatures_c1W=features_c1W[:pC1,:]
# trainFeatures_c2W=features_c2W[:pC2,:]
# trainFeatures_c3W=features_c3W[:pC3,:]

# trainFeaturesW=np.vstack((trainFeatures_c1W,trainFeatures_c2W,trainFeatures_c3W))

o2trainClass=np.vstack((oClass_c1[:pC1],oClass_c2[:pC2],oClass_c3[:pC3]))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))
#i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS))
i2trainFeatures=trainFeatures


# ## -- 9 -- ##
from sklearn.preprocessing import MaxAbsScaler

i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

# i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
# i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

# i3AtestFeaturesN=i2trainScaler.transform(i3testFeatures)          #sera q temos de fzr este? pq anomalias
# i3CtestFeaturesN=i3trainScaler.transform(i3testFeatures)

print(np.mean(i2trainFeaturesN,axis=0))
print(np.std(i2trainFeaturesN,axis=0))

# ## -- 10 -- ##
from sklearn.decomposition import PCA

pca = PCA(n_components=3, svd_solver='full')

i2trainPCA=pca.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

# i3trainPCA=pca.fit(i3trainFeaturesN)
# i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

# i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)      #sera q temos de fzr este? pq anomalias
# i3CtestFeaturesNPCA = i3trainPCA.transform(i3CtestFeaturesN)

print(i2trainFeaturesNPCA.shape,o2trainClass.shape)
# plt.figure(8)
# plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)

#ACHO QUE TEMOS DE USAR A PARTIR DAQUI ATE AO 15 e o 21 do guiao
# ## -- 11 -- ##
# from sklearn.preprocessing import MaxAbsScaler
# centroids={}
# for c in range(3):  # Only the first three classes
#     pClass=(o2trainClass==c).flatten()
#     centroids.update({c:np.mean(i2trainFeaturesN[pClass,:],axis=0)})
# print('All Features Centroids:\n',centroids)

# AnomalyThreshold=1.2
# print('\n-- Anomaly Detection based on Centroids Distances --')
# nObsTest,nFea=i3AtestFeaturesN.shape                                    #falta nos esta variavel
# for i in range(nObsTest):
#     x=i3AtestFeaturesN[i]
#     dists=[distance(x,centroids[0]),distance(x,centroids[1])]
#     if min(dists)>AnomalyThreshold:
#         result="Anomaly"
#     else:
#         result="OK"
       
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))

# ## -- 12 -- ##
# centroids={}
# for c in range(2):  # Only the first two classes
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


# ## -- 13 -- ##
# from scipy.stats import multivariate_normal
# print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
# means={}
# for c in range(2):
#     pClass=(o2trainClass==c).flatten()
#     means.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
# #print(means)

# covs={}
# for c in range(2):
#     pClass=(o2trainClass==c).flatten()
#     covs.update({c:np.cov(i2trainFeaturesNPCA[pClass,:],rowvar=0)})
# #print(covs)

# AnomalyThreshold=0.05
# nObsTest,nFea=i3AtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x=i3AtestFeaturesNPCA[i,:]
#     probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
#     if max(probs)<AnomalyThreshold:
#         result="Anomaly"
#     else:
#         result="OK"
    
#     print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*probs,result))


# ## -- 14 -- ##
# from sklearn import svm

# print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
# ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesNPCA)  
# rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesNPCA)  
# poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesNPCA)  

# L1=ocsvm.predict(i3AtestFeaturesNPCA)
# L2=rbf_ocsvm.predict(i3AtestFeaturesNPCA)
# L3=poly_ocsvm.predict(i3AtestFeaturesNPCA)

# AnomResults={-1:"Anomaly",1:"OK"}

# nObsTest,nFea=i3AtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

# ## -- 15 -- ##
# from sklearn import svm

# print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
# ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesN)  
# rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesN)  
# poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesN)  

# L1=ocsvm.predict(i3AtestFeaturesN)
# L2=rbf_ocsvm.predict(i3AtestFeaturesN)
# L3=poly_ocsvm.predict(i3AtestFeaturesN)

# AnomResults={-1:"Anomaly",1:"OK"}

# nObsTest,nFea=i3AtestFeaturesN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


# ## -- 16 -- ##
# centroids={}
# for c in range(3):  # All 3 classes
#     pClass=(o3trainClass==c).flatten()
#     centroids.update({c:np.mean(i3trainFeaturesNPCA[pClass,:],axis=0)})
# print('PCA Features Centroids:\n',centroids)

# print('\n-- Classification based on Centroids Distances (PCA Features) --')
# nObsTest,nFea=i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x=i3CtestFeaturesNPCA[i]
#     dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
#     ndists=dists/np.sum(dists)
#     testClass=np.argsort(dists)[0]
    
#     print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i,Classes[o3testClass[i][0]],*ndists,testClass,Classes[testClass]))


# ## -- 17-- # 
# from scipy.stats import multivariate_normal
# print('\n-- Classification based on Multivariate PDF (PCA Features) --')
# means={}
# for c in range(3):
#     pClass=(o3trainClass==c).flatten()
#     means.update({c:np.mean(i3trainFeaturesNPCA[pClass,:],axis=0)})
# #print(means)

# covs={}
# for c in range(3):
#     pClass=(o3trainClass==c).flatten()
#     covs.update({c:np.cov(i3trainFeaturesNPCA[pClass,:],rowvar=0)})
# #print(covs)

# nObsTest,nFea=i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     x=i3CtestFeaturesNPCA[i,:]
#     probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1]),multivariate_normal.pdf(x,means[2],covs[2])])
#     testClass=np.argsort(probs)[-1]
    
#     print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e},{:.4e}] -> Classification: {} -> {}'.format(i,Classes[o3testClass[i][0]],*probs,testClass,Classes[testClass]))


# ## -- 18 -- #
# print('\n-- Classification based on Support Vector Machines --')
# svc = svm.SVC(kernel='linear').fit(i3trainFeaturesN, o3trainClass)  
# rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesN, o3trainClass)  
# poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3trainFeaturesN, o3trainClass)  

# L1=svc.predict(i3CtestFeaturesN)
# L2=rbf_svc.predict(i3CtestFeaturesN)
# L3=poly_svc.predict(i3CtestFeaturesN)
# print('\n')

# nObsTest,nFea=i3CtestFeaturesN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))


# ## -- 19 -- #
# print('\n-- Classification based on Support Vector Machines  (PCA Features) --')
# svc = svm.SVC(kernel='linear').fit(i3trainFeaturesNPCA, o3trainClass)  
# rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesNPCA, o3trainClass)  
# poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3trainFeaturesNPCA, o3trainClass)  

# L1=svc.predict(i3CtestFeaturesNPCA)
# L2=rbf_svc.predict(i3CtestFeaturesNPCA)
# L3=poly_svc.predict(i3CtestFeaturesNPCA)
# print('\n')

# nObsTest,nFea=i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
    

# ## -- 20a -- ##
# from sklearn.neural_network import MLPClassifier
# print('\n-- Classification based on Neural Networks --')

# alpha=1
# max_iter=100000
# clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
# clf.fit(i3trainFeaturesN, o3trainClass) 
# LT=clf.predict(i3CtestFeaturesN) 

# nObsTest,nFea=i3CtestFeaturesN.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))

# ## -- 20b -- ##
# from sklearn.neural_network import MLPClassifier
# print('\n-- Classification based on Neural Networks (PCA Features) --')

# alpha=1
# max_iter=100000
# clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
# clf.fit(i3trainFeaturesNPCA, o3trainClass) 
# LT=clf.predict(i3CtestFeaturesNPCA) 

# nObsTest,nFea=i3CtestFeaturesNPCA.shape
# for i in range(nObsTest):
#     print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))
