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
    cObs,cFea=oClass.shape
    colors=['b','g','r', 'y']
    for i in range(nObs):
        if i < cObs: 
            plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()

def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


def printStats(tp, tn, fp, fn):
    print("True Positives: {}, True Negatives: {}".format(tp,tn))
    print("False Positives: {}, False Negatives: {}".format(fp,fn))
    print("Accuracy: {}%".format(((tp+tn)/(tp+tn+fp+fn))*100))
    precision=(tp)/(tp+fp)
    print("Precision: {}%".format((precision)*100))
    recall=(tp)/(tp+fn)
    print("Recall: {}%".format((recall)*100))
    print("F1-Score: {}".format(((2*(recall*precision))/(recall+precision))))
    




########### Main Code #############
Classes={0:'Client',1:"Attacker"}
#plt.ion()
nfig=1


#Load data from text file
features_c1=np.loadtxt("Captures/client1_d1_obs_features.dat")
features_c2=np.loadtxt("Captures/client2_d1_obs_features.dat")
features_c3=np.loadtxt("Captures/client3_d1_obs_features.dat")
features_attacker=np.loadtxt("Captures/attacker_d1_obs_features.dat")

#Returning arrays with ones of the size of the features extracted
oClass_client=np.ones((len(features_c1)+len(features_c2),1))*0
oClass_attacker=np.ones((len(features_attacker),1))*1
oClass_client_test=np.ones((len(features_c3),1))*0

#Stack arrays of features and classes vertically
features=np.vstack((features_c1,features_c2,features_attacker))
oClass=np.vstack((oClass_client,oClass_attacker))
print('Train Stats Features Size:',features.shape)
print('Classes Size: ', oClass.shape)

#Plot features
#plt.figure(4)
#plotFeatures(features,oClass,2,3) #0,27

########### Silence Features #############

features_c1S=np.loadtxt("Captures/client1_d1_obs_Sfeatures.dat")
features_c2S=np.loadtxt("Captures/client2_d1_obs_Sfeatures.dat")
features_c3S=np.loadtxt("Captures/client3_d1_obs_Sfeatures.dat")
features_attackerS=np.loadtxt("Captures/attacker_d1_obs_Sfeatures.dat")

featuresS=np.vstack((features_c1S,features_c2S,features_attackerS))
oClass=np.vstack((oClass_client,oClass_attacker))
print('Train Silence Features Size:',featuresS.shape)
print('Classes Size: ', oClass.shape)

# plt.figure(5)
# plotFeatures(featuresS,oClass,0,2) #0,23

#############----Feature Training----#############
#:1
#1 Build train features of normal behavior
#   trainClassClient -> good behavior only
trainFeaturesClient=np.vstack((features_c1,features_c2))
trainFeaturesClientS=np.vstack((features_c1S,features_c2S))
allTrainFeaturesClient=np.hstack((trainFeaturesClient,trainFeaturesClientS))
trainClassClient=oClass_client

#2 Build test features with attacker behavior
#   testClassAttacker -> bad behavior for anomaly detection
testFeaturesAttacker=features_attacker
testFeaturesAttackerS=features_attackerS
allTestFeaturesAttacker=np.hstack((testFeaturesAttacker,testFeaturesAttackerS))
testClassAttacker=oClass_attacker

#3 Build test features with a clients good behavior
#   testClassClient -> good behavior for anomaly detection
testFeaturesClient=features_c3
testFeaturesClientS=features_c3S
allTestFeaturesClient=np.hstack((testFeaturesClient,testFeaturesClientS))
testClassClient=oClass_client_test





#############----Feature Normalization----#############
from sklearn.preprocessing import MaxAbsScaler

#Without silences
#Normalize the train class
trainScaler = MaxAbsScaler().fit(trainFeaturesClient)
trainFeaturesN=trainScaler.transform(trainFeaturesClient)

##Normalize test classes with the train Scalers
AtestFeaturesNA=trainScaler.transform(testFeaturesAttacker)
AtestFeaturesNC=trainScaler.transform(testFeaturesClient)

# #With silences
# #Normalize the train class
# trainScaler = MaxAbsScaler().fit(allTrainFeaturesClient)
# trainFeaturesN=trainScaler.transform(allTrainFeaturesClient)

# ##Normalize test classes with the train Scalers
# AtestFeaturesNA=trainScaler.transform(allTestFeaturesAttacker)
# AtestFeaturesNC=trainScaler.transform(allTestFeaturesClient)


# print("Mean of TrainFeatures")
# print(np.mean(trainFeaturesN,axis=0))
# print("Standard deviation of TrainFeatures")
# print(np.std(trainFeaturesN,axis=0))

# print("Mean of TestFeatures")
# print(np.mean(AtestFeaturesNA,axis=0))
# print("Standard deviation of TestFeatures")
# print(np.std(AtestFeaturesNA,axis=0))





#############----Principal Components Analysis----#############
from sklearn.decomposition import PCA

#without silences - pca 11 --> max range 13
#with silences - pca 18 --> max range 18
pca = PCA(n_components=11, svd_solver='full')

#reduce train features data with PCA
trainPCA=pca.fit(trainFeaturesN)
trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

#Obtain test features from PCA training
AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)


# #Plot train PCA features
# print(trainFeaturesNPCA.shape,trainClassClient.shape)
# plt.figure(8)
# plotFeatures(trainFeaturesNPCA,trainClassClient,0,1)

# #Plot test PCA fratures
# print(AtestFeaturesNPCA.shape,testClassAttacker.shape)
# plt.figure(8)
# plotFeatures(AtestFeaturesNPCA,testClassAttacker,0,1)

# #Plot test PCA fratures
# print(CtestFeaturesNPCA.shape,testClassClient.shape)
# plt.figure(8)
# plotFeatures(CtestFeaturesNPCA,testClassClient,0,1)






#############----Anomaly Detection based on centroids distances----#############
from sklearn.preprocessing import MaxAbsScaler
#11
centroids={}
pClass=(trainClassClient==0).flatten() #Client Class = 0
centroids.update({0:np.mean(trainFeaturesN[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

AnomalyThreshold=1.2
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=AtestFeaturesNA.shape
for i in range(nObsTest):
    x=AtestFeaturesNA[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #True Positive
        tp += 1
    else:
        result="OK"
        #False Negative
        fn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*dists,result))

nObsTest,nFea=AtestFeaturesNC.shape
for i in range(nObsTest):
    x=AtestFeaturesNC[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #False Positive
        fp += 1
    else:
        result="OK"
        #True Negative
        tn += 1
    # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*dists,result))

printStats(tp, tn, fp, fn)






#############----Anomaly Detection based on centroids distances (PCA Features)----#############
from sklearn.preprocessing import MaxAbsScaler
#12
centroids={}
pClass=(trainClassClient==0).flatten() #Client Class = 0
centroids.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
#print('All Features Centroids:\n',centroids)

tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

AnomalyThreshold=1.2
print('\n-- Anomaly Detection based on Centroids Distances (PCA Features)--')
nObsTest,nFea=AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=AtestFeaturesNPCA[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #True Positive
        tp += 1
    else:
        result="OK"
        #False Negative
        fn += 1
    #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*dists,result))

nObsTest,nFea=CtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=CtestFeaturesNPCA[i]
    dists=[distance(x,centroids[0])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
        #False Positive
        fp += 1
    else:
        result="OK"
        #True Negative
        tn += 1
    #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*dists,result))

printStats(tp, tn, fp, fn)







#############----Anomaly Detection based Multivariate---#############
#13 ---- Este aqui é o que parece dar melhor
from scipy.stats import multivariate_normal
print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
means={}
pClass=(trainClassClient==0).flatten() #Client Class = 0
means.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
#print(means)

covs={}
pClass=(trainClassClient==0).flatten() #Client Class = 0
covs.update({0:np.cov(trainFeaturesNPCA[pClass,:],rowvar=0)})
#print(covs)


tp = 0 #True Positive
tn = 0 #True Negative
fp = 0 #False Positive
fn = 0 #False Negative

AnomalyThreshold=1.2
nObsTest,nFea=AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=AtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
        #True Positive
        tp += 1
    else:
        result="OK"
        #False Negative
        fn += 1
    
    # print('Obs: {:2} ({}): Probabilities: [{:.4e}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*probs,result))

nObsTest,nFea=CtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=CtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
        #False Positive
        fp += 1
    else:
        result="OK"
        #True Negative
        tn += 1
    
    # print('Obs: {:2} ({}): Probabilities: [{:.4e}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*probs,result))
#Print statistics   
printStats(tp, tn, fp, fn)







#############----Anomaly Detection based on One Class Support Vector Machines (PCA Features)---#############
#14
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesNPCA)  

tpL, tnL, fpL, fnL = 0, 0, 0, 0
tpRBF, tnRBF, fpRBF, fnRBF = 0, 0, 0, 0
tpP, tnP, fpP, fnP = 0, 0, 0, 0


L1=ocsvm.predict(AtestFeaturesNPCA)
L2=rbf_ocsvm.predict(AtestFeaturesNPCA)
L3=poly_ocsvm.predict(AtestFeaturesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=AtestFeaturesNPCA.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1


L1=ocsvm.predict(CtestFeaturesNPCA)
L2=rbf_ocsvm.predict(CtestFeaturesNPCA)
L3=poly_ocsvm.predict(CtestFeaturesNPCA)

nObsTest,nFea=CtestFeaturesNPCA.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1
print("\nKernel Linear Statistics")
printStats(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printStats(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
printStats(tpP, tnP, fpP, fnP)











#############----Anomaly Detection based on One Class Support Vector Machines ---#############
#15
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesN)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesN)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesN)  

tpL, tnL, fpL, fnL = 0, 0, 0, 0
tpRBF, tnRBF, fpRBF, fnRBF = 0, 0, 0, 0
tpP, tnP, fpP, fnP = 0, 0, 0, 0


L1=ocsvm.predict(AtestFeaturesNA)
L2=rbf_ocsvm.predict(AtestFeaturesNA)
L3=poly_ocsvm.predict(AtestFeaturesNA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=AtestFeaturesNA.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        tpL += 1
    else:
        fnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        tpRBF += 1
    else:
        fnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        tpP += 1
    else:
        fnP += 1


L1=ocsvm.predict(AtestFeaturesNC)
L2=rbf_ocsvm.predict(AtestFeaturesNC)
L3=poly_ocsvm.predict(AtestFeaturesNC)

nObsTest,nFea=AtestFeaturesNC.shape
for i in range(nObsTest):
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
    
    #Linear
    if AnomResults[L1[i]] == "Anomaly":
        fpL += 1
    else:
        tnL += 1
    #RBF
    if AnomResults[L2[i]] == "Anomaly":
        fpRBF += 1
    else:
        tnRBF += 1
    #Poly
    if AnomResults[L3[i]] == "Anomaly":
        fpP += 1
    else:
        tnP += 1
print("\nKernel Linear Statistics")
printStats(tpL, tnL, fpL, fnL)
print("\nKernel RBF Statistics")
printStats(tpRBF, tnRBF, fpRBF, fnRBF)
print("\nKernel Poly Statistics")
printStats(tpP, tnP, fpP, fnP)
