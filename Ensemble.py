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
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import multivariate_normal
from sklearn import svm


warnings.filterwarnings('ignore')



def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))



def printStats(tp, tn, fp, fn):
    # print("True Positives: {}, True Negatives: {}".format(tp,tn))
    # print("False Positives: {}, False Negatives: {}".format(fp,fn))
    accuracy=((tp+tn)/(tp+tn+fp+fn))*100
    # print("Accuracy: {}%".format(accuracy))
    if tp+fp!= 0:
        precision=((tp)/(tp+fp))
    else:
        precision=0
    # print("Precision: {}%".format(precision*100))
    recall=((tp)/(tp+fn))
    # print("Recall: {}%".format(recall*100))
    if recall ==0 and precision == 0:
        f1score = 0
    else:
        f1score=((2*(recall*precision))/(recall+precision))
    # print("F1-Score: {}".format(f1score))
    return [tp, fp, accuracy, precision*100, recall*100, f1score]
    




def detect_anomaly(nComponents, choice, silence):
    ########### Main Code #############
    Classes={0:'Client',1:"Attacker"}

    #Load data from text file
    features_c1=np.loadtxt("Captures/client1_d1_obs_features.dat")
    features_c2=np.loadtxt("Captures/client2_d1_obs_features.dat")
    features_c3=np.loadtxt("Captures/client3_d1_obs_features.dat")
    features_attacker=np.loadtxt("Captures/attacker_d1_obs_features.dat")

    #Returning arrays with ones of the size of the features extracted
    oClass_client1=np.ones((len(features_c1),1))*0
    oClass_client2=np.ones((len(features_c2),1))*0
    oClass_client3=np.ones((len(features_c3),1))*0
    oClass_attacker=np.ones((len(features_attacker),1))*1

    #Stack arrays of features and classes vertically
    features=np.vstack((features_c1,features_c2,features_c3))
    oClass=np.vstack((oClass_client1, oClass_client2, oClass_client3))


    ########### Silence Features #############

    features_c1S=np.loadtxt("Captures/client1_d1_obs_Sfeatures.dat")
    features_c2S=np.loadtxt("Captures/client2_d1_obs_Sfeatures.dat")
    features_c3S=np.loadtxt("Captures/client3_d1_obs_Sfeatures.dat")
    features_attackerS=np.loadtxt("Captures/attacker_d1_obs_Sfeatures.dat")

    featuresS=np.vstack((features_c1S,features_c2S,features_attackerS))
    oClass=np.vstack((oClass_client1, oClass_client2, oClass_client3))




    #############----Feature Training----#############
    percentage=0.5
    pC1=int(len(features_c1)*percentage)
    trainFeaturesC1=features_c1[:pC1,:]
    pC2=int(len(features_c2)*percentage)
    trainFeaturesC2=features_c2[:pC2,:]
    pC3=int(len(features_c3)*percentage)
    trainFeaturesC3=features_c3[:pC3,:]

    pC1S=int(len(features_c1S)*percentage)
    trainFeaturesC1S=features_c1S[:pC1S,:]
    pC2S=int(len(features_c2S)*percentage)
    trainFeaturesC2S=features_c2S[:pC2S,:]
    pC3S=int(len(features_c3S)*percentage)
    trainFeaturesC3S=features_c3S[:pC3S,:]


    #:1
    #1 Build train features of normal behavior
    #   trainClassClient -> good behavior only
    trainFeaturesClient=np.vstack((trainFeaturesC1, trainFeaturesC2, trainFeaturesC3))
    if silence:
        trainFeaturesClientS=np.vstack((trainFeaturesC1S, trainFeaturesC2S, trainFeaturesC3S))
        allTrainFeaturesClient=np.hstack((trainFeaturesClient,trainFeaturesClientS))
    else:
        allTrainFeaturesClient=trainFeaturesClient
    #trainClassClient=oClass_client
    trainClassClient=np.vstack((oClass_client1[:pC1],oClass_client2[:pC2],oClass_client3[:pC3]))



    #2 Build test features with attacker behavior
    #   testClassAttacker -> bad behavior for anomaly detection
    testFeaturesAttacker=features_attacker
    if silence:
        testFeaturesAttackerS=features_attackerS
        allTestFeaturesAttacker=np.hstack((testFeaturesAttacker,testFeaturesAttackerS))
    else:
        allTestFeaturesAttacker=testFeaturesAttacker
    testClassAttacker=oClass_attacker



    pC1=int(len(features_c1)*percentage)
    trainFeaturesC1=features_c1[pC1:,:]
    pC2=int(len(features_c2)*percentage)
    trainFeaturesC2=features_c2[pC2:,:]
    pC3=int(len(features_c3)*percentage)
    trainFeaturesC3=features_c3[pC3:,:]

    pC1S=int(len(features_c1S)*percentage)
    trainFeaturesC1S=features_c1S[pC1S:,:]
    pC2S=int(len(features_c2S)*percentage)
    trainFeaturesC2S=features_c2S[pC2S:,:]
    pC3S=int(len(features_c3S)*percentage)
    trainFeaturesC3S=features_c3S[pC3S:,:]

    #3 Build test features with a clients good behavior
    #   testClassClient -> good behavior for anomaly detection
    testFeaturesClient=np.vstack((trainFeaturesC1, trainFeaturesC2, trainFeaturesC3))
    if silence:
        testFeaturesClientS=np.vstack((trainFeaturesC1S, trainFeaturesC2S, trainFeaturesC3S))
        allTestFeaturesClient=np.hstack((testFeaturesClient,testFeaturesClientS))
    else:
        allTestFeaturesClient=testFeaturesClient
    #testClassClient=oClass_client_test
    testClassClient=np.vstack((oClass_client1[pC1:],oClass_client2[pC2:],oClass_client3[pC3:]))






    #############----Feature Normalization----#############

    #Normalize the train class
    trainScaler = MaxAbsScaler().fit(allTrainFeaturesClient)
    trainFeaturesN=trainScaler.transform(allTrainFeaturesClient)

    ##Normalize test classes with the train Scalers
    AtestFeaturesNA=trainScaler.transform(allTestFeaturesAttacker)
    AtestFeaturesNC=trainScaler.transform(allTestFeaturesClient)


    #############----Anomaly Detection based on centroids distances----#############
    #11
    centroids={}
    pClass=(trainClassClient==0).flatten() #Client Class = 0
    centroids.update({0:np.mean(trainFeaturesN[pClass,:],axis=0)})
    # print('All Features Centroids:\n',centroids)


    # tp = 0 #True Positive
    # tn = 0 #True Negative
    # fp = 0 #False Positive
    # fn = 0 #False Negative

    result1_A = []
    result1_C = []

    AnomalyThreshold=1.2
    # print('\n-- Anomaly Detection based on Centroids Distances --')
    nObsTest,nFea=AtestFeaturesNA.shape
    for i in range(nObsTest):
        x=AtestFeaturesNA[i]
        dists=[distance(x,centroids[0])]
        if min(dists)>AnomalyThreshold:
            result1_A.append("Anomaly")
            #True Positive
            # tp += 1
        else:
            result1_A.append("OK")
            #False Negative
            # fn += 1
        # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*dists,result))

    nObsTest,nFea=AtestFeaturesNC.shape
    for i in range(nObsTest):
        x=AtestFeaturesNC[i]
        dists=[distance(x,centroids[0])]
        if min(dists)>AnomalyThreshold:
            result1_C.append("Anomaly")
            #False Positive
            # fp += 1
        else:
            result1_C.append("OK")
            #True Negative
            # tn += 1
        # print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*dists,result))

    # printStats(tp, tn, fp, fn)



    #############----Anomaly Detection based on centroids distances (PCA Features)----#############

    #############----Principal Components Analysis----#############

    pca = PCA(n_components=nComponents[0], svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)


    #12
    centroids={}
    pClass=(trainClassClient==0).flatten() #Client Class = 0
    centroids.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
    #print('All Features Centroids:\n',centroids)

    # tp = 0 #True Positive
    # tn = 0 #True Negative
    # fp = 0 #False Positive
    # fn = 0 #False Negative

    result2_A = []
    result2_C = []

    AnomalyThreshold=1.2
    #print('\n-- Anomaly Detection based on Centroids Distances (PCA Features)--')
    nObsTest,nFea=AtestFeaturesNPCA.shape
    for i in range(nObsTest):
        x=AtestFeaturesNPCA[i]
        dists=[distance(x,centroids[0])]
        if min(dists)>AnomalyThreshold:
            result2_A.append("Anomaly")
            #True Positive
            # tp += 1
        else:
            result2_A.append("OK")
            #False Negative
            # fn += 1
        #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*dists,result))

    nObsTest,nFea=CtestFeaturesNPCA.shape
    for i in range(nObsTest):
        x=CtestFeaturesNPCA[i]
        dists=[distance(x,centroids[0])]
        if min(dists)>AnomalyThreshold:
            result2_C.append("Anomaly")
            #False Positive
            # fp += 1
        else:
            result2_C.append("OK")
            #True Negative
            # tn += 1
        #print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*dists,result))

    
    #############----Anomaly Detection based Multivariate (PCA Features)---#############
    #13
    pca = PCA(n_components=nComponents[1], svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)

    means={}
    pClass=(trainClassClient==0).flatten() #Client Class = 0
    means.update({0:np.mean(trainFeaturesNPCA[pClass,:],axis=0)})
    #print(means)

    covs={}
    pClass=(trainClassClient==0).flatten() #Client Class = 0
    covs.update({0:np.cov(trainFeaturesNPCA[pClass,:],rowvar=0)})
    #print(covs)

    result3_A = []
    result3_C = []

    AnomalyThreshold=0.05
    nObsTest,nFea=AtestFeaturesNPCA.shape
    for i in range(nObsTest):
        x=AtestFeaturesNPCA[i,:]
        probs=np.array([multivariate_normal.pdf(x,means[0],covs[0])])
        if max(probs)<AnomalyThreshold:
            result3_A.append("Anomaly")
            #True Positive
            # tp += 1
        else:
            result3_A.append("OK")
            #False Negative
            # fn += 1
        
        # print('Obs: {:2} ({}): Probabilities: [{:.4e}] -> Result -> {}'.format(i,Classes[testClassAttacker[i][0]],*probs,result))

    nObsTest,nFea=CtestFeaturesNPCA.shape
    for i in range(nObsTest):
        x=CtestFeaturesNPCA[i,:]
        probs=np.array([multivariate_normal.pdf(x,means[0],covs[0])])
        if max(probs)<AnomalyThreshold:
            result3_C.append("Anomaly")
            #False Positive
            # fp += 1
        else:
            result3_C.append("OK")
            #True Negative
            # tn += 1
        
        # print('Obs: {:2} ({}): Probabilities: [{:.4e}] -> Result -> {}'.format(i,Classes[testClassClient[i][0]],*probs,result))
    #Print statistics   
    
    
    #############----Anomaly Detection based on One Class Support Vector Machines (PCA Features)---#############
    #14 -- linear

    pca = PCA(n_components=nComponents[2], svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)

    #print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
    ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesNPCA)   


    L1=ocsvm.predict(AtestFeaturesNPCA)

    AnomResults={-1:"Anomaly",1:"OK"}

    result4_A = []
    result4_C = []

    nObsTest,nFea=AtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #Linear
        if AnomResults[L1[i]] == "Anomaly":
            result4_A.append("Anomaly")
            # tpL += 1
        else:
            result4_A.append("OK")
            # fnL += 1


    L1=ocsvm.predict(CtestFeaturesNPCA)

    nObsTest,nFea=CtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #Linear
        if AnomResults[L1[i]] == "Anomaly":
            result4_C.append("Anomaly")
            # fpL += 1
        else:
            result4_C.append("OK")
            # tnL += 1

    #14 -- rbf

    pca = PCA(n_components=nComponents[3], svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)

    #print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
    rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesNPCA)  


    L2=rbf_ocsvm.predict(AtestFeaturesNPCA)

    AnomResults={-1:"Anomaly",1:"OK"}

    result5_A = []
    result5_C = []

    nObsTest,nFea=AtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #RBF
        if AnomResults[L2[i]] == "Anomaly":
            result5_A.append("Anomaly")
            # tpRBF += 1
        else:
            result5_A.append("OK")
            # fnRBF += 1

    L2=rbf_ocsvm.predict(CtestFeaturesNPCA)

    nObsTest,nFea=CtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #RBF
        if AnomResults[L2[i]] == "Anomaly":
            result5_C.append("Anomaly")
            # fpRBF += 1
        else:
            result5_C.append("OK")
            # tnRBF += 1

    #14 -- poly

    pca = PCA(n_components=nComponents[4], svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)

    #print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
    poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesNPCA)  

    L3=poly_ocsvm.predict(AtestFeaturesNPCA)

    AnomResults={-1:"Anomaly",1:"OK"}

    result6_A = []
    result6_C = []

    nObsTest,nFea=AtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassAttacker[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #Poly
        if AnomResults[L3[i]] == "Anomaly":
            result6_A.append("Anomaly")
            # tpP += 1
        else:
            result6_A.append("OK")
            # fnP += 1


    L3=poly_ocsvm.predict(CtestFeaturesNPCA)

    nObsTest,nFea=CtestFeaturesNPCA.shape
    for i in range(nObsTest):
        #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[testClassClient[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
        
        #Poly
        if AnomResults[L3[i]] == "Anomaly":
            result6_C.append("Anomaly")
            # fpP += 1
        else:
            result6_C.append("OK")
            # tnP += 1


    tp = 0 #True Positive
    tn = 0 #True Negative
    fp = 0 #False Positive
    fn = 0 #False Negative

    #All methods
    for i in range(len(result1_A)):

        results = []
        results.append(result1_A[i])
        results.append(result2_A[i])
        results.append(result3_A[i])
        results.append(result4_A[i])
        results.append(result5_A[i]) 
        results.append(result6_A[i])

        if results.count("Anomaly") >= int(len(results)/2):
            result = "Anomaly"
            tp += 1
        else:
            result = "OK"
            fn += 1

    for i in range(len(result1_C)):

        results = []
        results.append(result1_C[i])
        results.append(result2_C[i])
        results.append(result3_C[i])
        results.append(result4_C[i])
        results.append(result5_C[i]) 
        results.append(result6_C[i])

        # print("Resultados:")
        # print(results)

        if results.count("Anomaly") >= int(len(results)/2):
            result = "Anomaly"
            fp += 1
        else:
            result = "OK"
            tn += 1

    stats=printStats(tp, tn, fp, fn)
    return stats
    
