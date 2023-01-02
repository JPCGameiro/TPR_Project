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



def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))



def printStats(tp, tn, fp, fn):
    #print("True Positives: {}, True Negatives: {}".format(tp,tn))
    #print("False Positives: {}, False Negatives: {}".format(fp,fn))
    accuracy=((tp+tn)/(tp+tn+fp+fn))*100
    #print("Accuracy: {}%".format(accuracy))
    precision=((tp)/(tp+fp))
    #print("Precision: {}%".format(precision*100))
    recall=((tp)/(tp+fn))
    #print("Recall: {}%".format(recall*100))
    if recall ==0 and precision == 0:
        f1score = 0
    else:
        f1score=((2*(recall*precision))/(recall+precision))
    #print("F1-Score: {}".format(f1score))
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
    from sklearn.preprocessing import MaxAbsScaler

    #Normalize the train class
    trainScaler = MaxAbsScaler().fit(allTrainFeaturesClient)
    trainFeaturesN=trainScaler.transform(allTrainFeaturesClient)

    ##Normalize test classes with the train Scalers
    AtestFeaturesNA=trainScaler.transform(allTestFeaturesAttacker)
    AtestFeaturesNC=trainScaler.transform(allTestFeaturesClient)






    #############----Principal Components Analysis----#############
    from sklearn.decomposition import PCA

    #without silences - pca 11 --> max range 13
    #with silences - pca 18 --> max range 18
    pca = PCA(n_components=nComponents, svd_solver='full')

    #reduce train features data with PCA
    trainPCA=pca.fit(trainFeaturesN)
    trainFeaturesNPCA = trainPCA.transform(trainFeaturesN)

    #Obtain test features from PCA training
    AtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNA)
    CtestFeaturesNPCA = trainPCA.transform(AtestFeaturesNC)



    if choice==1:
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
        #print('\n-- Anomaly Detection based on Centroids Distances (PCA Features)--')
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

        stats=printStats(tp, tn, fp, fn)
        return stats
    elif choice==2:
        #############----Anomaly Detection based Multivariate (PCA Features)---#############
        #13 ---- Este aqui é o que parece dar melhor
        from scipy.stats import multivariate_normal
        #print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
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
        stats=printStats(tp, tn, fp, fn)
        return stats
    elif choice==3:
        #############----Anomaly Detection based on One Class Support Vector Machines (PCA Features)---#############
        #14
        from sklearn import svm

        #print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
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
        #print("\nKernel Linear Statistics")
        statsL = printStats(tpL, tnL, fpL, fnL)
        #print("\nKernel RBF Statistics")
        statsRBF = printStats(tpRBF, tnRBF, fpRBF, fnRBF)
        #print("\nKernel Poly Statistics")
        statsP = printStats(tpP, tnP, fpP, fnP)
        return (statsL, statsRBF, statsP)
