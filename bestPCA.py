from ProfileClassFunctions import detect_anomaly


############### Centroids distances ###############
maxPCACentroids=16+1 #Resuls stabilize from 16 components

tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(2,maxPCACentroids):
    res = detect_anomaly(i, 1, False)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------PCA Stats Centroid Distances --------------------")
print("With {} components the maximum number of true positives is {}".format(tps.index(max(tps))+2, max(tps)))
print("With {} components the minimum number of false positives is {}".format(fps.index(min(fps))+2, min(fps)))
print("With {} components the best accuracy is {}".format(acc.index(max(acc))+2, max(acc)))
print("With {} components the best precision is {}".format(pre.index(max(pre))+2, max(pre)))
print("With {} components the best recall is {}".format(rec.index(max(rec))+2, max(rec)))
print("With {} components the best f1-score is {}".format(f1s.index(max(f1s))+2, max(f1s)))



############### Centroids distances w/Silence ###############
maxPCACentroids=16+1 #Resuls stabilize from 16 components

tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(2,maxPCACentroids):
    res = detect_anomaly(i, 1, True)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------PCA Stats Centroid Distances w/Silence--------------------")
print("With {} components the maximum number of true positives is {}".format(tps.index(max(tps))+2, max(tps)))
print("With {} components the minimum number of false positives is {}".format(fps.index(min(fps))+2, min(fps)))
print("With {} components the best accuracy is {}".format(acc.index(max(acc))+2, max(acc)))
print("With {} components the best precision is {}".format(pre.index(max(pre))+2, max(pre)))
print("With {} components the best recall is {}".format(rec.index(max(rec))+2, max(rec)))
print("With {} components the best f1-score is {}".format(f1s.index(max(f1s))+2, max(f1s)))











############### Multivariate ###############
maxPCAMultiVariate=12+1 #Does not work with 13 components

tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(2,maxPCAMultiVariate):
    res = detect_anomaly(i, 2, False)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------PCA Stats Multivariate --------------------")
print("With {} components the maximum number of true positives is {}".format(tps.index(max(tps))+2, max(tps)))
print("With {} components the minimum number of false positives is {}".format(fps.index(min(fps))+2, min(fps)))
print("With {} components the best accuracy is {}".format(acc.index(max(acc))+2, max(acc)))
print("With {} components the best precision is {}".format(pre.index(max(pre))+2, max(pre)))
print("With {} components the best recall is {}".format(rec.index(max(rec))+2, max(rec)))
print("With {} components the best f1-score is {}".format(f1s.index(max(f1s))+2, max(f1s)))



############### Multivariate w/Silence ###############
maxPCAMultiVariate=18+1 #Does not work with 19 components

tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(2,maxPCAMultiVariate):
    res = detect_anomaly(i, 2, True)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------PCA Stats Multivariate w/Silence--------------------")
print("With {} components the maximum number of true positives is {}".format(tps.index(max(tps))+2, max(tps)))
print("With {} components the minimum number of false positives is {}".format(fps.index(min(fps))+2, min(fps)))
print("With {} components the best accuracy is {}".format(acc.index(max(acc))+2, max(acc)))
print("With {} components the best precision is {}".format(pre.index(max(pre))+2, max(pre)))
print("With {} components the best recall is {}".format(rec.index(max(rec))+2, max(rec)))
print("With {} components the best f1-score is {}".format(f1s.index(max(f1s))+2, max(f1s)))
















############### One Class Support Vector Machines ###############
maxPCASVM=18+1 #Results stabilize for more than 18 components

tpsL, fpsL, accL, preL, recL, f1sL = [], [], [], [], [], []
tpsRBF, fpsRBF, accRBF, preRBF, recRBF, f1sRBF = [], [], [], [], [], []
tpsP, fpsP, accP, preP, recP, f1sP = [], [], [], [], [], []

for i in range(2,maxPCASVM):
    res = detect_anomaly(i, 3, False)

    #Linear kernel stuff
    tpsL.append(res[0][0])
    fpsL.append(res[0][1])
    accL.append(res[0][2])
    preL.append(res[0][3])
    recL.append(res[0][4])
    f1sL.append(res[0][5])
    #RBF kernel stuff
    tpsRBF.append(res[1][0])
    fpsRBF.append(res[1][1])
    accRBF.append(res[1][2])
    preRBF.append(res[1][3])
    recRBF.append(res[1][4])
    f1sRBF.append(res[1][5])
    #Poly kernel stuff
    tpsP.append(res[2][0])
    fpsP.append(res[2][1])
    accP.append(res[2][2])
    preP.append(res[2][3])
    recP.append(res[2][4])
    f1sP.append(res[2][5])

print("\n--------------------PCA Stats SVM Linear--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsL.index(max(tpsL))+2, max(tpsL)))
print("With {} components the minimum number of false positives is {}".format(fpsL.index(min(fpsL))+2, min(fpsL)))
print("With {} components the best accuracy is {}".format(accL.index(max(accL))+2, max(accL)))
print("With {} components the best precision is {}".format(preL.index(max(preL))+2, max(preL)))
print("With {} components the best recall is {}".format(recL.index(max(recL))+2, max(recL)))
print("With {} components the best f1-score is {}".format(f1sL.index(max(f1sL))+2, max(f1sL)))



print("\n--------------------PCA Stats SVM RBF--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsRBF.index(max(tpsRBF))+2, max(tpsRBF)))
print("With {} components the minimum number of false positives is {}".format(fpsRBF.index(min(fpsRBF))+2, min(fpsRBF)))
print("With {} components the best accuracy is {}".format(accRBF.index(max(accRBF))+2, max(accRBF)))
print("With {} components the best precision is {}".format(preRBF.index(max(preRBF))+2, max(preRBF)))
print("With {} components the best recall is {}".format(recRBF.index(max(recRBF))+2, max(recRBF)))
print("With {} components the best f1-score is {}".format(f1sRBF.index(max(f1sRBF))+2, max(f1sRBF)))



print("\n--------------------PCA Stats SVM Poly--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsP.index(max(tpsP))+2, max(tpsP)))
print("With {} components the minimum number of false positives is {}".format(fpsP.index(min(fpsP))+2, min(fpsP)))
print("With {} components the best accuracy is {}".format(accP.index(max(accP))+2, max(accP)))
print("With {} components the best precision is {}".format(preP.index(max(preP))+2, max(preP)))
print("With {} components the best recall is {}".format(recP.index(max(recP))+2, max(recP)))
print("With {} components the best f1-score is {}".format(f1sP.index(max(f1sP))+2, max(f1sP)))




############### One Class Support Vector Machines w/Silence###############
maxPCASVM=18+1 #Results stabilize for more than 18 components

tpsL, fpsL, accL, preL, recL, f1sL = [], [], [], [], [], []
tpsRBF, fpsRBF, accRBF, preRBF, recRBF, f1sRBF = [], [], [], [], [], []
tpsP, fpsP, accP, preP, recP, f1sP = [], [], [], [], [], []

for i in range(2,maxPCASVM):
    res = detect_anomaly(i, 3, True)

    #Linear kernel stuff
    tpsL.append(res[0][0])
    fpsL.append(res[0][1])
    accL.append(res[0][2])
    preL.append(res[0][3])
    recL.append(res[0][4])
    f1sL.append(res[0][5])
    #RBF kernel stuff
    tpsRBF.append(res[1][0])
    fpsRBF.append(res[1][1])
    accRBF.append(res[1][2])
    preRBF.append(res[1][3])
    recRBF.append(res[1][4])
    f1sRBF.append(res[1][5])
    #Poly kernel stuff
    tpsP.append(res[2][0])
    fpsP.append(res[2][1])
    accP.append(res[2][2])
    preP.append(res[2][3])
    recP.append(res[2][4])
    f1sP.append(res[2][5])

print("\n--------------------PCA Stats SVM Linear w/Silence--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsL.index(max(tpsL))+2, max(tpsL)))
print("With {} components the minimum number of false positives is {}".format(fpsL.index(min(fpsL))+2, min(fpsL)))
print("With {} components the best accuracy is {}".format(accL.index(max(accL))+2, max(accL)))
print("With {} components the best precision is {}".format(preL.index(max(preL))+2, max(preL)))
print("With {} components the best recall is {}".format(recL.index(max(recL))+2, max(recL)))
print("With {} components the best f1-score is {}".format(f1sL.index(max(f1sL))+2, max(f1sL)))



print("\n--------------------PCA Stats SVM RBF w/Silence--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsRBF.index(max(tpsRBF))+2, max(tpsRBF)))
print("With {} components the minimum number of false positives is {}".format(fpsRBF.index(min(fpsRBF))+2, min(fpsRBF)))
print("With {} components the best accuracy is {}".format(accRBF.index(max(accRBF))+2, max(accRBF)))
print("With {} components the best precision is {}".format(preRBF.index(max(preRBF))+2, max(preRBF)))
print("With {} components the best recall is {}".format(recRBF.index(max(recRBF))+2, max(recRBF)))
print("With {} components the best f1-score is {}".format(f1sRBF.index(max(f1sRBF))+2, max(f1sRBF)))



print("\n--------------------PCA Stats SVM Poly w/Silence--------------------")
print("With {} components the maximum number of true positives is {}".format(tpsP.index(max(tpsP))+2, max(tpsP)))
print("With {} components the minimum number of false positives is {}".format(fpsP.index(min(fpsP))+2, min(fpsP)))
print("With {} components the best accuracy is {}".format(accP.index(max(accP))+2, max(accP)))
print("With {} components the best precision is {}".format(preP.index(max(preP))+2, max(preP)))
print("With {} components the best recall is {}".format(recP.index(max(recP))+2, max(recP)))
print("With {} components the best f1-score is {}".format(f1sP.index(max(f1sP))+2, max(f1sP)))