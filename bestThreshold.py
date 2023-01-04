from ProfileClassFunctionsV2 import detect_anomaly




############### Centroids distances with PCA ###############
tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(0,21):
    res = detect_anomaly(i*0.1, 1, True)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------Threshold Stats Centroid Distances with PCA--------------------")
print("For threshold {} the maximum number of true positives is {}".format((tps.index(max(tps))+1)*0.1, max(tps)))
print("For threshold {}  the minimum number of false positives is {}".format((fps.index(min(fps))+1)*0.1, min(fps)))
print("For threshold {:.1f} the best accuracy is {}".format((acc.index(max(acc))+1)*0.1, max(acc)))
print("For threshold {} the best precision is {}".format((pre.index(max(pre))+1)*0.1, max(pre)))
print("For threshold {} the best recall is {}".format((rec.index(max(rec))+2)*0.1, max(rec)))
print("For threshold {} the best f1-score is {}".format((f1s.index(max(f1s))+2)*0.1, max(f1s)))




############### Centroids distances without PCA ###############
tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(1,21):
    res = detect_anomaly(i*0.1, 2, True)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------Threshold Stats Centroid Distances without PCA--------------------")
print("For threshold {} the maximum number of true positives is {}".format((tps.index(max(tps))+1)*0.1, max(tps)))
print("For threshold {}  the minimum number of false positives is {}".format((fps.index(min(fps))+1)*0.1, min(fps)))
print("For threshold {:.1f} the best accuracy is {}".format((acc.index(max(acc))+1)*0.1, max(acc)))
print("For threshold {} the best precision is {}".format((pre.index(max(pre))+1)*0.1, max(pre)))
print("For threshold {} the best recall is {}".format((rec.index(max(rec))+2)*0.1, max(rec)))
print("For threshold {} the best f1-score is {}".format((f1s.index(max(f1s))+2)*0.1, max(f1s)))







############### Multivariate with PCA ###############
tps = []    #True Positives
fps = []    #False Positives
acc = []    #Accuracy
pre = []    #Precision
rec = []    #Recall
f1s = []    #F1 Score


for i in range(1,21):
    res = detect_anomaly(i*0.1, 3, True)
    
    tps.append(res[0])
    fps.append(res[1])
    acc.append(res[2])
    pre.append(res[3])
    rec.append(res[4])
    f1s.append(res[5])

print("\n--------------------Threshold Stats Multivariate --------------------")
print("For threshold {:.1f} the maximum number of true positives is {}".format((tps.index(max(tps))+1)*0.1, max(tps)))
print("For threshold {}  the minimum number of false positives is {}".format((fps.index(min(fps))+1)*0.1, min(fps)))
print("For threshold {:.1f} the best accuracy is {}".format((acc.index(max(acc))+1)*0.1, max(acc)))
print("For threshold {} the best precision is {}".format((pre.index(max(pre))+1)*0.1, max(pre)))
print("For threshold {} the best recall is {}".format((rec.index(max(rec))+2)*0.1, max(rec)))
print("For threshold {} the best f1-score is {}".format((f1s.index(max(f1s))+2)*0.1, max(f1s)))