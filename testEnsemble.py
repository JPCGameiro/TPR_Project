from Ensemble import detect_anomaly

pcaComponents = [2, 12, 9, 2, 7]

pcaComponentsSilence = [2, 18, 2, 4, 5]


res = detect_anomaly(pcaComponents, 1, False)


print("\n--------------------Ensemble Stats--------------------")
print("True positives: {}".format(res[0]))
print("False positives: {}".format(res[1]))
print("Accuracy: {}".format(res[2]))
print("Precision: {}".format(res[3]))
print("Recall: {}".format(res[4]))
print("F1-score: {}".format(res[5]))


resSilence = detect_anomaly(pcaComponentsSilence, 1, True)


print("\n--------------------Ensemble Stats w/Silence--------------------")
print("True positives: {}".format(resSilence[0]))
print("False positives: {}".format(resSilence[1]))
print("Accuracy: {}".format(resSilence[2]))
print("Precision: {}".format(resSilence[3]))
print("Recall: {}".format(resSilence[4]))
print("F1-score: {}".format(resSilence[5]))