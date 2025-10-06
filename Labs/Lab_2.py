from sklearn.metrics import confusion_matrix

true_label = [0,1,0,0,1,0,1,1,0]
predicted_labels1 =[0,1,0,0,1,0,1,1,0]

conf_matrix=confusion_matrix(true_label,predicted_labels1)

TN = conf_matrix[0,0]
FP = conf_matrix[0,1]
FN = conf_matrix[1,0]
TP = conf_matrix[1,1]

print(f'TN: {TN}')
print(f'FP: {FP}')
print(f'FN: {FN}')
print(f'TP: {TP}')




