import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

df = pd.read_csv("D:\study\я_профессионал\курс\classification.csv")
score = pd.read_csv("D:\study\я_профессионал\курс\scores.csv")

TP, TN, FP, FN = 0,0,0,0
for index, row in df.iterrows():
    if row['true'] == row['pred']:
        if row['true'] == 1:
            TP += 1
        else:
            TN += 1
    else:
        if (row['true'] == 0) and (row['pred'] == 1):
            FP += 1
        elif (row['true'] == 1) and (row['pred'] == 0):
            FN += 1
print(TP, FP, FN, TN)

accuracy = accuracy_score(df['true'], df['pred'])
precision = precision_score(df['true'], df['pred'])
recall = recall_score(df['true'], df['pred'])
f1 = f1_score(df['true'], df['pred'])

print(round(accuracy,2), round(precision,2), round(recall,2), round(f1,2))

roc_auc_log = roc_auc_score(score['true'], score['score_logreg'])
roc_auc_scm = roc_auc_score(score['true'], score['score_svm'])
roc_auc_knn = roc_auc_score(score['true'], score['score_knn'])
roc_auc_tree = roc_auc_score(score['true'], score['score_tree'])

print("roc_auc_log:", roc_auc_log)
print("roc_auc_scm:", roc_auc_scm)
print("roc_auc_knn:", roc_auc_knn)
print("roc_auc_tree:", roc_auc_tree)

precision_log, recall_log, thresholds_log = precision_recall_curve(score['true'], score['score_logreg'])
precision_svm, recall_svm, thresholds_svm = precision_recall_curve(score['true'], score['score_svm'])
precision_knn, recall_knn, thresholds_knn = precision_recall_curve(score['true'], score['score_knn'])
precision_tree, recall_tree, thresholds_tree = precision_recall_curve(score['true'], score['score_tree'])

curve = {}

for i in score.keys()[1:]:
    df = pd.DataFrame(columns=('precision', 'recall'))
    df.precision, df.recall, thresholds = precision_recall_curve(score.true, score[i])
    curve[i] = df[df['recall'] >= 0.7]['precision'].max()
    
print(curve)