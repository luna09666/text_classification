from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
from TFIDF import *
from countVec import *
from LDA import *
from sfm import *
from fasttextpt import *


print("\t\t准确率\t召回率\tF1指数\n")
for k in ["TF-IDF", "countVec", "LDA", "SFM"]:
    List = ["SVC", "Perceptron", "RandomForest"]
    if(k == "SFM"):
        List = ["Perceptron", "RandomForest"]
    for i in List:
        print(k[:], "+", i[:3] + '\t', end="")
        if k == "TF-IDF":
            y_pred = pre_tfidf(i)
        elif k == "countVec":
            y_pred = pre_countVec(i)
        elif k == "LDA":
            y_pred = pre_LDA(i)
        else:
            y_pred = pre_sfm(i)
        p = precision_score(test_flag, y_pred, average='weighted')
        r = recall_score(test_flag, y_pred, average='weighted')
        f1 = f1_score(test_flag, y_pred, average='weighted')
        print("%.3f\t"%p, end="")
        print("%.3f\t"%r, end="")
        print("%.3f"%f1)
    print()

y_pred = predicted
print("fasttext\t", end="")
p = precision_score(test_flag, y_pred, average='weighted')
r = recall_score(test_flag, y_pred, average='weighted')
f1 = f1_score(test_flag, y_pred, average='weighted')
print("%.3f\t"%p, end="")
print("%.3f\t"%r, end="")
print("%.3f"%f1)
