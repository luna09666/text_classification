
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from build_data import *

text_vect = CountVectorizer(max_df=0.2).fit(train_data + test_data)
all_vect = text_vect.transform(train_data + test_data)
word_vect = text_vect.transform(train_data)
test_vect = text_vect.transform(test_data)

lda = LatentDirichletAllocation(n_components=200, max_iter=3, learning_method='online')
text_lda = lda.fit(all_vect)
word_lda = text_lda.transform(word_vect)
test_lda = text_lda.transform(test_vect)

def pre_LDA(pre):
    global predicted
    if pre == "SVC":
        text_clf = SVC(C=50).fit(word_lda, train_flag)
        predicted = text_clf.predict(test_lda)
    elif pre == "Perceptron":
        text_clf = Perceptron().fit(word_lda, train_flag)
        predicted = text_clf.predict(test_lda)
    elif pre == "RandomForest":
        text_clf = RandomForestClassifier(n_estimators=250).fit(word_lda, train_flag)
        predicted = text_clf.predict(test_lda)
    return predicted

