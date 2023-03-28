from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from build_data import *

text_vect = TfidfVectorizer(max_df=0.2).fit(train_data)
word_vect = text_vect.transform(train_data)
test_vect = text_vect.transform(test_data)

def pre_sfm(pre):
    if pre == "Perceptron":
        per = Perceptron(shuffle=True, max_iter=500, random_state=41).fit(word_vect, train_flag)
        sfm = SelectFromModel(per)
        sfm.fit(word_vect, train_flag)
        word_sfm = sfm.transform(word_vect)
        test_sfm = sfm.transform(test_vect)
        text_clf = Perceptron(shuffle=True, max_iter=500, random_state=41).fit(word_sfm, train_flag)
        predicted = text_clf.predict(test_sfm)
    else:
        RFC = RandomForestClassifier(oob_score=True, random_state=418).fit(word_vect, train_flag)
        sfm = SelectFromModel(RFC)
        sfm.fit(word_vect, train_flag)
        word_sfm = sfm.transform(word_vect)
        test_sfm = sfm.transform(test_vect)
        text_clf = RandomForestClassifier(oob_score=True, random_state=418).fit(word_sfm, train_flag)
        predicted = text_clf.predict(test_sfm)
    return predicted



