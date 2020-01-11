import pandas as pd
import scipy
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import  LinearSVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import timeit
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


data = pd.read_csv("reddit_train.csv")
data.drop ('id', axis=1, inplace=True)

data = sklearn.utils.shuffle(data)

X_train, X_validate, y_train, y_validate = train_test_split(data.comments, data.subreddits, train_size=0.70, test_size=0.30)
X_train = X_train.values.astype('U')
X_validate = X_validate .values.astype('U')


# apply word level tf-idf vectorization
vectorizer = TfidfVectorizer( analyzer = 'word', sublinear_tf = True, norm='l2', max_df=0.6, min_df=2, ngram_range=(1,3))
vectors_train1 = (vectorizer.fit_transform(X_train))
vectors_validate1 = (vectorizer.transform(X_validate))

# scale down word features to 25000
selector = SelectKBest(chi2, k=25000)
selector.fit(vectors_train1, y_train)
vectors_train1 = selector.transform(vectors_train1)
vectors_validate1 = selector.transform(vectors_validate1)

# apply character level tfidfVectorizer
char_vectorizer = TfidfVectorizer( analyzer = 'char_wb', stop_words='english', sublinear_tf = True, norm='l2', max_df=0.5, min_df=4, ngram_range=(2,5))
vectors_train2 = (char_vectorizer.fit_transform(X_train))
vectors_validate2 = (char_vectorizer.transform(X_validate))


#scale down character features to 8000
selector2 = SelectKBest(chi2, k=8000)
selector2.fit(vectors_train2, y_train)
vectors_train2 = selector2.transform(vectors_train2)
vectors_validate2 = selector2.transform(vectors_validate2)

# combine word level vectors and character level vectors
vectors_train = scipy.sparse.hstack([vectors_train1, vectors_train2])
vectors_validate = scipy.sparse.hstack([vectors_validate1, vectors_validate2])

# normalize vectors
vectors_train = normalize(vectors_train)
vectors_validate = normalize(vectors_validate)


#base models
sgdc= linear_model.SGDClassifier(penalty='l2')
svc=LinearSVC(class_weight='balanced', loss='squared_hinge', C=0.3)
nb = MultinomialNB(alpha=0.0001)
lr = LogisticRegression(multi_class='auto',max_iter=100, C=1, fit_intercept=False)
rf = ExtraTreesClassifier(n_estimators=100,  random_state=0)



# base line for meta model
scores = cross_val_score(nb, scipy.sparse.vstack((vectors_train, vectors_validate)), np.append(y_train, y_validate), cv=2)
print("The prediction accuracy by NB (base line for meta model) is:")
print(scores)

# construct meta model
start = timeit.default_timer()
eclf = VotingClassifier(estimators=[('nb', nb), ('sgdc',sgdc), ('svc',svc), ('lr',lr)], weights=[4.9,2,2,1], voting = 'hard')
eclf.fit(vectors_train, y_train)
stop = timeit.default_timer()
print('Training Time: ', stop - start)

start = timeit.default_timer()
y_pred = eclf.predict(vectors_validate)
stop = timeit.default_timer()
print('Prediction Time: ', stop - start)
#print(metrics.accuracy_score(y_validate, y_pred))


# accuracy of meta model
scores = cross_val_score(eclf, scipy.sparse.vstack((vectors_train, vectors_validate)), np.append(y_train, y_validate), cv=2)
print("The prediction accuracy by ensemble voting model is:")
print(scores)
