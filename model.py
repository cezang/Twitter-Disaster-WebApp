import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.svm import SVC
import re
from sklearn.linear_model import SGDClassifier
import os
#import data
train_set = pd.read_csv('dane/train.csv')
os.chdir(os.path.dirname(__file__))
#preprocessing process
def preprocessor(text):
    emoticons = re.findall(r'(?::|;|=|x|X)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ', text.lower()) + \
        ' '.join(emoticons).replace('-','')
    text = re.sub('[#,=,<,>,@,\.,\,,+]','',text)
    text = re.sub(r'http\S+', 'http', text.lower())
    return text

text1 = train_set[['text','target']].copy()


#tokenizer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#X and y
X = text1['text'].values
y = text1['target'].values

#model
tfidf = TfidfVectorizer(ngram_range=(1,1),
    strip_accents=None,
    lowercase=False,
    preprocessor=preprocessor,
    tokenizer=tokenizer_porter)

X = tfidf.fit_transform(X)

dictionary = tfidf.vocabulary_
print('len of vocabulary:', len(tfidf.vocabulary_))
clf = SGDClassifier(loss='log')
#clf = SVC(kernel='linear', C=1.0, gamma=0.0001, probability=True)
clf.fit(X, y)

#savig model
import pickle
import os
dest = os.path.join('nlp_classifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
print('model')