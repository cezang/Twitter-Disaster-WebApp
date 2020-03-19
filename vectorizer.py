from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import pandas as pd
from model import dictionary
import pickle


cur_dir = os.path.dirname(__file__)

def preprocessor(text):
    emoticons = re.findall(r'(?::|;|=|x|X)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ', text.lower()) + \
        ' '.join(emoticons).replace('-','')
    text = re.sub('[#,=,<,>,@,\.,\,,+]','',text)
    text = re.sub(r'http\S+', 'http', text.lower())
    return text

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

vect = TfidfVectorizer(
                       ngram_range=(1,1),
                       vocabulary=dictionary,
    strip_accents=None,
    lowercase=False,
    preprocessor=preprocessor,
    tokenizer=tokenizer_porter)
print('vectorizer')