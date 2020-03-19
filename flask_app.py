from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# importuje obiekt HashingVectorizer z lokalnego katalogu
from vectorizer import vect

app = Flask(__name__)

######## Przygotowuje klasyfikator
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'nlp_classifier',
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'tweet.sqlite')

def classify(document):
    label = {0: 'not real disaster', 1: 'real disaster'}
    X = vect.fit_transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.fit_transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO Tweet_db (Tweet, Fake)"\
    " VALUES (?, ?)", (document, y))
    conn.commit()
    conn.close()

######## Flask
class TweetForm(Form):
    tweet = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = TweetForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['tweet']
        y, proba = classify(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'not real disaster': 0, 'real disaster': 1}
    y = inv_label[prediction]
    if feedback == 'not Correct':
        y = int(not(y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
