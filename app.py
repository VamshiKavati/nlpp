import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import pandas as pd
import nltk
import re
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():

    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    text = [request.form['text1']]
    text1 = text[0]

    def corpus_fun(data):
        data1 = []
        for i in range(len(data)):
            review = re.sub('[^a-zA-Z]', ' ', data[i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            review = [ps.stem(word)
                      for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            data1.append(review)
        return data1

    corpus = corpus_fun(dataset['Review'])
    text = corpus_fun(text)

    corpus.append(text[0])

    def vectorization(corpus):
        cv = CountVectorizer(max_features=1500)
        X = cv.fit_transform(corpus).toarray()
        return X

    X = vectorization(corpus)
    y = dataset.iloc[:, -1].values
    test = [X[-1]]
    X = X[:-1]
    X_train, y_train = X, y
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    def pred(data):
        return classifier.predict(data)

    final = pred(test)
    if final == 0:
        final = 2
    return render_template('index.html', final=final, text1=text1)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.3", port=5002, threaded=True)
