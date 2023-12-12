from snownlp import SnowNLP
from snownlp import sentiment
import pandas as pd
import utils
import os

from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

dirlist = os.listdir()


def models(commands):
    words_list = utils.word_frequency_statistics()
    for comm in commands:
        if comm == 'nan':
            continue
        else:
            comm = utils.Extract_Commands(comm).extract_command()
            for w in words_list:
                if w in comm:
                    score = SnowNLP(comm)
                    # 预训练模型分类
                    if score.sentiments > 0.8:
                        with open('pos.txt', mode='a', encoding='utf-8') as p:
                            p.writelines(comm + '\n')
                    elif score.sentiments < 0.2:
                        with open('neg.txt', mode='a', encoding='utf-8') as n:
                            n.writelines(comm + '\n')
                    break


def train_snownlp():
    if 'raw_data.csv' in dirlist:
        df = pd.read_csv('raw_data.csv')
    else:
        raise Exception('请先创建raw_data.csv文件')
    df.fillna('nan')
    commands = df.评论内容.dropna().tolist()
    models(commands)
    sentiment.train('neg.txt', 'pos.txt')
    sentiment.save('mysentiment.marshal')
    print('得到模型后需拷贝到snownlp的sentiment文件夹下\
        并修改__init.py__的路径来加载新权重')

def train_multinomial_nb():

    pos_reviews = pd.read_csv('pos_reviews.txt', header=None, names=['review'])
    pos_reviews['label'] = 1  # Positive sentiment

    neg_reviews = pd.read_csv('neg_reviews.txt', header=None, names=['review'])
    neg_reviews['label'] = 0  # Negative sentiment

    # Combine the dataframes
    reviews = pd.concat([pos_reviews, neg_reviews])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(reviews['review'], reviews['label'], test_size=0.2,
                                                        random_state=42)

    # Create a pipeline to vectorize the data, then train and fit a model
    nb_pipeline = Pipeline([
        ('vect', CountVectorizer()),  # Convert text to counts
        ('tfidf', TfidfTransformer()),  # Convert counts to TF-IDF
        ('clf', MultinomialNB()),  # Train a Naive Bayes classifier
    ])

    # Train the model with the training data
    nb_pipeline.fit(X_train, y_train)

    # # Score the sentiment based on the positive class probability
    # sentiment_score = predictions[:, 1] * 10 - 5  # Scaling to -5 to 5 range


    # from joblib import dump, load
    # # Save the pipeline to a file
    # dump(nb_pipeline, 'sentiment_classifier.joblib')
    #
    # # At a later point, you can load the pipeline back into memory
    # # Load the pipeline from a file
    # nb_pipeline_loaded = load('sentiment_classifier.joblib')
    #
    # # Now you can use the loaded pipeline to make predictions as before
    # new_review = ["This product is great!"]
    # new_prediction = nb_pipeline_loaded.predict_proba(new_review)
    # new_sentiment_score = new_prediction[:, 1] * 10 - 5
    # print(f"Sentiment score for new review: {new_sentiment_score}")

    # # Predict the sentiment for the testing data
    # predictions = nb_pipeline.predict_proba(X_test)


    # Print the classification report
    # print(classification_report(y_test, nb_pipeline.predict(X_test)))
    #
    # # Example of using the model to predict a new review
    # new_review = ["This product is great!"]
    # new_prediction = nb_pipeline.predict_proba(new_review)
    # new_sentiment_score = new_prediction[:, 1] * 10 - 5
    # print(f"Sentiment score for new review: {new_sentiment_score}")

