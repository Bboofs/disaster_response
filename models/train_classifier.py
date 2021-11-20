# import libraries
import re
import sys
import time
import pickle
import pandas as pd
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    sql = 'SELECT * from messages_and_categories;'
    df = pd.read_sql(sql, engine)

    y_cols = ['related', 'request', 'offer',
              'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
              'security', 'military', 'child_alone', 'water', 'food', 'shelter',
              'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
              'infrastructure_related', 'transport', 'buildings', 'electricity',
              'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
              'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
              'other_weather', 'direct_report']

    X = df['message']
    y = df[y_cols]

    return X, y, y_cols


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('knn_clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    knn_est_params = {
        # 'knn_clf__estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'knn_clf__estimator__n_neighbors': [5, 10, 15],
        'knn_clf__estimator__weights': ['uniform', 'distance'],
        # 'knn_clf__metric': ['euclidian', 'manhattan']
    }

    cv = GridSearchCV(pipeline, param_grid=knn_est_params)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    start = datetime.now()
    print('Starting at:', start)
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        tic = datetime.now()
        print('Loading data...{}\n    DATABASE: {}'.format(tic, database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        toc = datetime.now()
        print('Loading data took:', toc - tic)

        tic = datetime.now()
        print('Building model...', tic)
        model = build_model()
        toc = datetime.now()
        print('Building model took:', toc - tic)

        tic = datetime.now()
        print('Training model...', tic)
        model.fit(X_train, Y_train)
        toc = datetime.now()
        print('Training model took:', toc - tic)

        tic = datetime.now()
        print('Evaluating model...', tic)
        evaluate_model(model, X_test, Y_test, category_names)
        toc = datetime.now()
        print('Evaluating model took:', toc - tic)

        tic = datetime.now()
        print('Saving model...{}\n    MODEL: {}'.format(tic, model_filepath))
        save_model(model, model_filepath)
        toc = datetime.now()
        print('Saving model took:', toc - tic)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
