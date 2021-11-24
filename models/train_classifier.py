# import libraries
import re
import sys
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt

from datetime import datetime
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
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

    print("Loaded data {} rows by {} columns.".format(df.shape[0], df.shape[1]))
    print('*' * 65)
    print('{0:25}{1:>20}{2:>20}'.format('COLUMN', 'TOTAL NO. OF NULLS', 'NULLS (%)'))

    for col in df.isnull().sum().keys():
        print('{0:25}{1:>20}{2:>20.2f}'.format(col, df.isnull().sum()[col], df.isnull().sum()[col] * 100 / df.shape[0]))

    df.drop(df[df['related'] == 2].index, inplace=True)
    df.drop(columns=['original'], inplace=True)

    print('*' * 65)
    print("Only the 'original' column has nulls, but it's not used. It's safe to delete.")

    print('*' * 65)
    print("Dropping {} 'related' rows with values as 2.".format(df[df['related'] == 2].shape[0]))

    print('*' * 65)
    print("Loaded data now has {} rows by {} columns.".format(df.shape[0], df.shape[1]))

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
    # accuracy = (y_pred == y_test).mean()
    # print("Accuracy:", accuracy)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    print('Confusion Matrix, Precision, Recall, Accuracy, and F1 Score')

    for col in category_names:
        n = random.randint(0, len(plt.colormaps()))
        colour_map = plt.colormaps()[n]
        conf_matrix = confusion_matrix(y_true=y_test[col], y_pred=y_pred_df[col])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(conf_matrix, cmap=colour_map)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predicted Values', fontsize=16)
        plt.ylabel('Actual Values', fontsize=16)
        plt.title('Confusion Matrix: ' + col.replace('_', ' ').title(), fontsize=16)
        try:
            plot_name = '../graphs/matrix_' + col + '.png'
            plt.savefig(plot_name)
        except Exception as e:
            print(str(e))
        plt.draw()
        print('Precision: %.3f' % precision_score(y_test[col], y_pred_df[col]))
        print('Recall: %.3f' % recall_score(y_test[col], y_pred_df[col]))
        print('Accuracy: %.3f' % accuracy_score(y_test[col], y_pred_df[col]))
        print('F1 Score: %.3f' % f1_score(y_test[col], y_pred_df[col]))
    plt.show()


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
