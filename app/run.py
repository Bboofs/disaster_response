import json
import plotly
import pandas as pd
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_and_categories', engine)

# load model
model = joblib.load("../models/finalized_model.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    sub_cols = ['related', 'request', 'offer',
                'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                'infrastructure_related', 'transport', 'buildings', 'electricity',
                'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                'other_weather', 'direct_report']

    sub_cols.sort()
    print(sub_cols)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    for col in sub_cols:
        title = col.replace('_', ' ').title()
        col_dict = {
            'data': [
                Bar(
                    x=(df[col].value_counts(normalize=True)*100).tolist(),
                    y=[str(y_val) for y_val in df[col].value_counts(normalize=True).keys().tolist()],
                    orientation='h'
                )
            ],

            'layout': {
                'title': "Distribution of " + title,
                'yaxis': {
                    'title': "Category of " + title,
                    'type': 'category'
                },
                'xaxis': {
                    'title': "% Classification of " + title,
                    'range': [0, 100]
                }
            }
        }
        graphs.append(col_dict)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
