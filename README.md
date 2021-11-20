# Disaster Response Pipeline Project

![Python](https://img.shields.io/badge/Python%3D%3D-3.7.6-blueviolet)
![jupyter](https://img.shields.io/badge/jupyter%3D%3D-1.0.0-yellowgreen)
![nltk](https://img.shields.io/badge/nltk%3D%3D-3.6.5-blueviolet)
![pandas](https://img.shields.io/badge/pandas%3D%3D-1.3.4-yellowgreen)
![sklearn](https://img.shields.io/badge/sklearn%3D%3D-0.0-blueviolet)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy%3D%3D-1.4.26-yellowgreen)
![Plotly](https://img.shields.io/badge/plotly%3D%3D-5.4.0-blueviolet)
![Flask](https://img.shields.io/badge/jupyter%3D%3D-1.1.2-yellowgreen)



## Table of Contents

1. [Installation](#installation)
2. [Directory Structure](#dir_structure)
3. [Instructions](#instructions)
4. [Example Working](#examples)
5. [Analysis of the Various Categories](#analysis)

6. [About Me](#about)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*.

## Directory Structure <a name="dir_structure"></a>

![Directory Structure](./graphs/dir_structure.jpg "Directory Structure")

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/finalized_model.pickle`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

>Notes:
>1. Because the training of the model takes a long time (over 1hr), the pickle file has been included to avoid training everytime
>2. The pickle file has been zipped to be able to upload it to github. When you clone/download the repo, just unzip the pickle file

#Example Working<a name="examples"></a>
Te Home Page shows the analysis of all categories in the training data
![home_screen_cropped](./graphs/home_screen_cropped.png "home_screen_cropped")


This tweet https://twitter.com/GlobalFloodNews/status/1448943021576294430 has been classified into 4 catogories
![categorised_tweet](./graphs/categorised_tweet.png "categorised_tweet")


This tweet https://twitter.com/GraphQLGalaxy/status/1451291277958471682 is about a completely unrelated subject and returns no classification at all.
![unrelated_msg](./graphs/unrelated_msg.png "unrelated_msg")

# Analysis of the Various Categories <a name="analysis"></a>
## Less than one option categories
 ### Child Alone
All values of messages did not have any <i>child_alone</i> category (all were 0). This implies no tweet will ever have 
a prediction of 1.
![child_alone](./graphs/child_alone.png "child_alone")
Deleting this category from the model may not have impact on the outcome of predictions

##More than two option categories
###Related
The <i>related</i> category have 0, 1 & 2 options, with the 2 consisting of less than 1%.

![related](./graphs/related.png "child_alone")

##Imbalanced Categories
Majority of the categories had imbalanced classifications

###Aid Centers
This had only two classifications, 0 and 1, but were so imbalanced with 1 only accounting for 1.2% of all messages.
![aid_centers](./graphs/aid_centers.png "aid_centers") 

Other categories with this imbalanced message categorizations were:
- buildings with 1 at 5.1%
- clothing with 1 at 1.5%
- cold with 1 at 2%
- death with 1 at 4.6%
- earthquake with 1 at 9% 
- elecricity with 1 at 2%
- etc

##Balanced Categories
These had a relative closeness in the message categorizations with neither 0 nor 1 taking up less than 30% f the total message classifications
###Aid Related
Had a good message classification with 1 at 42% of the messages
![aid_related](./graphs/aid_related.png "aid_related")

##About Me <a name="about"></a>
- <a href="https://www.linkedin.com/in/mcoluga/">LinkedIn</a>
- <a href="https://twitter.com/McOluga">Twiiter</a>