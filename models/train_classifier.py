# import libraries
import pandas as pd
from sqlalchemy import create_engine 
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import _pickle as cPickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """
        Function to load data from a database 

        Inputs:
            database_filepath (path): location of the database
        Returns:
            X (pandas dataframe): messages (features)
            Y (pandas dataframe): categories (targets)
            category_names (list): category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from dataset', engine)
    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
        Function to process the text input

        Input:
            text: a string that needs to be tokenized
        Returns:
            lemmatized (list): processed words from the input text

    """
    #convert text to lower case
    normalized = text.lower()
    #remove punctuations
    normalized = re.sub(r'[^A-Za-z0-9]', ' ', normalized)
    #normalize text
    words = word_tokenize(normalized)
    #remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]

    #perform lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, pos='n') for w in words]
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in lemmatized]
    
    return lemmatized


def build_model():
    """
        Function to build an ML model

        Inputs:
            None
        Returns:
            GridSearchCV object
    """
    
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
        ])

    parameters = {'tfidf__use_idf': [True, False],
             'clf__estimator__n_estimators': [10, 20]}

    cv = GridSearchCV(pipeline, parameters, verbose=True)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Function to evaluate model performance

        Inputs:
            model (sklearn predictor): fitted ML model
            X_test (pandas dataframe): features for the test set
            Y_test (pandas dataframe): target for the test set
            category_names (list): individual category names
        Outputs:
            Classification Report, Accuracy Score for each category
    """
    Y_pred = model.predict(X_test)

    for en, c in enumerate(category_names):
        print('*'*60)
        print('Category Name: {}'.format(c))
        print('Accuracy: {:.2f}'.format(accuracy_score(Y_test.iloc[:, en].values, Y_pred[:, en])))
        print('Classification Report')
        print(classification_report(Y_test.iloc[:, en].values, Y_pred[:, en]))
        print('*'*60)

def save_model(model, model_filepath):
    """
        Function to save the trained model

        Inputs:
            model (sklearn predictor): a fitted model to be saved
            model_filepath (path): location onto which the model is to be saved
        Returns:
            None
    """

    with open(model_filepath, 'wb') as f:
        cPickle.dump(model, f)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()