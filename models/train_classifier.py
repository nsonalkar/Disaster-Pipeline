import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster',engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X,Y, list(Y.columns)

def tokenize(text):
    text = text.lower()
    p = re.compile('[:,.!?]')
    text = re.sub(p,'',text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w,pos='v') for w in words]
    return lemmed


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'clf__estimator__max_features':['sqrt', 0.5],
              'clf__estimator__n_estimators':[50, 100],
              
             }

    cv = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = cv.predict(X_test)
    for i in range(len(category_names)):
        print(f1_score(Y_test.iloc[:,i],y_pred[:,i],average=None))
        print(precision_score(Y_test.iloc[:,i],y_pred[:,i],average=None))
        print(recall_score(Y_test.iloc[:,i],y_pred[:,i],average=None))


def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


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