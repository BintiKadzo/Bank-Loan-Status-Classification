
# Loading the libraries
import xgboost as xgb
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
from warnings import filterwarnings
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from scipy.stats import bartlett, chi2, loguniform
import os
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import bartlett
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import time
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from PIL import Image
from bs4 import BeautifulSoup
import networkx as nx
import pickle
from PIL import Image
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import load_data
from load_data import Data

sb.set(rc={'figure.figsize': (30, 5)})
filterwarnings('ignore')
# df = pd.read_csv("credit_train.csv")
# let's create an object of class Data
#loan_train = Data('credit_train.csv')
#loan_test = Data('credit_test.csv')

# Streamlit
selected = st.sidebar.selectbox(
    "Select the section", ['Admin', 'Prediction', 'Recommendation'])

siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()


pickle_in = open('trained_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

if selected == 'Admin':

    def prediction(Loan_Status, Current_Loan_Amount, Term, Annual_Income, Years_in_current_job, Home_Ownership, Purpose,
                   Monthly_Debt, Years_of_Credit_History, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit,
                   Bankruptcies, Tax_Liens):
        values = [[Loan_Status, Current_Loan_Amount, Term, Annual_Income, Years_in_current_job, Home_Ownership, Purpose,
                   Monthly_Debt, Years_of_Credit_History, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit,
                   Bankruptcies, Tax_Liens]]

        # with st.form(key='my_form'):
        #     name = st.text_area("Name: ", key="<255>")
        #     loanId = st.text_area("Loan Id: ", key="<254>")
        #     customerId = st.text_area("Customer Id: ", key="<253>")
        # Term = st.text_area("Term: ", key="<252>")
        # # credit = st.number_input("Credit Class:", key=251)
        # Annual_Income = st.number_input("Annual Income:", key=256)
        # Maximum_Open_Credit = st.number_input(
        #     "Maximum Open Credit:", key=256)
        # Current_Loan_Amount = st.number_input(
        #     "Current Loan Amount:", key=256)
        # Monthly_Debt = st.number_input("Monthly Dept:", key=260)
        # Years_of_Credit_History = st.slider('Years in curent Job', min_value=0,
        #                                     max_value=20, value=2, step=1)
        # Current_Credit_Balance = st.number_input(
        #     "Current Credit Balance Amount:", key=256)
        # Number_of_Open_Accounts = st.number_input(
        #     "Number of open Accounts:", key=256)
        # Years_in_current_job = st.slider('Years in current Job', min_value=0,
        #                                  max_value=20, value=2, step=1)
        # Purpose = st.selectbox(
        #     'Purpose', ('Debt Consolidation', 'Home Improvements'))
        # Loan_Status = st.selectbox(
        #     'Loan Status', ('Fully Paid', 'Charged Off'))
        # Home_Ownership = st.selectbox(
        #     'Home Ownership', ('Home Mortgage', 'Rent', 'Own Home'))
        # Number_of_Credit_Problems = st.number_input(
        #     "Number of Credit Problems:", key=256)
        # Bankruptcies = st.selectbox(
        #     'Bankruptcies', ('0', '1'))
        # Tax_Liens = st.number_input("Tax Liens:", key=260)
        # credit = st.number_input(
        #     "What number would you like to be contacted with?:", key=251)

        # submit_button = st.form_submit_button(label='Submit')
        prediction = classifier(values)
        if prediction == [0]:
            pred = 'Fair. Your Loan Limit is upto 10M'
        elif prediction == [1]:
            pred = 'Good. Your Loan Limit is upto 11M'
        else:
            pred = 'Very good. Your Loan Limit is upto 23M'
        return pred

    def main():

        st.title('LOAN CLASSIFICATION APPLICATION')
        with st.form(key='my_form'):
            name = st.text_area("Name: ", key="<255>")
            loanId = st.text_area("Loan Id: ", key="<254>")
            customerId = st.text_area("Customer Id: ", key="<253>")
            Term = st.selectbox(
                'Term', ('Short Term', 'Long Term'))
            # credit = st.number_input("Credit Class:", key=251)
            Annual_Income = st.number_input("Annual Income:", key=256)
            Maximum_Open_Credit = st.number_input(
                "Maximum Open Credit:", key=256)
            Current_Loan_Amount = st.number_input(
                "Current Loan Amount:", key=256)
            Monthly_Debt = st.number_input("Monthly Debt:", key=260)
            Years_of_Credit_History = st.slider('Years of Credit History', min_value=0,
                                                max_value=20, value=2, step=1)
            Current_Credit_Balance = st.number_input(
                "Current Credit Balance Amount:", key=256)
            Number_of_Open_Accounts = st.number_input(
                "Number of open Accounts:", key=256)
            Years_in_current_job = st.slider('Years in current Job', min_value=0,
                                             max_value=20, value=2, step=1)
            Purpose = st.selectbox(
                'Purpose', ('Debt Consolidation', 'Home Improvements', 'Buy House', 'Business Loan', 'Major Purchase', 'Take a Trip', 'Small Business', 'Medical Bills', 'Wedding', 'Vacation', 'Education', 'Moving', 'Renweable Energy', 'Other'))
            Loan_Status = st.selectbox(
                'Loan Status', ('Fully Paid', 'Charged Off'))
            Home_Ownership = st.selectbox(
                'Home Ownership', ('Have Mortgage', 'Home Mortgage', 'Rent', 'Own Home'))
            Number_of_Credit_Problems = st.number_input(
                "Number of Credit Problems:", key=256)
            Bankruptcies = st.slider(
                'How Many Times Has This Client Filled For Bankruptcies?',  min_value=0,
                max_value=100, value=2, step=1)
            Tax_Liens = st.number_input(
                "Legal Claims against this client:", key=260)
            credit = st.number_input(
                "What number would you like to be contacted with?:", key=251)
            # submit_button = st.form_submit_button(label='Submit')

    # Converting values into Integer values

            if Purpose == 'Debt Consolidation':
                Purpose = 3
            elif Purpose == 'Home Improvements':
                Purpose = 5
            elif Purpose == 'Buy House':
                Purpose = 1
            elif Purpose == 'Business Loan':
                Purpose = 2
            elif Purpose == 'Major Purchase':
                Purpose = 9
            elif Purpose == 'Take a Trip':
                Purpose = 8
            elif Purpose == 'Small Business':
                Purpose = 13
            elif Purpose == 'Medical Bills':
                Purpose = 6
            elif Purpose == 'Wedding':
                Purpose = 15
            elif Purpose == 'vacation':
                Purpose = 14
            elif Purpose == 'Education':
                Purpose = 4
            elif Purpose == 'Moving':
                Purpose = 10
            elif Purpose == 'Renewable Energy':
                Purpose = 12
            else:
                Purpose = 7

            if Term == 'Short Term':
                Term = 0
            else:
                Term = 1

            if Loan_Status == 'Fully Paid':
                Loan_Status = 1
            else:
                Loan_Status = 0

            if Home_Ownership == 'Have Mortgage':
                Home_Ownership = 0
            elif Home_Ownership == 'Home Mortgage':
                Home_Ownership = 1
            elif Home_Ownership == 'Own Home':
                Home_Ownership = 2
            elif Home_Ownership == 'Rent':
                Home_Ownership = 3

            if Years_in_current_job == 7:
                Years_in_current_job = 7
            elif Years_in_current_job == 0:
                Years_in_current_job = 10
            elif Years_in_current_job == 1:
                Years_in_current_job = 0
            elif Years_in_current_job == 2:
                Years_in_current_job = 2
            elif Years_in_current_job == 3:
                Years_in_current_job = 3
            elif Years_in_current_job == 4:
                Years_in_current_job = 4
            elif Years_in_current_job == 5:
                Years_in_current_job = 5
            elif Years_in_current_job == 6:
                Years_in_current_job = 6
            elif Years_in_current_job == 8:
                Years_in_current_job = 8
            elif Years_in_current_job == 9:
                Years_in_current_job = 9
            elif Years_in_current_job >= 10:
                Years_in_current_job = 1

            prediction = ""
            # if Term and Annual_Income and Maximum_Open_Credit and Current_Loan_Amount and Monthly_Debt and Years_of_Credit_History and Current_Credit_Balance and Number_of_Open_Accounts and Years_in_current_job and Purpose and Loan_Status and Home_Ownership and Number_of_Credit_Problems and Bankruptcies and Tax_Liens:
            if st.form_submit_button("Predict"):
                prediction = prediction(Loan_Status, Current_Loan_Amount, Term, Annual_Income, Years_in_current_job, Home_Ownership, Purpose, Monthly_Debt, Years_of_Credit_History, Number_of_Open_Accounts, Number_of_Credit_Problems, Current_Credit_Balance, Maximum_Open_Credit,
                                        Bankruptcies, Tax_Liens)
                st.success('Your FICO SCORE is {}'.format(prediction))
    if __name__ == "__main__":
        main()

    #        if Term == "long_term".upper():
    #             Term = 1
    #         else:
    #             Term = 0

    #         elif Home_Ownership == "Home Mortgage":
    #             Home_Ownership = 1
    #         elif Home_Ownership == "Rent":
    #             Home_Ownership = 2
    #         else Home_Ownership == "Own Home"

    #     # class imbalance

    #     # let's import counters from imbalanced-learn
    #     from collections import Counter

    #     # import smote(synthetic minority over-sampling)
    #     from imblearn.over_sampling import SMOTE

    #     # define independent and dependent variables
    #     X = loan_train.data.drop(['Credit_Score', 'Credit_Class'],
    #                              axis=1)    # independent variables
    #     y = loan_train.data['Credit_Class']     # dependent variables

    #     # transform the dataset
    #     oversample = SMOTE()
    #     X, y = oversample.fit_resample(X, y)

    #     # summarize the new class distribution
    #     counter = Counter(y)
    #     print(counter)
    #     # let's divide the data into independent and dependent variables

    #     # divide the data into train and test split
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.2, random_state=0)

    # # standardize the data
    #     sc = StandardScaler()
    #     X_train = sc.fit_transform(X_train)
    #     X_test = sc.transform(X_test)

    # # create a xgboost object
    #     xgb = XGBClassifier(eval_metric='mlogloss', learning_rate=0.300000012,
    #                         max_depth=6, n_estimators=500, n_jobs=16, random_state=0)

    # # fit the model to our data
    #     xgb.fit(X_train, y_train)

    # # let's predict the test data
    #     y_pred = xgb.predict(X_test)

    # # let's check the accuracy of the model using confusion matrix
    #     confusion_matrix(y_test, y_pred)

    # # deployment
    #     file_name = 'trained_model.pkl'
    #     pickle.dump(xgb, open(file_name, 'wb'))

    # # load the saved model
    #     loaded_model = pickle.load(open('trained_model.pkl', 'rb'))
    # st.header("Training")
    # st.write("training in progres ...")
    # df[['Label']] = df[['Label']].apply(LabelEncoder().fit_transform)
    # tokenizer = Tokenizer(oov_token="<OOV>")
    # split = round(len(df)*0.8)
    # train_reviews = df['URL'][:split]
    # train_label = df['Label'][:split]
    # test_reviews = df['URL'][split:]
    # test_label = df['Label'][split:]
    # training_sentences = []
    # training_labels = []
    # testing_sentences = []
    # testing_labels = []
    # for row in train_reviews:
    #     training_sentences.append(str(row))
    # for row in train_label:
    #     training_labels.append(row)
    # for row in test_reviews:
    #     testing_sentences.append(str(row))
    # for row in test_label:
    #     testing_labels.append(row)
    # vocab_size = 20000
    # embedding_dim = 16
    # max_length = 120
    # trunc_type = 'post'
    # oov_tok = '<OOV>'
    # padding_type = 'post'
    # tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # tokenizer.fit_on_texts(training_sentences)
    # word_index = tokenizer.word_index
    # sequences = tokenizer.texts_to_sequences(training_sentences)
    # padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    # testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
    # testing_padded = pad_sequences(testing_sentences, maxlen=max_length)
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding(vocab_size, embedding_dim,
    #                             input_length=max_length),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(6, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    # model.compile(optimizer='adam', metrics=[
    #               'accuracy'], loss='binary_crossentropy')
    # training_labels_final = np.array(training_labels)
    # testing_labels_final = np.array(testing_labels)
    # num_epochs = 1
    # history = model.fit(padded, training_labels_final, epochs=num_epochs,
    #                     validation_data=(testing_padded, testing_labels_final))
    # st.write("Congratulations,Training is completed")
    # st.header("Prediction")
    # data = st.text_input("Insert the URL to test here")
    # link = data
    # st.write('Prediction Started ...')
    # t0 = time.perf_counter()
    # data = str(data)
    # tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # tokenizer.fit_on_texts(data)
    # word_index = tokenizer.word_index
    # sequences = tokenizer.texts_to_sequences(data)
    # padded_data = pad_sequences(
    #     sequences, maxlen=max_length, truncating=trunc_type)
    # score = model.predict(padded_data).round(0).astype('int')
    # score = np.average(score)
    #  t1 = time.perf_counter() - t0
    #   st.write('Prediction Completed\nTime taken', t1, 'sec')
    #    if score <= 0.2:
    #         st.write(
    #             "The URL is probaly a phising URL. Kindly read through the reccomendations")
    #     else:
    #         st.write("The website is secure, kindly click the link to proceed")
    #         st.write(link)
    #         st.write("Thank You!")
