#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:05:55 2022

@author: celia
"""

import json
import requests
import pandas as pd
import re
from pandas.io.json import json_normalize
import numpy as np
from itertools import combinations
import random
import io
import base64
import datetime as dt
import time


import sys

# graphiques
import matplotlib.pyplot as plt
import seaborn as sns

# nuage de mots
from PIL import Image
from wordcloud import WordCloud

#Dashboard
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.figure_factory as ff


#connexion base de données
import pymongo
from pymongo import MongoClient
from bson import ObjectId

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler


# MAchine learning
from sklearn import datasets, ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

# Score of models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import classification_report



def regex_text_to_df_avec_MongoDB(fichier_texte, bdd, collection):
    # Connection au serveur MongoDB
    client = MongoClient('mongodb://localhost/')

    # Récupération de la BDD nomée Projet_6_Spam_Celia_Cesar
    db = client[bdd]
    print(db)

    # Récupération de la collection sms
    collec = db[collection]
    print(collec)
    
    # Cette partie permet de charger les données dans la ddb
    # généralement fait une seule fois
    with open(fichier_texte, 'r') as f:
        for line in f:
            line=line.replace('\n', '') # suppression du CR en fin de line
            (type, text) = line.split('\t')
            obj = {"type":type, "text":text}
            #print(obj)
            #collec.insert_one(obj)
    
    # Définition des fonctions pour les regex
    def get_len(txt):
        l = len(txt)
        return l

    def get_email(txt):
        email_list = re.findall("[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,4}", txt)
        return email_list

    def get_url(txt):
        url_list=re.findall('https?://|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
        return url_list

    def get_emoticone(txt):
        emoticone_list=re.findall("(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)",txt)
        return emoticone_list

    def get_specharacter(txt):
        spechar_list = re.findall('[!]{2,}|[.]{3,}|[!?.]',txt)
        return spechar_list

    def get_majuscule(txt):
        majuscule_list= re.findall('[A-Z]{3,}',txt)
        return majuscule_list

    def get_moneysymbol(txt):
        moneysymbol_list = re.findall("£|\$|\€|\¥",txt)
        return moneysymbol_list

    def get_phonenumber(txt):
        phone_list = re.findall("[0-9]{10}",txt)
        return phone_list

    def get_date(txt):
        date_list=re.findall('^\d+.\d+.\d+|\d+/\d+/\d+|\d+-\d+-\d+|[\d]{1,2} [ADFJMNOS]\w* [\d]{4}',txt)
        return date_list

    def get_wordcall(txt):
        call_list=re.findall('(?i)(call)',txt)
        return call_list   

    def get_wordfree(txt):
        free_list=re.findall('(?i)(free)',txt)
        return free_list   

    def get_wordmobile(txt):
        mobile_list=re.findall('(?i)(mobile)',txt)
        return mobile_list   

    def get_wordprize(txt):
        prize_list=re.findall('(?i)(prize)',txt)
        return prize_list  


    def get_wordclaim(txt):
        claim_list=re.findall('(?i)(claim)',txt)
        return claim_list  

    def get_wordwon(txt):
        won_list=re.findall('(?i)(won)',txt)
        return won_list   

    def get_wordwin(txt):
        win_list=re.findall('(?i)(win)',txt)
        return win_list   

    def get_wordcash(txt):
        cash_list=re.findall('(?i)(cash)',txt)
        return cash_list   

    def get_wordlove(txt):
        love_list=re.findall('(?i)(love)',txt) 
        return love_list  

    def get_wordcontact(txt):
        contact_list=re.findall('(?i)(contact)',txt)
        return contact_list  

    def get_avg_lenword(txt): #pour compter nb de caracteres, nombre de mots et la longueur moyenne d'un mot 😄
        total_chars = len(re.sub(r'[^a-zA-Z0-9]', '', txt))
        num_words = len(re.sub(r'[^a-zA-Z0-9 ]', '', txt).split())
        if num_words == 0:
            return 0  
        else :
            len_avg_word=round(total_chars/float(num_words),2)
            return len_avg_word

    def get_num_words(txt): #pour compter nb de caracteres, nombre de mots et la longueur moyenne d'un mot 😄
        num_words = len(re.sub(r'[^a-zA-Z0-9 ]', '', txt).split())
        return num_words 
    
    
    # Modification des features dans la BDD mongo
    for doc in collec.find():
        txt = doc['text']
        l = get_len(txt)
        email = get_email(txt)
        url= get_url(txt)
        emoticone=get_emoticone(txt)
        specharacter=get_specharacter(txt)
        maj= get_majuscule(txt)
        moneysymbol= get_moneysymbol(txt)
        phone= get_phonenumber(txt)
        date=get_date(txt)
        claim= get_wordclaim(txt)
        free= get_wordfree(txt)
        call= get_wordcall(txt)
        prize= get_wordprize(txt)
        mobile= get_wordmobile(txt)
        won= get_wordwon(txt)
        win= get_wordwin(txt)
        cash= get_wordcash(txt)
        love= get_wordlove(txt)
        contact= get_wordcontact(text)
        avg_lenword= get_avg_lenword(txt)
        num_words = get_num_words(txt)

        id = doc['_id']

        #print(id, l, email,url)

        #collec.update_one({'_id': id}, {"$set": {"len": l}})
        #collec.update_one({'_id': id}, {"$set": {"n_email": len(email)}})
        #collec.update_one({'_id': id}, {"$set": {"n_url": len(url)}})
        #collec.update_one({'_id': id}, {"$set": {"n_emoticone": len(emoticone)}})
        #collec.update_one({'_id': id}, {"$set": {"n_sp_character": len(specharacter)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_majuscules": len(maj)}})
        #collec.update_one({'_id': id}, {"$set": {"n_money_symbol": len(moneysymbol)}})
        #collec.update_one({'_id': id}, {"$set": {"n_phone": len(phone)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_date": len(date)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_claim": len(claim)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_free": len(free)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_call": len(call)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_prize": len(prize)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_won": len(won)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_win": len(win)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_cash": len(cash)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_love": len(love)}})
        #collec.update_one({'_id': id}, {"$set": {"n_mots_contact": len(contact)}})
        #collec.update_one({'_id': id}, {"$set": {"avg_len_words": avg_lenword}})
        #collec.update_one({'_id': id}, {"$set": {"nombre_mots": num_words}})

    # Récuperation dans un dataframe pandas
    global df
    df = pd.DataFrame(list(collec.find()))
    
    
    # Suppressiones lignes répétées
    print()
    print('Lignes et colonnes du df original', df.shape)

    # global permet à une variable interne d'une fonction 
    # d'être utilisée en dehors de la fonction.

    global data
    data =df.drop_duplicates(subset='text', keep="last")
    print('Lignes et colonnes du df sans doublons (appelé data)', data.shape)
    
    # Suppression des colonnes non utiles
    global data_clean
    data_clean = data.drop(columns=['_id','text', 'n_mots_contact'], axis=1)
    
regex_text_to_df_avec_MongoDB('SMSSpamCollection', 'Projet_6_Spam_Celia_Cesar', 'sms')



# ML function
def entrainement_evaluation_du_modele(data, modele_donne, parameters, metrics_GS):
        
    # target preprocessing
    lb_encod = LabelEncoder()
    global y

    y = lb_encod.fit_transform(data['type'])
        
    
    # features preprocessing
    global X

    X = data.drop(columns='type')
    X.head()    

    # Division en groupes de training et d'évaluation
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    

    # Declare model for Grid Search
    model_GS = modele_donne

    # Declare the pipeline
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model_GS)]
        )

    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
    
    # Declare the Grid Search method
    grid = GridSearchCV(estimator = pipe, param_grid = parameters, scoring = metrics,
                        refit = metrics_GS, cv = 5, n_jobs =-1, verbose = 1)

    # Fit the model
    grid.fit(X_train, y_train)

    # Evaluate cross validation performance 
    print()
    print("model: ", modele_donne)
    print("CV - Best score:", round(grid.best_score_,3))
    print("CV - best parameters:", grid.best_params_)
    #print("CV - best estimator :", grid.best_params_)
    
    # cv_results_['mean_fit_time'] donne un array avec le résultat de chaque split, 
    # cette fonction fait une moyenne de toutes ces valeurs.
    def moyennes(keys_cv):        
        a1 = grid.cv_results_[keys_cv]
        Avg_key = sum(a1) / float(len(a1))
        #print(Avg_key)
        return Avg_key
    
    # Make predictions
    y_pred = grid.predict(X_test)
    
    # Evaluate model performance
    print()    
    print("++ CV - mean fit time:", round(moyennes('mean_fit_time'),2), 'seg', '++')
    global time
    time = round(moyennes('mean_fit_time'),2)
    print()
    #print("CV - mean_test_accuracy:", round(moyennes('mean_test_accuracy'),3))
    print("Test Accuracy:", round(accuracy_score(y_test, y_pred),3))
    global accuracy_final
    accuracy_final = round(accuracy_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_precision:", round(moyennes('mean_test_precision'),3))
    print("Test precision:", round(precision_score(y_test, y_pred),3))
    global precision_final
    precision_final = round(precision_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_recall:", round(moyennes('mean_test_recall'),3))
    print("Test recall:", round(recall_score(y_test, y_pred),3))
    global recall_final
    recall_final = round(recall_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_f1:", round(moyennes('mean_test_f1'),3))
    print("Test f1:", round(f1_score(y_test, y_pred),3))
    global f1_final
    f1_final = round(f1_score(y_test, y_pred),3)
    
    print()
    #print("CV - mean_test_roc_auc:", round(moyennes('mean_test_roc_auc'),3))
    print("Test roc_auc:", round(roc_auc_score(y_test, y_pred),3))
    global roc_auc_final
    roc_auc_final = round(roc_auc_score(y_test, y_pred),3)
        
    print()
    print("classification_report:")
    print()
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    global cm
    cm=confusion_matrix(y_test,y_pred)
    x= ['0', '1']
    y= ['1', '0']
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in cm]
    global fig_cm
    
    colorscale=[[0.0, 'rgb(97, 132, 232 )'], [.2, 'rgb(97, 232, 228 )'],
            [.4, 'rgb(142, 232, 97 )'], [.6, 'rgb(208, 232, 97 )'],
            [.8, 'rgb(232, 179, 97 )'],[1.0, 'rgb(232, 121, 97)']]
    fig_cm = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale=colorscale)
    layout = {
        "title": "Confusion Matrix", 
        "xaxis": {"title": "Predicted value"}, 
        "yaxis": {"title": "Real value"}
    }
        # add title
    fig_cm.update_layout(title_text='<b>Confusion matrix</b>',xaxis = dict(title='Real value'),
                  yaxis = dict(title='Predicted value')
                     )
    
    
    # adjust margins to make room for yaxis title
    fig_cm.update_layout(margin=dict(t=50, l=50))
    
    try:        
        # Make predictions and  Courbe de ROC
        y_pred = grid.predict(X_test)
        global y_pred_proba
        y_pred_proba =grid.predict_proba(X_test)[:, 1]
        global fpr
        global tpr
        global thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        global fig_roc
        fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'<b>ROC Curve (AUC={auc(fpr, tpr):.4f})</b>', 
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500)
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_roc.update_xaxes(constrain='domain')    

    except: 
        print("Cet estimateur n'a pas la propriété predict_proba pour pouvoir calculer la courbe ROC.")    
    
    
    try:        
        FI = grid.best_estimator_[1].feature_importances_
        
        d_feature = {'Stats':X.columns,
             'FI':FI}
        df_feature = pd.DataFrame(d_feature)

        df_feature = df_feature.sort_values(by='FI', ascending=0)
        print(df_feature)

        fig = px.bar_polar(df_feature, r="FI", theta="Stats",
                           color="Stats", template="plotly_dark",
                           color_discrete_sequence= px.colors.sequential.Plasma_r)
        #fig.show()       
     
    except:
        print()
        print('**********************************************************')
        print("Cet estimateur n'a pas la propriété de feature importances")
        print('**********************************************************')
    
    global df_test_1
    df_test_1 = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'time'])
    df_test_1[modele_donne] = (accuracy_final, precision_final, recall_final, f1_final, roc_auc_final, time)
    
    return accuracy_final, precision_final, recall_final, f1_final, roc_auc_final, time, df_test_1
    
    
metrics_GS = 'roc_auc'

# #Essaie KNN
# modele_donne = KNeighborsClassifier()
# parameters = {'model__n_neighbors':[1,3,5,12,15], 'model__weights': ('uniform','distance')}

# entrainement_evaluation_du_modele(data, modele_donne, parameters, metrics_GS)


# Essaie EXTRA TREES
# modele_donne = ExtraTreesClassifier()
# parameters= {'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'),
#  'model__min_samples_split':[2, 3, 4]}

# Essaie GRADIENTBOOST
# modele_donne = GradientBoostingClassifier()
# parameters = {'model__loss':('deviance', 'exponential'), 'model__learning_rate': [0.1, 0.2],
#              'model__n_estimators':[50, 100, 150]}



# -------------------------------- Préparation des élements du dashboard : variables utilisées dans le dashboard 

header_ =                     html.Br() 

nb_url= data_clean.groupby('type').agg({'n_url':'sum'}).reset_index()   

nb_email = data_clean.groupby('type').agg({'n_email':'sum'}).reset_index()
nb_tel= data_clean.groupby('type').agg({'n_phone':'sum'}).reset_index()

nb_tel[nb_tel['type']== "spam"  ].iat[0,1]  

word_claim = data_clean.groupby('type').agg({'n_mots_claim':'sum'})
word_free = data_clean.groupby('type').agg({'n_mots_free':'sum'})
word_won = data_clean.groupby('type').agg({'n_mots_won':'sum'})
word_win= data_clean.groupby('type').agg({'n_mots_win':'sum'})
word_love = data_clean.groupby('type').agg({'n_mots_love':'sum'})
word_cash = data_clean.groupby('type').agg({'n_mots_cash':'sum'})
word_call = data_clean.groupby('type').agg({'n_mots_call':'sum'})
word_prize = data_clean.groupby('type').agg({'n_mots_prize':'sum'})



# A faire pour spam et ham
nb_web_features_spam= [ dbc.Card(
        [
            html.H2(nb_url[nb_url['type']== "spam"  ].iat[0,1]  , className="card-title"),
            html.P("Url", className="card-text"),
        ],
        body=True,
        color="danger",
        inverse=True,  style={'height': '10vh', 'width': '30vh'}
    ),
    
    dbc.Card(
            [
                html.H2(nb_email[nb_email['type']== "spam"  ].iat[0,1]  , className="card-title"),
                html.P("E-mails", className="card-text"),
            ],
            body=True,
            color="danger",
            inverse=True,  style={'height': '10vh', 'width': '30vh'}
        ),
    
    dbc.Card(
            [
                html.H2(nb_tel[nb_tel['type']== "spam"  ].iat[0,1]  , className="card-title"),
                html.P("Phone numbers", className="card-text"),
            ],
            body=True,
            color="danger",
            inverse=True,  style={'height': '10vh', 'width': '30vh'}
        )
    ]

nb_web_features_ham= [ dbc.Card(
        [
            html.H2(nb_url[nb_url['type']== "ham"  ].iat[0,1]  , className="card-title"),
            html.P("Url", className="card-text"),
        ],
        body=True,
        color="success",
        inverse=True,  style={'height': '10vh', 'width': '30vh'}
    ),
    
    dbc.Card(
            [
                html.H2(nb_email[nb_email['type']== "ham"  ].iat[0,1]  , className="card-title"),
                html.P("E-mails", className="card-text"),
            ],
            body=True,
            color="success",
            inverse=True,  style={'height': '10vh', 'width': '30vh'}
        ),
    
    dbc.Card(
            [
                html.H2(nb_tel[nb_tel['type']== "ham"  ].iat[0,1]  , className="card-title"),
                html.P("Phone numbers", className="card-text"),
            ],
            body=True,
            color="success",
            inverse=True,  style={'height': '10vh', 'width': '30vh'}, 
        )
    ]



data_kpi = pd.DataFrame(data)

kpi_nb_data =[ dbc.Card(
        [
            html.H2( len(data_kpi['type']), className="card-title"),
            html.P("Nombre de messages", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,  style={'height': '13vh', 'width': '50vh','textAlign': 'center'},className="h-100"
    ),
    
    dbc.Card(
            [
                html.H2(len(data_clean.columns)-1, className="card-title"),
                html.P("Nombre de features", className="card-text"),
            ],
            body=True,
            color="primary",
            inverse=True, style={'height': '13vh', 'width': '50vh', 'textAlign': 'center'}, className="h-100"
        ) 
    ]



# Graph analyse textuelle
fig_len = px.histogram(df, x="len",color='type',color_discrete_map = {'ham':'aquamarine',"spam":'darkorange'})
fig_len.update_layout(showlegend=False)
fig_len.update_layout({
'plot_bgcolor':'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig_len.update_xaxes(title_text='Character counts')
fig_len.update_yaxes(title_text='Total')

fig_word = px.histogram(df, x="nombre_mots", color="type",color_discrete_map = {'ham':'aquamarine',"spam":'darkorange'})
fig_word.update_layout(showlegend=False)
fig_word.update_layout({
'plot_bgcolor':'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig_word.update_xaxes(title_text='Word counts')
fig_word.update_yaxes(title_text='Total')

avg_len_word = px.histogram(df, x="avg_len_words", color="type",color_discrete_map = {'ham':'aquamarine',"spam":'darkorange'} )
avg_len_word.update_layout({
'plot_bgcolor':'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
avg_len_word.update_xaxes(title_text='Average of word counts')
avg_len_word.update_yaxes(title_text='Total')



#  --Pour l'ajout de l'image du wordcloud Matplotlib, (images en local)
image_filenamedown = 'rotatethumb_final.png'
image_filenameup = 'upfinal.png'
image_filehamfq = 'ham_fq.png'
image_filespamfq = 'spam_fq.png'
img_plolty = "plotly.png" 
img_accueil= "nospam.png" 

# Fonction pour lire une image enregistrée en local
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')




fig_hist_spamham = px.pie(data_clean, names=['ham','spam'], values=data_clean.type.value_counts() , color=['ham','spam'], color_discrete_map={'ham':'aquamarine',"spam":'darkorange'})




#### PRE- STRUCTURATION : STYLING DASHBOARD 


# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 10,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "5rem 1rem",
    "background-color": "rgba(0, 0, 0, 0)",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}

CONTENT_STYLE_LESSMARGE = {
    "margin-left": "8rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}


sidebar = html.Div(
    [
        html.Img(src=b64_image(img_accueil),style={'width':'80%' ,"margin-left": "1rem"} ),
        html.Hr(),
        html.P(
            "Spam or Ham ?", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Data", href="/", active="exact"),
                dbc.NavLink("Text message analysis", href="/page-1", active="exact"),
                dbc.NavLink("Word frequency analysis", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

sidebar1 = html.Div(
    [
        html.Img(src=b64_image(img_accueil),style={'width':'80%' ,"margin-left": "1rem"} ),
        html.Hr(),
        html.P(
            "Spam or Ham ?", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink(" K-nearest neighbor classifier results", href="/", active="exact"),
                dbc.NavLink(" Extra-trees classifier results", href="/page-1", active="exact"),
                dbc.NavLink("Gradient Boosting classifier results", href="/page-2", active="exact"),
                dbc.NavLink("Model comparisons", href="/page-3", active="exact"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)
content1 = html.Div(id="my-output", children=[], style=CONTENT_STYLE)

# Appel de la fonction dashboard Plotly

app = dash.Dash(__name__,
               external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions = True)
                  

# -------------------------------------- Ouverture du dashboard

####  Affichage : Structure du dashbord et composants
app.layout =  html.Div([
  
    ###  1er ONGLET ###

    dcc.Tabs([
        
        dcc.Tab( label='Exploratory Data Analysis', children=[

     # activated once/week or when page refreshed
     dcc.Interval(id='interval_db', interval=86400000 * 7, n_intervals=0),
        html.Br(),   
        html.Br(),   

   html.Div([
      dcc.Location(id="url"),
      sidebar , content,
     
      
       ])
      ])
        ,
       
         ### FIN DU 1er ONGLET 
        
         #-----------------------------------------------------------
        
         ### DEBUT 2EME ONGLET
       dcc.Tab( label='Machine learning ', children=[
         
            html.Div([
               dcc.Location( id="my-input"),
               sidebar1 ,
               
               html.Div(
                   children=[
                       dbc.Row(dbc.Col(
                           dbc.Spinner(children=[

               content1, 
             
              

  ], size="sm", color="success", type="grow", fullscreen=True,
                               spinnerClassName="spinner", spinner_style={"width": "6rem", "height": "6rem"}) ))])
])
])
])
    ])

    ### FIN DU 2eme ONGLET 
   

# --------------------------------- Fonctions callback pour exploiter la base de données mongo db et à utiliser dans le dashboard 

# Display Datatable with data from Mongo database *************************
@app.callback(Output('mongo-datatable', 'children'),
              [Input('interval_db', 'n_intervals')])

def spamham_datatable(n_intervals):
    print(n_intervals)
    # Convert the Collection (table) date to a pandas DataFrame
    data
    #Drop the _id column generated automatically by Mongo
    global df_
    df_ = data.iloc[:, 1:]
    print(df_.head(20))

    return [
        dash_table.DataTable(
            id='my-table',
            columns=[{
                'name': x,
                'id': x,
            } for x in df_.columns],
            data=df_.to_dict('records'),
            editable=True,
            row_deletable=False,
            filter_action="native",
            style_as_list_view=True,
            filter_options={"case": "sensitive"},
            sort_action="native",  # give user capability to sort columns
            sort_mode="single",  # sort across 'multi' or 'single' columns
            page_current=0,  # page number that user is on
            page_size=3,  # number of rows visible per page
            css=[{'selector': '.row', 'rule': 'margin: 0'}],
            style_cell={'textAlign': 'center', 'minWidth': '100px','fontSize': 17,
            'padding': '10px','width': '100px', 'maxWidth': '300px'},
            style_table={
        'overflowX': 'scroll',
        'width': '80%',"borderRadius": "50px",'display':'inline-block', 'margin':50
    },  
            style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
        'overflowX': 'auto', 'textAlign': 'center',
            'fontWeight': 'bold',
            'fontSize': 15},
        style_header={
            'fontSize': 15,
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell_conditional=[
        {'if': {'column_id': 'text'},
         'width': '90%'}]
        
        )
    ]

#---------------------------- SIDEBAR ONGLET 1

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
html.Div(
    children=[
        dbc.Row(dbc.Col(
            dbc.Spinner(children=[

                    # html.H1("Tableau de données", style={'textAlign': 'center'}),
                html.Div(
                    
                    id='mongo-datatable', children=[  ],
                                              style=CONTENT_STYLE),   
        
                html.H2("Ham and Spam Proportion",  style=CONTENT_STYLE),
                dcc.Graph(figure=fig_hist_spamham, style=CONTENT_STYLE),],size="sm", color="success", type="grow", fullscreen=True,spinnerClassName="spinner", spinner_style={"width": "6rem", "height": "6rem"}) ))])
                ]
    elif pathname == "/page-1":
        return [

           
                
                #html.H1("Analyse de données textuelles",  style=CONTENT_STYLE),
                
    html.Div(className='row', children=[
                 dbc.Row([
                     dbc.Col(
                    dcc.Graph(figure=fig_len), style={'display': 'inline-block', 'width':'40%'}),
                 
                     dbc.Col(
                    dcc.Graph(figure=fig_word), style={'display': 'inline-block','width':'40%'}),
                    
                    dbc.Col(
                    dcc.Graph(figure=avg_len_word), style={'display': 'inline-block','width':'40%'}),
                      ]) ],  style=CONTENT_STYLE_LESSMARGE),
                 
                   html.Div(className='row', children=[
                      dbc.Row(
                          [ dbc.Col(card) for card in nb_web_features_spam] )
                      ],  style=CONTENT_STYLE_LESSMARGE), 
                   
                    html.Br(),   

                
                   html.Div(className='row', children=[
                      dbc.Row(
                          [ dbc.Col(card) for card in nb_web_features_ham] )
                      ], style=CONTENT_STYLE_LESSMARGE) 
                ]
    elif pathname == "/page-2":
        return [
            #html.H1("Analyse fréquentielle des mots",  style=CONTENT_STYLE_LESSMARGE),
        
        html.Div(
            children=[
                dbc.Row(dbc.Col(
                    dbc.Spinner(children=[
        
            html.Div(className='row', children=[
             
                dbc.Row(
                    [
            
                dbc.Col(
               
                            html.Img(src=b64_image(image_filenameup),style={'width':'60%','display':'inline-block'
                           } ),  style={'textAlign': 'center'}),
          
                dbc.Col(             
               
                            html.Img(src=b64_image(image_filenamedown),style={ 'width':'60%','display':'inline-block'
                           }),  style={'textAlign': 'center'} )
                ]),

            dbc.Row(
                [
        
            dbc.Col(
           
                        html.Img(src=b64_image(image_filehamfq),style={'width':'100%','display':'inline-block'
                       } ), style={'textAlign': 'center'}),
      
            dbc.Col(             
           
                        html.Img(src=b64_image(image_filespamfq),style={ 'width':'100%','display':'inline-block'
                       }), style={'textAlign': 'center'} )
            ]),
         
            ],  style=CONTENT_STYLE_LESSMARGE) ,] ,size="sm", color="success", type="grow", fullscreen=True,spinnerClassName="spinner", spinner_style={"width": "6rem", "height": "6rem"}) ))])
               
                ]
 

# -------------------------------------------------------- SIDEBAR ONGLET 2

@app.callback(Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='pathname')
)
def render_page_content1(pathname):
    if pathname == "/":
        # #Essai KNN
        # modele_donne = KNeighborsClassifier()
        # parameters = {'model__n_neighbors':[1,3,5,12,15], 'model__weights': ('uniform','distance')}

        entrainement_evaluation_du_modele(data_clean, KNeighborsClassifier(),{'model__n_neighbors':[1,3,5,12,15], 'model__weights': ('uniform','distance')}, 'roc_auc')
        
        
        kpi_ml =   [ dbc.Card(
                [
                    html.H2(precision_final, className="card-title"),
                    html.P("Precision ", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},  className="h-100"
            ),
            
            dbc.Card(
                    [
                        html.H2(accuracy_final, className="card-title"),
                        html.P("Accuracy", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True,style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(recall_final, className="card-title"),
                        html.P("Recall", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(f1_final, className="card-title"),
                        html.P("F1 Score", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(roc_auc_final, className="card-title"),
                        html.P("ROC AUC", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            
        dbc.Card(
                [
                    html.H2(time , className="card-title"),
                    html.P("Temps d'exécution moyen CV (s)", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            )
            
            ]       


        return [
                    
           html.Div(className='row', children=[

              dbc.Row([dbc.Col(card) for card in kpi_ml])
              ], style=CONTENT_STYLE
             ),
           
    html.Div(className='row', children=[
                 
   
           dbc.Row([
               dbc.Col( 
                   dcc.Graph(
                       figure = fig_roc),  style={'display': 'inline-block', 'width':'50%'}),  
           
            
               dbc.Col( 
                   dcc.Graph(
                       figure = fig_cm),  style={'display': 'inline-block',"padding": "2rem 4rem"}) ]) ])
                ]
    
    elif pathname == "/page-1":
        # Essai EXTRA TREES
        # modele_donne = ExtraTreesClassifier()
        # parameters= {'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'),
        #  'model__min_samples_split':[2, 3, 4]}
        
        entrainement_evaluation_du_modele(data_clean, ExtraTreesClassifier(),{'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'), 'model__min_samples_split':[2, 3, 4]}
        , 'roc_auc')
        
        kpi_ml =   [ dbc.Card(
                [
                    html.H2(precision_final, className="card-title"),
                    html.P("Precision ", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},  className="h-100"
            ),
            
            dbc.Card(
                    [
                        html.H2(accuracy_final, className="card-title"),
                        html.P("Accuracy", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True,style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(recall_final, className="card-title"),
                        html.P("Recall", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(f1_final, className="card-title"),
                        html.P("F1 Score", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(roc_auc_final, className="card-title"),
                        html.P("ROC AUC", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            
        dbc.Card(
                [
                    html.H2(time, className="card-title"),
                    html.P("Temps d'exécution moyen CV (s)", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            )
            
            ]       
        
        #Graph features importance
        xtc = ensemble.ExtraTreesClassifier()
        xtc.fit(X_train, y_train)
        d = {'Stats':X.columns,
              'FI': xtc.feature_importances_}
        df_ftimp = pd.DataFrame(d)

        df_ft = df_ftimp.sort_values(by='FI', ascending=0)
        fig_ft = px.bar_polar(df_ft, r="FI", theta="Stats",
                            color="Stats", template="plotly_white",
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
        fig_ft.layout.height = 800
        fig_ft.layout.width = 800
        fig_ft.update_layout( showlegend=False,font=dict(
        size=12), title={
        'text': "<b>Feature Importances</b>",
        'xanchor': 'left',
        'yanchor': 'top'})
        
        return  [



           # PARTIE EXTRATREES

            html.Div(className='row', children=[
               dbc.Row([dbc.Col(card) for card in kpi_ml])
               ], style=CONTENT_STYLE
              ),
            html.Div(className='row', children=[
   
            dbc.Row([
                
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_roc
                ), style={'display': 'inline-block', 'width':'50%'}),
            
              
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_cm
                ), style={'display': 'inline-block',"padding": "2rem 4rem"}) ]) ]),
            
            
       dbc.Row([
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_ft
                ),style=CONTENT_STYLE) 
         ],  style= CONTENT_STYLE_LESSMARGE)     
       ]    
        
                
               
                
    elif pathname == "/page-2":
        
        # Essai GRADIENTBOOST
        # modele_donne = GradientBoostingClassifier()
        # parameters = {'model__loss':('deviance', 'exponential'), 'model__learning_rate': [0.1, 0.2],
        #              'model__n_estimators':[50, 100, 150]}
        
        entrainement_evaluation_du_modele(data_clean,  GradientBoostingClassifier(),{'model__loss':('deviance', 'exponential'), 'model__learning_rate': [0.1, 0.2]}, 'roc_auc')
        kpi_ml =   [ dbc.Card(
                [
                    html.H2(precision_final, className="card-title"),
                    html.P("Precision ", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},  className="h-100"
            ),
            
            dbc.Card(
                    [
                        html.H2(accuracy_final, className="card-title"),
                        html.P("Accuracy", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True,style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(recall_final, className="card-title"),
                        html.P("Recall", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(f1_final, className="card-title"),
                        html.P("F1 Score", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            dbc.Card(
                    [
                        html.H2(roc_auc_final, className="card-title"),
                        html.P("ROC AUC", className="card-text"),
                    ],
                    body=True,
                    color="primary",
                    inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
                ),
            
        dbc.Card(
                [
                    html.H2(time, className="card-title"),
                    html.P("Temps d'exécution moyen CV (s)", className="card-text"),
                ],
                body=True,
                color="primary",
                inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},   className="h-100"
            )
            
            ]       
            #Graph features importance
        gbc = ensemble.GradientBoostingClassifier()
        gbc.fit(X_train, y_train)
        d = {'Stats':X.columns,
              'FI':gbc.feature_importances_}
        df_ftimp = pd.DataFrame(d)
        
        df_ft = df_ftimp.sort_values(by='FI', ascending=0)
        fig_ft = px.bar_polar(df_ft, r="FI", theta="Stats",
                            color="Stats", template="plotly_white",
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
        fig_ft.layout.height = 800
        fig_ft.layout.width = 800
        fig_ft.update_layout( showlegend=False,font=dict(
        size=12), title={
        'text': "<b>Feature Importances</b>",
        'xanchor': 'left',
        'yanchor': 'top'})

        
        return [
            
            #PARTIE GRADIENT BOOST



            html.Div(className='row', children=[
               dbc.Row([dbc.Col(card) for card in kpi_ml])
               ], style=CONTENT_STYLE
              ),
            html.Div(className='row', children=[
   
                
            dbc.Row([
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_roc
                ), style={'display': 'inline-block', 'width':'50%'}),
                
               
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_cm
                ), style={'display': 'inline-block',"padding": "2rem 4rem"}),]),]), 
            
            
      dbc.Row([
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_ft
                ),style=CONTENT_STYLE) 
              ], style= CONTENT_STYLE_LESSMARGE)
           ]
                
          
               
                
    elif pathname == "/page-3":
 
        accuracy_SVC, precision_SVC, recall_SVC, f1_SVC, roc_auc_SVC, time_SVC, df = entrainement_evaluation_du_modele(data_clean, svm.SVC(),{'model__kernel':('linear', 'rbf'), 'model__C':[1, 10]}, 'roc_auc')
        accuracy_knn, precision_knn, recall_knn, f1_knn, roc_auc_knn, time_knn, df2 = entrainement_evaluation_du_modele(data_clean, KNeighborsClassifier(),{'model__n_neighbors':[1,3,5,12,15], 'model__weights': ('uniform','distance')}, 'roc_auc')
        accuracy_GBC, precision_GBC, recal_GBCl, f1_GBC, roc_auc_GBC, time_GBC, df3 = entrainement_evaluation_du_modele(data_clean,  GradientBoostingClassifier(),{'model__loss':('deviance', 'exponential'), 'model__learning_rate': [0.1, 0.2]}, 'roc_auc')
        accuracy_ETC, precision_ETC, recall_ETC, f1_ETC, roc_auc_ETC, time_ETC, df4 = entrainement_evaluation_du_modele(data_clean, ExtraTreesClassifier(),{'model__n_estimators':[50, 100, 150], 'model__criterion': ('gini', 'entropy'), 'model__min_samples_split':[2, 3, 4]}, 'roc_auc')
        accuracy_LSVC, precision_LSVC, recall_LSVC, f1_LSVC, roc_auc_LSVC, time_LSVC, df5 = entrainement_evaluation_du_modele(data_clean,  LinearSVC(), {'model__C':[0.001,0.01,0.1], 'model__dual': [False,True],
             'model__multi_class':['ovr', 'crammer_singer']}, 'roc_auc')
        accuracy_ABC, precision_ABC, recall_ABC, f1_ABC, roc_auc_ABC, time_ABC, df6 = entrainement_evaluation_du_modele(data_clean, AdaBoostClassifier(), {'model__n_estimators':[25, 50, 100], 'model__learning_rate': [0.5, 1.0, 2.0],
             'model__algorithm':['SAMME', 'SAMME.R']}, 'roc_auc') 
        accuracy_RF, precision_RF, recall_RF, f1_RF, roc_auc_RF, time_RF, df7 = entrainement_evaluation_du_modele(data_clean, RandomForestClassifier(), {'model__n_estimators':[50, 100, 150], 'model__criterion': ['gini', 'entropy'],
             'model__min_samples_split':[2, 5, 10]}, 'roc_auc')
        
        df_tot_1 = pd.concat([df, df2, df3, df4, df5, df6, df7], axis=1)
        df_tot_1.columns = ['SVC', 'KNeighbors', 'GradientBoosting', 'ExtraTrees', 'LinearSVC', 'AdaBoost', 'RandomForest']
        df_tot = df_tot_1.transpose()
        df_tot.sort_values(by = 'roc_auc', ascending= False, inplace= True)
        df_tot.sort_values(by = 'accuracy', ascending= False, inplace= True)
        df_tot_heatmap = df_tot.drop(columns='time')
        fig_heatmap = px.imshow(df_tot_heatmap)
        fig_heatmap.update_layout(xaxis={'title': 'Metrics'}, yaxis={'title': 'Models'})
        fig_heatmap.layout.height = 800
        fig_heatmap.layout.width = 1000
        
        return [
            
            
            

            # PARTIE HEATMAP
            html.Div(className='row', children=[
                dbc.Col( 
                    dcc.Graph(
                        figure = fig_heatmap
                )) ], style={
                    "margin-left": "9rem",
                    "margin-right": "1rem",
                    "padding": "2rem 1rem",
                }),  ]

                



# Run le dashboard sur le server http://127.0.0.1:8050/
if __name__ == '__main__':
    app.run_server(debug=True)