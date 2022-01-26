#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:00:18 2022

@author: celia
"""

import json
import requests
from pymongo import MongoClient
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Import packages
import numpy as np
from PIL import Image
from wordcloud import WordCloud

#Connexion à la base de données et collection Mongo
myclient = MongoClient("mongodb://localhost:27017/")
mydb = myclient["Projet_6_Spam_Celia_Cesar"]
mycol = mydb["sms"]

#Requête pour touver tous les documents
mydoc = mycol.find()

# Insertion des documents dans une liste
for x in mydoc:
  list_cur = list(mydoc)
  print(list_cur)
 
# Conversion des documents en df
df = pd.DataFrame(list_cur)

# Trouver les doublons
duplicateDFRow = df[df.duplicated(['text'])]
print(duplicateDFRow)


# Supprimer les doublons 
data =df.drop_duplicates(subset='text', keep="last")
data.describe()
data.groupby("type").describe()


# Caractères speciaux dans un spam et un ham
myquery_spechar_exc= { "text": {"$regex": "[!]{2}" } }
mydoc_spechar_exc = mycol.find(myquery_spechar_exc)
list_spechar_exc= list(mydoc_spechar_exc)

myquery_spechar_excint= { "text": {"$regex": "[?!]" } }
mydoc_spechar_excint = mycol.find(myquery_spechar_excint)
list_spechar_excint= list(mydoc_spechar_excint)

myquery_spechar_dot= { "text": {"$regex": "[.]{3}" } }
mydoc_spechar_dot = mycol.find(myquery_spechar_dot)
list_spechar_dot= list(mydoc_spechar_dot)


# Majuscules dans un spam et un ham
myquery_spechar_maj= { "text": {"$regex": "[A-Z]{3}" } }
mydoc_spechar_maj= mycol.find(myquery_spechar_maj)
list_spechar_maj=list(mydoc_spechar_maj)


# Présence d'url dans un spam et ham 
myquery_spechar_url= { "text": {"$regex": "(https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*" } }
mydoc_spechar_url= mycol.find(myquery_spechar_url)
list_spechar_url=list(mydoc_spechar_url)


# Présence d'email dans un spam et ham 
myquery_spechar_email= { "text": {"$regex": "[a-zA-Z0-9_.+%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z0-9-.]+" } }
mydoc_spechar_email = mycol.find(myquery_spechar_email)
list_spechar_email= list(mydoc_spechar_email)


# Présence de symboles £|\$|\€"
myquery_moneysymbol = { "text": {"$regex": "£|\$|\€"} }
mydoc_moneysymbol = mycol.find(myquery_moneysymbol)
list_moneysymbol = list(mydoc_moneysymbol)
print(mydoc_moneysymbol)

# Présence de numéro telephone
myquery_tel = { "text": {"$regex": "[0-9]{10}"} }
mydoc_tel = mycol.find(myquery_tel)
list_tel  = list(mydoc_tel)

# Présence de dates
# dates 01.01.1999
myquery_date1 = { "text": {"$regex": "^\d+.\d+.\d+"} }
mydoc_date1 = mycol.find(myquery_date1)
list_date1  = list(mydoc_date1)

#dates 01/01/1999
myquery_date2 = { "text": {"$regex": "\d+/\d+/\d+"} }
mydoc_date2 = mycol.find(myquery_date2)
list_date2  = list(mydoc_date2)

#Smileys
myquery_smiley = { "text": {"$regex": "(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"} }
mydoc_smiley = mycol.find(myquery_smiley)
list_smiley  = list(mydoc_smiley)


#Regex trouver les mots clés : call, claim, free, mobile, prize
myquery_call = { "text": {"$regex": "(?i)(\W|^)(call)(\W|$)"} }
mydoc_call = mycol.find(myquery_call)
list_call  = list(mydoc_call)


# Fréquence des mots dans un spam et dans un ham

#data df
word_freq=data.text.str.split(expand=True).stack().value_counts().reset_index()
word_mode=data.type.str.split(expand=True).stack().value_counts().reset_index()

#ham type
hamtype= data[data['type'] == 'ham']
word_freq_hamtype=hamtype.text.str.split(expand=True).stack().value_counts().reset_index()
word_freq_hamtype=word_freq_hamtype.rename(columns={0:"count"})
stopw =["I","a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
#word_freq_hamtype['word'] = word_freq_hamtype['index'].apply(lambda x: ' '.join([word for word in x.split() if word not in (N)]))


word_freq_hamtype=word_freq_hamtype[word_freq_hamtype['index'].isin(stopw) == False]

#Graph bar frq des mots hams
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='index', y='count', 
            data=word_freq_hamtype.head(50) ,ax=ax)
plt.title("Fréquence des mots dans les hams")
plt.xticks(rotation='vertical')
plt.show()


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

# Import image to np.array
mask = np.array(Image.open('/home/celia/Documents/Projet_spam/thumbup.png'))
# Generate wordcloud
wordcloudham = WordCloud(width = 3000, height = 2000, background_color='white', colormap='Set2', collocations=False,stopwords=stopw ,mask=mask).generate(' '.join(word_freq_hamtype['index']))
# Plot
plot_cloud(wordcloudham)


#spam type
spamtype= data[data['type']== 'spam']
word_freq_spamtype=spamtype.text.str.split(expand=True).stack().value_counts().reset_index()
word_freq_spamtype=word_freq_spamtype.rename(columns={0:"count"})
word_freq_spamtype=word_freq_spamtype[word_freq_spamtype['index'].isin(stopw) == False]



#bar plot spam
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='index', y='count', 
            data=word_freq_spamtype.head(60) ,ax=ax)
plt.title("Fréquence des mots dans les spams")
plt.xticks(rotation='vertical')

# Import image to np.array
mask_1 = np.array(Image.open('/home/celia/Documents/Projet_spam/rotatethumb.png'))
# Generate wordcloud
wordcloudspam = WordCloud(width = 3000, height = 2000, background_color='white', colormap='Set2', stopwords=stopw,collocations=False,  mask=mask_1).generate(' '.join(word_freq_spamtype['index']))
# Plot
plot_cloud(wordcloudspam)

