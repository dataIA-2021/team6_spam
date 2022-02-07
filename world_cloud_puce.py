#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:49:20 2022

@author: celia
"""



import json
import requests
from pymongo import MongoClient
import pandas as pd
import re

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

check=data.type.value_counts()



# # # Fréquence des mots dans un spam et dans un ham # # #

#-----------------------

# Ham type
hamtype= data[data['type'] == 'ham']
word_freq_hamtype=hamtype.text.str.split(expand=True).stack().value_counts().reset_index()
word_freq_hamtype=word_freq_hamtype.rename(columns={0:"count"})
stopw =["I","a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]


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

# Import image to np.array, le mask pouce
mask = np.array(Image.open('/home/celia/Documents/Projet_spam/thumbup.png'))
# Generate wordcloud
wordcloudham = WordCloud(width = 3000, height = 2000, background_color='white', colormap='Set2', collocations=False,stopwords=stopw ,mask=mask).generate(' '.join(word_freq_hamtype['index']))
# Plot
plot_cloud(wordcloudham)


#-----------------------


# Spam type
spamtype= data[data['type']== 'spam']
word_freq_spamtype=spamtype.text.str.split(expand=True).stack().value_counts().reset_index()
word_freq_spamtype=word_freq_spamtype.rename(columns={0:"count"})
word_freq_spamtype=word_freq_spamtype[word_freq_spamtype['index'].isin(stopw) == False]


# Bar plot frq des mots spam
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='index', y='count', 
            data=word_freq_spamtype.head(60) ,ax=ax)
plt.title("Fréquence des mots dans les spams")
plt.xticks(rotation='vertical')

# Import image to np.array, le mask pouce
mask_1 = np.array(Image.open('/home/celia/Documents/Projet_spam/rotatethumb.png'))
# Generate wordcloud
wordcloudspam = WordCloud(width = 3000, height = 2000, background_color='white', colormap='Set2', stopwords=stopw,collocations=False,  mask=mask_1).generate(' '.join(word_freq_spamtype['index']))
# Plot
plot_cloud(wordcloudspam)
