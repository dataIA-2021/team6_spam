#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:34:24 2022

@author: celia
"""

import json
import requests
from pymongo import MongoClient

#co à mongoDB
client = MongoClient('mongodb://localhost:27017/')

db = client['dataia2021'] # use a database called "dataia2021"
collec = db['spam_collection']   # and inside that DB, a collection called "spam_collection"


# the file to be converted to 
# json format 
filename = '/home/celia/Documents/Projet_spam/SMSSpamCollection'

file_content = []

# creating list 
with open(filename) as fh: 
    for line in fh: 
		# reads each line and trims of extra the spaces 
		# and gives only the valid words 
        target, message = line.strip().split("\t", maxsplit=1)
        file_content.append({"target": target, "message": message})

# creating json file 
# the JSON file is named as test1 
out_file = open("/home/celia/Documents/Projet_spam/spam.json", "w") 
json.dump(file_content, out_file)

#Import données dans collection
x = collec.insert_many(file_content)



