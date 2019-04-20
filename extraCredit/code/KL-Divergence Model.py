#!/usr/bin/env python
# coding: utf-8

# In[48]:


import os
from bs4 import BeautifulSoup
from queue import Queue, Empty
from urllib.parse import urljoin, urlparse
from nltk.util import ngrams
import nltk
from string import punctuation
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
import numpy as np
import math
import operator
nltk.download('punkt')
import pandas as pd
import re
import collections
import string
k1=1.2
b=0.75
k2=100


# In[49]:


stops = os.getcwd().replace("extra\\code","general")+"\\test-collection\\common_words" #load stopwords
f = open(stops, 'r',encoding='utf-8')
stopwords = f.read().split("\n")


# In[50]:


Collection = [] # load documents and do parsing job
Doc_name = []
folder_d = os.getcwd().replace("extra\\code","general")+"\\test-collection\\cacm"
for filename in os.listdir(folder_d):
    Doc_name.append(filename.replace(".html",""))
    file_d = folder_d + "\\" + filename
    f = open(file_d, 'r',encoding='utf-8')
    remove = string.punctuation # remove punctuations
    pattern = r"[{}]".format(remove)
    content = f.read().replace("\n"," ").replace("\t"," ") 
    content = BeautifulSoup(content, "html5lib").get_text()
    content = re.sub(pattern, " ", content).lower() 
    content = content.replace(" cacm" + content.split("cacm")[-1],"")  
    #ignore the digits that commonly appear in the end of the documents’ content.
    Collection.append(re.sub(r'\s+', ' ', content).strip())


# In[51]:


file_query = os.getcwd().replace("extra\\code","general")+"\\test-collection\\cacm.query.txt" #load query list
f = open(file_query, 'r',encoding='utf-8')
remove = string.punctuation # remove punctuations
remove = remove.replace("-", "")
pattern = r"[{}]".format(remove)
content = f.read().replace("\n", "").replace("\t","").replace("<DOC>","SSSS")
content = BeautifulSoup(content, "html5lib").get_text()
content = re.sub(r'[0-9]+', '', content)
content = re.sub(pattern, "", content).lower() 
query = content.split("ssss")
query.remove("") #Query list


# In[52]:


def Create_Inverted_Index(n_grams, collection): #create inverted index
    gramDict = {}
    for i in range(len(Doc_name)):
        document = collection[i]
        DocID = i
        g = ngrams(document.split(),n_grams)
        d = nltk.FreqDist(g)
        for key in d:
            if(key[0] in gramDict):
                gramDict[key[0]].append([DocID,d[key]])
            else:
                gramDict[key[0]] = [[DocID,d[key]]]
    return gramDict


# In[53]:


unigram = Create_Inverted_Index(1,Collection)
s = 0 
for d in Collection:
    content = d.split(" ")
    s = s + len(content)
avgdl = s/len(Collection) # Average document length


# In[54]:


def BM25score(query, unigram, stopwords, collection): #BM25 model
    score_map = {}
    q = []
    remove = string.punctuation # remove punctuations
    pattern = r"[{}]".format(remove)
    query =  re.sub(pattern, "", query) 
    for word in query.split():
        if word not in stopwords:
            q.append(word)
    words =  collections.Counter(q)
    for word in words.keys():
        if word not in stopwords and word in unigram:
            index = unigram[word]
            idf = np.log((len(collection) - len(index) + 0.5)/(len(index) + 0.5)) # total document number = 1000
            for ind in index:
                doc_id = ind[0]
                f = ind[1]
                qf = words[word]
                dl = len(collection[doc_id].split(" "))
                K = k1*(1-b+b*dl/avgdl)
                R = (f*(k1+1)/(f+K)) * ((qf*(k2+1))/(qf+k2))
                if doc_id not in score_map:
                    score_map[doc_id] = R*idf
                if doc_id in score_map:
                    score_map[doc_id] = score_map[doc_id] + R*idf   
    sorted_list = sorted(score_map.items(), key=lambda kv: kv[1])
    sorted_list.reverse() 
    return sorted_list


# In[55]:


def dice_coef(bi_a,bi_b): # calculate dice coef with two bigram list
    nab = 0
    for bi in bi_a:
        if bi in bi_b:
            nab += 1
    na = len(bi_a)
    nb = len(bi_b)
    return 2*nab/(na+nb)


# In[56]:


def getletterbigram(string): #string to bigram
    res_list = []
    for i in range(len(string)-1):
        if string[i] != " " and string[i+1] != " ":
            res_list.append(string[i:i+2])
    return res_list  


# In[57]:


def prf_expansion(q, unigram, stopwords, collection, k, n):
    bigram_query = getletterbigram(q)
    result = BM25score(q, unigram, stopwords, collection)
    rel_doc_id = []
    for i in range(k):
        rel_doc_id.append(result[i][0])
    dice_score_map = {} #build word and dice's coef map
    for docid in rel_doc_id:
        doc = collection[docid]
        for word in doc.split(" "):
            if len(word) >= 2 and word not in stopwords:
                bi_word = getletterbigram(word)
                dice = dice_coef(bigram_query,bi_word)
                if word in dice_score_map:
                    dice_score_map[word] = dice_score_map[word] + 1
                else:
                    dice_score_map[word] = dice
    dice_list = sorted(dice_score_map.items(), key=lambda kv: kv[1])
    dice_list.reverse() #get top words
    for i in range(n):
        q = q + " " + dice_list[i][0]
    return q


# In[58]:


def pwq(word,q,bucket,C):
    pd = 1/len(C)
    res = 0
    for doc in C:
        temp = pd
        n_doc = len(doc.split(" "))
        wordmap = collections.Counter(doc.split(" "))
        if word in wordmap:
            pwd = wordmap[word]/n_doc
        else:
            pwd = 1/(avgdl * 10)
        for qword in q.split(" "):
            if qword not in stopwords:
                if qword in wordmap:
                    pq = wordmap[qword]/n_doc
                else:
                    pq = 1/(avgdl * 10)
                temp = temp * pq
        temp = temp * pwd
        res = res + temp
    return res


# In[59]:


def pf_model(q, unigram, stopwords, collection, k): #pseudo relevance feedback based on kl divergence
    result = BM25score(q, unigram, stopwords, collection)[:k] # select top k documents as relevant ones
    exq = prf_expansion(q, unigram, stopwords, collection, 10, int(len(q.split(" "))/3)) #query expansion
    bucket = []
    C = []
    i = 0
    for record in result:
        i = i + 1
        words = collection[record[0]].split(" ")
        C.append(collection[record[0]])
        for word in words:
            if word not in stopwords and word not in bucket:
                bucket.append(word)
    model = {}
    total = 0
    for word in bucket:
        model[word] = pwq(word,q,bucket,C)
    for word in model:
        total += model[word]
    for word in model:
        model[word] = model[word]/total
    return model


# In[60]:


def kl_rank(q, unigram, stopwords, collection, k):
    rel_model = pf_model(q, unigram, stopwords, collection, k)
    score_map = {}
    docid = 0
    for doc in collection:
        kl_score = 0
        wordmap = collections.Counter(doc.split(" "))
        l = len(doc.split(" "))
        for word in doc.split(" "):
            if word in rel_model:
                if word in wordmap:
                    kl_score = kl_score + rel_model[word] * np.log(wordmap[word]/l)
        score_map[docid] = kl_score
        docid = docid + 1
    rlist = sorted(score_map.items(), key=lambda kv: kv[1])
    return rlist


# In[61]:


query_id = 1
for q in query:
    result = kl_rank(q, unigram, stopwords, Collection, 10)
    directory = os.getcwd().replace("code","result")
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result[:100]:
        n = n + 1
        txt.write(str(query_id) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + "BM25" +"\n")
    txt.close()
    query_id += 1

