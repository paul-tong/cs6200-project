#!/usr/bin/env python
# coding: utf-8

# # Phase 1

# In[1]:


import requests
import os
from bs4 import BeautifulSoup
from queue import Queue, Empty
from urllib.parse import urljoin, urlparse
from nltk.util import ngrams
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
nltk.download('punkt')
import pandas as pd
import re
import collections
import string
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


stops = os.getcwd().replace("phase1","general")+"\\test-collection\\common_words" #load stopwords
f = open(stops, 'r',encoding='utf-8')
stopwords = f.read().split("\n")


# In[3]:


Collection = [] # load documents and do parsing job
Doc_name = []
folder_d = os.getcwd().replace("phase1","general")+"\\test-collection\\cacm"
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


# In[4]:


file_query = os.getcwd().replace("phase1","general")+"\\test-collection\\cacm.query.txt" #load query list
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


# ## Task 1

# In[5]:


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


# In[6]:


unigram = Create_Inverted_Index(1,Collection)


# In[7]:


s = 0 
for d in Collection:
    content = d.split(" ")
    s = s + len(content)
avgdl = s/len(Collection) # Average document length


# ### tfidf

# In[8]:


def tfidf(unigram, Collection): # tdidf vetorizer
    bag = []
    n_collection = len(Collection)
    for word in unigram.keys():
        bag.append(word)
    v_doc = np.zeros((n_collection, len(bag)))
    for i in range(len(bag)):
        word = bag[i]
        index = unigram[word]
        idf = np.log(n_collection/len(index))
        for ind in index:
            did = ind[0]
            freq = ind[1]
            v_doc[did,i] = idf * freq/len(Collection[did].split(" "))
    return v_doc


# In[9]:


v_doc = tfidf(unigram, Collection)


# In[10]:


def tfidf_search(query, v_doc, unigram, Collection): # tfidf model
    q = []
    bag = []
    n_collection = len(Collection)
    for word in unigram.keys():
        bag.append(word)
    v_query = np.zeros(len(bag))
    remove = string.punctuation # remove punctuations
    remove = remove.replace("-", "")
    pattern = r"[{}]".format(remove)
    query =  re.sub(pattern, "", query) 
    for word in query.split():
        q.append(word)
    words =  collections.Counter(q)
    for word in words:
        if word in bag:
            pos = bag.index(word)
            index = unigram[word]
            idf = np.log(n_collection/len(index))
            v_query[pos] = words[word]/len(q)
    Sim_matrix = {}
    for i in range(n_collection):
        Sim_matrix[i] = cosine_similarity([v_doc[i], v_query])[0,1]
    sorted_list = sorted(Sim_matrix.items(), key=lambda kv: kv[1])
    sorted_list.reverse() 
    return sorted_list


# ### Binary Independence Model

# In[11]:


import pandas as pd


# In[12]:


file_query = os.getcwd().replace("phase1","general")+"\\test-collection\\cacm.rel.txt" # load relevance information
f = open(file_query, 'r',encoding='utf-8')
content = f.read()
rel = content.split("\n")
r = []
for line in rel:
    l = line.split(" ")
    r.append(l)
rel = pd.DataFrame(r)


# In[13]:


for i in range(len(rel)-1):
    while (len(rel.iloc[i,2]) < 9):
        rel.iloc[i,2] = rel.iloc[i,2].replace("-","-0")


# In[14]:


def relevance_index(rel,query_id): #BIM model
    query_id = query_id
    q = query[query_id].split(" ")
    if "" in q:
        q.remove("")
    records = rel.loc[rel[0] == str(query_id+1)]
    vec = np.zeros(len(q))
    p = np.zeros(len(q))
    s = np.zeros(len(q))
    rel_list = []
    for i in range(len(records)):
        rel_doc_name = records.iloc[i,2]
        rel_list.append(rel_doc_name)
        rel_doc_id = Doc_name.index(rel_doc_name)
        rel_doc = Collection[rel_doc_id]
        for j in range(len(q)):
            word = q[j]
            if(word.lower() in rel_doc.lower().split(" ")):
                p[j] = p[j] + 1
    nrel_list = list(set(Doc_name) - set(rel_list))
    for i in range(len(nrel_list)):
        nrel_doc_id = Doc_name.index(nrel_list[i])
        nrel_doc = Collection[nrel_doc_id]
        for j in range(len(q)):
            word = q[j]
            if(word.lower() in nrel_doc.lower().split(" ")):
                s[j] = s[j] + 1
    score_map = {}
    for i in range(len(Collection)):
        score = 0
        for j in range(len(q)):
            word = q[j]
            if(word.lower() in Collection[i].lower().split(" ")):
                pi = (p[j]+0.5)/(len(rel_list)+1)
                si = (s[j]+0.5)/(len(nrel_list)+1)
                score = score + np.log((pi*(1-si)) / (si*(1-pi))) 
        score_map[i] = score
    sorted_list = sorted(score_map.items(), key=lambda kv: kv[1])
    sorted_list.reverse() 
    return sorted_list


# ### BM25

# In[15]:


k1=1.2
b=0.75
k2=100


# In[16]:


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


# ### Output

# In[17]:


def write(query_id, option, folder_name): # Task1 output
    query_id = query_id - 1 
    q = query[query_id]
    model = ""
    if(option == 1): #option = 1 -TFIDF
        model = "TFIDF"
        result = tfidf_search(q, v_doc, unigram, Collection)
    elif(option == 2): #option = 2 -BIM
        model = "BIM"
        result = relevance_index(rel,query_id)
    elif(option == 3): #option = 2 -BM25
        model = "BM25"
        result = BM25score(q, unigram, "", Collection)
    else:
        return 0
    directory = os.getcwd() + "\\Task1\\" + folder_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id+1) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result:
        n = n + 1
        txt.write(str(query_id+1) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + model +"\n")
    txt.close()


# In[18]:


def Task1_output(query):
    n = len(query)
    for i in range(n):
        query_id = i + 1
        write(query_id, 1, "tfidf")
        write(query_id, 2, "BIM")
        write(query_id, 3, "BM25")


# In[19]:


Task1_output(query)


# ## Task2

# ### Query Time Stemming Using BM25

# In[20]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()


# In[21]:


QTS_query = [] #Do stemming query expansion
for q in query:
    l = q.split(" ")
    nl = []
    for i in range(len(l)):
        if(l[i] != ""):
            nl.append(ps.stem(l[i]))
    QTS_query.append(q + " ".join(nl))


# In[22]:


query_id = 1 #output
for q in QTS_query:
    result = BM25score(q, unigram, "", Collection)
    directory = os.getcwd() + "\\Task2\\" + "Query Time Stemming"
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result:
        n = n + 1
        txt.write(str(query_id) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + "BM25" +"\n")
    txt.close()
    query_id += 1


# ###  pseudo relevance feedback using BM25

# In[23]:


def getletterbigram(string): #string to bigram
    res_list = []
    for i in range(len(string)-1):
        if string[i] != " " and string[i+1] != " ":
            res_list.append(string[i:i+2])
    return res_list      


# In[24]:


def dice_coef(bi_a,bi_b): # calculate dice coef with two bigram list
    nab = 0
    for bi in bi_a:
        if bi in bi_b:
            nab += 1
    na = len(bi_a)
    nb = len(bi_b)
    return 2*nab/(na+nb)


# In[25]:


def pseudo_search(q, unigram, stopwords, collection, k, n): #n terms be expanded, consider top k document
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
    result = BM25score(q, unigram, stopwords, collection)
    return result


# In[26]:


query[0]


# In[27]:


pseudo_search(query[0], unigram, stopwords, Collection, 10,int(len(query[0].split(" ")[3:])/3))[:5] # top 10 doc


# In[28]:


pseudo_search(query[0], unigram, stopwords, Collection, 20,int(len(query[0].split(" ")[3:])/3))[:5] #top 20 doc


# In[29]:


pseudo_search(query[0], unigram, stopwords, Collection, 5, int(len(query[0].split(" ")[3:])/3))[:5] #top 5 doc


# In[30]:


Collection[1641]


# In[31]:


Collection[1937]


# In[32]:


query_id = 1
for q in query:
    result = pseudo_search(q, unigram, stopwords, Collection, 10, int(len(q.split(" ")[3:])/3))
    directory = os.getcwd() + "\\Task2\\" + "pseudo"
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result:
        n = n + 1
        txt.write(str(query_id) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + "BM25" +"\n")
    txt.close()
    query_id += 1


# Justification: 
# 
# In Pseudo Relevant Feedback, we choose to use BM25 baseline model to find the most relevant documents. And in stemming expansion, we choose to use nltk package to do stemming processing.
# 
# For Pseudo Relevant Feedback, we choose to consider top 20 documents. From the experiments we performed, we can find that if we consider k = 20, 10, 5 situations, k = 20 gives us most relevant documents. From the baseline BM25 model, the score of top 20 documents are pretty close which means top 20 documents are all pretty relevant to the query, so k = 20 is a proper decision. About n's value, we choose to use one-third the length of the query. Because if we set n with a small number the expansion is meaningless and if we set n with a large number some of the expansion term is unrelevant to the query. After experiments, we decide to let n = one-third the length of the query.

# ## Task3

# ### Stopwords

# In[33]:


query_id = 1 #add stopword processing
for q in query:
    result = BM25score(q, unigram, stopwords, Collection)
    directory = os.getcwd() + "\\Task3\\" + "Stopwords"
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result:
        n = n + 1
        txt.write(str(query_id+1) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + "BM25" +"\n")
    txt.close()
    query_id += 1


# ### Stemming Search

# In[34]:


file_query_stemming = os.getcwd().replace("phase1","general")+"\\test-collection\\cacm_stem.query.txt"
f = open(file_query_stemming, 'r',encoding='utf-8')
stem_query = f.read().split("\n")
stem_query.remove("")


# In[35]:


file_query_stemming = os.getcwd().replace("phase1","general")+"\\test-collection\\cacm_stem.txt"
f = open(file_query_stemming, 'r',encoding='utf-8')
stem_doc = f.read().split("#")
stem_doc.remove("")


# In[36]:


docID = 0 #parsing the documents
stem_doc_refine = []
for line in stem_doc:
    docID += 1
    ID = " " + str(docID)
    content = line.replace(ID,"").replace("\n" , "").replace("pm","")
    content = re.sub(r'[0-9]+', '', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.replace(" cacm" + content.split("cacm")[-1],"")
    stem_doc_refine.append(content)


# In[37]:


unigram_stem = Create_Inverted_Index(1,stem_doc_refine)


# In[38]:


query_id = 1
for q in stem_query:
    result = BM25score(q, unigram_stem, "", stem_doc_refine)
    directory = os.getcwd() + "\\Task3\\" + "stem"
    if not os.path.exists(directory):
        os.makedirs(directory)
    n = 0
    txt = open(directory + "\\"+ "Query" + str(query_id) + ".txt",'w',encoding='utf-8')
    txt.write("Query:" + q[3:] +"\n")
    for record in result:
        n = n + 1
        txt.write(str(query_id+1) + ", "+ "Q0" + ","  + Doc_name[record[0]] + "," + str(n) + "," + str(record[1]) + "," + "BM25" +"\n")
    txt.close()
    query_id += 1


# In[39]:


stem_query


# In[40]:


BM25score(stem_query[2], unigram_stem, "", stem_doc_refine)[:10]


# In[41]:


BM25score(stem_query[3], unigram_stem, "", stem_doc_refine)[:10]


# In[42]:


BM25score(stem_query[6], unigram_stem, "", stem_doc_refine)[:10]


# In[43]:


Collection[2663]


# In[44]:


Collection[2277]


# In[45]:


Collection[1261]


# By the experiment we performed, the result is satisfying and pretty relevant to those queries.
# 
# I selected three queries which are relevant in some way, and we can find some similarities in the result. In document CACM-2664 (whose id in my program is 2663) contains both "parallel", "process" and it was ranked in both queries ("parallel algorithm", "parallel processor in inform retriev").
