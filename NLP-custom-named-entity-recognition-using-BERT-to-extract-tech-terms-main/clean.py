# -*- coding: utf-8 -*-

import pandas as pd 
import collections 

# this function: 
# 1. read txt files where data is tagged with B,I,O
# 2. clean the tags and tranform format from dataframe to list of lists
# 3. return list of lists for both sentences and tags

def load_and_clean_data(file_name_list):
    #  read all training data from a number of txt files
    if len(file_name_list)>=2:       
        df = [pd.read_csv("data/all_tagged_data/{}.txt".format(file_name),header = None, names=["text", "-", "--","mixed_tag"],skip_blank_lines= False,sep = " ") for file_name in file_name_list]
        df = pd.concat(df)
    else:        
        df = pd.read_csv("data/all_tagged_data/{}.txt".format(''.join(file_name_list)),header = None, names=["text", "-", "--","mixed_tag"],skip_blank_lines= False,sep = " ") 
    
    df = df.fillna(" ")
    df['tag'] = df['mixed_tag'].apply(lambda x : x.split("-")[0] if x.startswith(('B-','I-')) else x)
    print(df['tag'].value_counts())
    
    # transform data from dataframe to list of lists
    tokens = []
    token_tags = []
    sentences = []
    tags = []
    for _,row in df.iterrows():   
        if row['tag']!=" " and row['text']!=" ":
            tokens.append(row['text'])
            token_tags.append(row['tag'])                  
        else:            
            sentences.append(tokens)
            tags.append(token_tags) 
            
            tokens = []
            token_tags = []  
            
    # make sure the lengths of sentences and tags are equal         
    len_label = []    
    for l in tags:
        len_label.append(len(l))    
    len_text = [] 
    for t in sentences:
        len_text.append(len(t))
    
    if collections.Counter(len_label) == collections.Counter(len_text): 
        print ("The lengths of tokens and tags are equal") 
    else : 
        print ("The lengths of tokens and tags are not equal")
    print("the number of examples is {}".format(len(len_label)))
    
    return sentences, tags