# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import TensorDataset,DataLoader, SequentialSampler
import nltk
nltk.download('punkt')

"""
tech_term_extractor: detect_tech_terms_in_articles_csv()
The function reads a csv file with one column "text" containing articles. One article each row. 
It returns a dataframe with two columns "text" and "tech terms" containing the articles and the predicted tech terms respectively. One article and a list of identified tech terms each row. 
"""


BATCH_SIZE = 8
NUM_WORKERS = 1
def detect_tech_terms_in_df_articles(df_articles, model_and_tokenizers_dir):
    df_articles['tokens'] = df_articles['text'].apply(lambda x : nltk.word_tokenize(x))
    df_articles['tokenized_text'] = df_articles['tokens'].apply(lambda x : ' '.join(x))
    df_articles['length'] = df_articles['text'].apply(lambda x : len(nltk.word_tokenize(x)) )
    threshold1 = 500
    threshold2 = 1500 

    def tech_term_detector(articles):
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #try to use GPU.If not, use CPU
      #device = torch.device("cpu")
      #device = torch.device("cuda") 
      loaded_model = AutoModelForTokenClassification.from_pretrained(model_and_tokenizers_dir)
      loaded_model = loaded_model.to(device)
      loaded_tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizers_dir)
      label_map_reverse = {2: 'B-tech', 1:'I-tech', 0:'O'}
      loaded_model.eval()
      total_tech_terms = []
      predictions = []
      new_labels_for_articles = []
      new_tokens_for_articles = []

      pt_articles = loaded_tokenizer(articles, padding=True, truncation=True,return_tensors="pt", is_split_into_words=False)
      input_ids = torch.stack(tuple(pt_articles['input_ids']), dim=0)
      attention_masks = torch.stack(tuple(pt_articles['attention_mask']), dim=0)
      dataset = TensorDataset(input_ids, attention_masks)
      dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset),batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)
      for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask= batch
        with torch.no_grad():
          outputs = loaded_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits = outputs[0]
          logits = logits.detach().cpu().numpy()
          predictions.append(logits)

      all_predictions = np.concatenate(predictions, axis=0)
      predicted_label_ids = np.argmax(all_predictions, axis=2)
      
      #merging sub-tokens to words for all articles
      for num, predicted_label_id in enumerate(predicted_label_ids):
        tokenized_article = loaded_tokenizer.convert_ids_to_tokens(pt_articles["input_ids"][num]) 
        new_tokens = []
        new_labels = []
        #within one article
        for token, label_idx in zip(tokenized_article, predicted_label_id):
          if (token != "[CLS]" and token !="[SEP]" and token !="[PAD]"):
            if token.startswith('##'):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(label_map_reverse[label_idx])
                new_tokens.append(token)
        new_labels_for_articles.append(new_labels)
        new_tokens_for_articles.append(new_tokens)

      for (new_tokens_for_article, new_labels_for_article) in zip(new_tokens_for_articles, new_labels_for_articles):
        tech_terms = []
        for index, (word,label) in enumerate(zip(new_tokens_for_article, new_labels_for_article)):
          if label == 'B-tech':
            if index !=0:
              if new_labels_for_article[index-1] != 'O': #if previous word's label is B or I, then combine words to a phrase
                tech_terms[-1] = tech_terms[-1]+" "+word
              else:  
                tech_terms.append(word)
            else:
                tech_terms.append(word)        
          elif label == 'I-tech':
            if index == 0:
              tech_terms.append(word)
            else:
              if new_labels_for_article[index-1] != 'O':
                tech_terms[-1] = tech_terms[-1]+" "+ word
              else:
                tech_terms.append(word)
        total_tech_terms.append(tech_terms)
      return total_tech_terms

    # for articles under threshold
    df_under_thre = df_articles[df_articles['length']<threshold1]
    if df_under_thre.empty == False:
        tech_terms_under = tech_term_detector(df_under_thre.tokenized_text.tolist())
        df_tech_terms_under = pd.DataFrame(zip(df_under_thre.index,df_under_thre.tokenized_text.tolist(),tech_terms_under),columns=['NO.','text','tech_terms']) #.set_index(data.iloc[:, 0])
        df_tech_terms_under1 = df_tech_terms_under.drop(['NO.'], axis=1, inplace=False)
    else:
        pass

    def split_combine_output(df, num_split):      
        splited_sents = []
        for row_index, row in df.iterrows():
            for i in range(num_split):
                sents  = ' '.join(row['tokens'][round(row['length']*i/num_split):round(row['length']*(i+1)/num_split)])
                splited_sents.append(sents)
                
        tech_terms = tech_term_detector(splited_sents)
        index_inter = [[str(index) for index in df.index],[str(num) for num in list(range(num_split))]]
        mul_index = pd.MultiIndex.from_product(index_inter, names=["NO.", "split_index"])
        df_tech_terms = pd.DataFrame(zip(splited_sents,tech_terms), index = mul_index, columns=['text','tech_terms'])
        df_tech_terms = df_tech_terms.groupby(level='NO.').agg(text = pd.NamedAgg(column ='text',aggfunc= sum), tech_terms = pd.NamedAgg(column = 'tech_terms', aggfunc = sum)).reset_index() #.reset_index(name = 'index')      
        df_tech_terms1 = df_tech_terms.drop(['NO.'], axis=1, inplace=False)
        return df_tech_terms,df_tech_terms1


    # for articles b/w threshold1 and threshold2
    df_bw_thre = df_articles[(df_articles['length']>=threshold1)&(df_articles['length']<threshold2)]
    if df_bw_thre.empty == False:
       df_tech_terms_bw, df_tech_terms_bw1 =split_combine_output(df_bw_thre, 3)
    else:
        pass

    # for articles over threshold2
    df_over_thre = df_articles[df_articles['length']>=threshold2]
    if df_over_thre.empty == False:  
       df_tech_terms_over, df_tech_terms_over1 =split_combine_output(df_over_thre, 10)
    else:
        pass
    
    def combine_tech_term_dfs(tuple_dfs):    
        df_tech_terms= pd.DataFrame(np.concatenate(tuple_dfs,axis = 0), columns= ['NO.','text', 'tech_terms'])
        df_tech_terms['NO.'] = df_tech_terms['NO.'].astype('int')
        df_tech_terms.sort_values(by=['NO.'], inplace=True)
        df_tech_terms.drop(['NO.'], axis=1, inplace=True)
        return df_tech_terms


    if df_under_thre.empty == False and df_bw_thre.empty == False and df_over_thre.empty == True:
        df_tech_terms_full = combine_tech_term_dfs((df_tech_terms_under, df_tech_terms_bw))
        return df_tech_terms_full
    elif df_under_thre.empty == False and df_bw_thre.empty == True and df_over_thre.empty == False:
        df_tech_terms_full = combine_tech_term_dfs((df_tech_terms_under, df_tech_terms_over))
        return df_tech_terms_full
    elif df_under_thre.empty == True and df_bw_thre.empty == False and df_over_thre.empty == False:
        df_tech_terms_full = combine_tech_term_dfs((df_tech_terms_bw, df_tech_terms_over))
        return df_tech_terms_full
    elif df_under_thre.empty == False and df_bw_thre.empty == False and df_over_thre.empty == False:
        df_tech_terms_full = combine_tech_term_dfs((df_tech_terms_under,df_tech_terms_bw,df_tech_terms_over))
        return df_tech_terms_full
    elif df_under_thre.empty == False and df_bw_thre.empty == True and df_over_thre.empty == True:
        return df_tech_terms_under1
    elif df_under_thre.empty == True and df_bw_thre.empty == False and df_over_thre.empty == True:
        return df_tech_terms_bw1
    elif df_under_thre.empty == True and df_bw_thre.empty == True and df_over_thre.empty == False:
        return df_tech_terms_over1
    else:
        print("Can't find any articles in the csv file.")

    return



def tech_term_detector_single_article(article, model_and_tokenizers_dir):
    loaded_model = AutoModelForTokenClassification.from_pretrained(model_and_tokenizers_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizers_dir)
    label_map_reverse = {2: 'B-tech', 1:'I-tech', 0:'O'}
    loaded_model.eval()
  #total_tech_terms = []
  #for article in articles:
    new_tokens = []
    new_labels = []
    tech_terms = []

    pt_batch = loaded_tokenizer(
    article,
    padding=True,
    truncation=True,
    is_split_into_words=False,
    return_tensors="pt")

    pt_outputs = loaded_model(**pt_batch)
    logits = pt_outputs[0]
    predicted_label_id = np.argmax(logits.detach().numpy(), axis=2)
    tokenized_article = loaded_tokenizer.convert_ids_to_tokens(pt_batch["input_ids"][0])

    for token, label_idx in zip(tokenized_article, predicted_label_id[0]):

      if token != "[CLS]" and token !="[SEP]":
        if token.startswith('##'):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(label_map_reverse[label_idx])
            new_tokens.append(token)

    #prediction  = pd.DataFrame(zip(new_tokens, new_labels), columns = ['word','pred_label'])

    for index, (word,label) in enumerate(zip(new_tokens, new_labels)):
      if label == 'B-tech':
        if index !=0:
          if new_labels[index-1] != 'O': #if previous word's label is B or I, then combine words to a phrase
            tech_terms[-1] = tech_terms[-1]+" "+word
          else:  
            tech_terms.append(word)
        else:
            tech_terms.append(word)        
      elif label == 'I-tech':
        if index == 0:
          tech_terms.append(word)
        else:
          if new_labels[index-1] != 'O':
            tech_terms[-1] = tech_terms[-1]+" "+word
          else:
            tech_terms.append(word)  
    #total_tech_terms.append(tech_terms)                         

    return tech_terms