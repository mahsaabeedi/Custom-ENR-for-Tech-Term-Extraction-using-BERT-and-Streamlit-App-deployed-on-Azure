#from __future__ import unicode_literals
#from flask import Flask,render_template,url_for,request
# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen
import streamlit as st 
from extractor import tech_term_detector_single_article, detect_tech_terms_in_df_articles
import pandas as pd
import pybase64
model_and_tokenizers_dir = "saved_bert_model_and_tokenizer_m2_v16/"

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.
    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False, encoding='utf-8-sig')
    # some strings <-> bytes conversions necessary here
    b64 = pybase64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

st.title('Technology Detector') 
st.write('**NLP Named Entity Recognition (Token Classification) Using BERT**')
st.write('**The detector detects technologies, tech concepts, inventions, instruments, products. (No brand or entity names)**') 

st.write('**Option 1:** Enter a url below. Our detector will return the scraped text from the url and the technology terms in the text.')
raw_url = st.text_input('Enter a Url:')
if(st.button('SubmitUrl')): 
    try:
        rawtext = get_text(raw_url)
        tech_terms = tech_term_detector_single_article(rawtext, model_and_tokenizers_dir)
    except:
        st.error("Please enter a valid url")
        st.stop() #don't show error details
    if tech_terms:
        tech_terms = "Technology terms: "+ ",".join(tech_terms)
        st.success(tech_terms)
        st.write('**Scraped Text from the Above Url: **', rawtext)
    else:       
        st.info("No technology terms detected.")
        st.write('**Scraped Text from the Above Url: **', rawtext)

st.write('**Option 2:** Enter some text below. Our detector will return the input text and the technology terms in the text.')
name = st.text_area(label = "Enter Some Text:", height = 100)    
if(st.button('SubmitText')): 
    try:
        tech_terms = tech_term_detector_single_article(name, model_and_tokenizers_dir)
    except:
        st.error("Please enter some text")
        st.stop() #don't show error details
    if tech_terms:
        tech_terms = "Technology terms: "+ ",".join(tech_terms)
        st.success(tech_terms)
        st.write('**Input text: **', name)
    else:            
        st.info("No technology terms detected.")
        st.write('**Input Text: **', name)

st.write('**Option 3 (for multiple articles):** Upload a file with one column named "text". Our detector will return a csv file with the input articles and the detected technology terms for each article.')            
col1, col2 = st.beta_columns(2)
with col1:
    st.text("Uploaded file should look like:")
    #st.write(df_sample,width = 310)
    st.image("screenshot_before.png", width = 310,  use_column_width=False)
with col2:
    st.text("Output file available for downloading will look like:")
    #st.write(df_sample,width = 310)
    st.image("screenshot_after.png", width = 310, use_column_width=False)


uploaded_file = st.file_uploader("Upload a file:",type=['csv','xlsx'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)
    if uploaded_file.type =="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":    
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type =="application/vnd.ms-excel":
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Please upload a file (.csv / .xlsx)")
    df_tech_term = detect_tech_terms_in_df_articles(df, model_and_tokenizers_dir)
    df_tech_term.reset_index(drop=True,inplace = True)
    st.write("The output dataframe with the detected techmology terms for each article: ")
    st.dataframe(df_tech_term)    


if st.button('Download the output as a CSV file'):
    try:
        tmp_download_link = download_link(df_tech_term, 'output.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    except:
        st.error("Please upload a file first")