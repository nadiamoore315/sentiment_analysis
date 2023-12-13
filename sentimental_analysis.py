# Title: Sentimental Analysis Project
##############################################################
# Description/Purpose: Code for running a sentimental analysis
# on Yelp Reviews to detect for how negative/positive a review
# is. Could aid companies on detecting comments that go against
# policy guidelines and to gauge audience interest/disinterest
# for particular businesses, goods, services, etc.
##############################################################
# Author: Nadia Moore
##############################################################
# User Note(s): Original code's tutorial/explanation used for
# code will be linked on the README.md file for reference
##############################################################

# Step 1: Install and Import Dependencies

import pandas as pd
import numpy as np

# AutoTokenizer - allows for string into num sequence for NLP model
# AutoModel...tion - gives architecture from transformers to load in NLP model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# pytorch - use of arg max to get highest result
import torch
# requests - grabs data (i.e. Yelp reviews)
import requests
# BeautifulSoup - allows for traversing of yelp results
from bs4 import BeautifulSoup
# re - create regex function to extract specific comments
import re

# Instantiate Model
def instantiate_model():
    # load up pre-trained NLP model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    global model
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Collect Reviews ("Scraper")
def collect_reviews():
    # grab web page
    r = requests.get('https://www.yelp.com/biz/mejico-sydney-2')

    # parse text to beautiful soup, set parser
    soup = BeautifulSoup(r.text, 'html.parser')

    # look for comment
    regex = re.compile('.*comment.*')

    # find all tags that match format (i.e. class of comment)
    results = soup.find_all('p', {'class':regex})

    # make list of text results (reviews)
    global reviews
    reviews = [result.text for result in results]

# Step 5: Load Review into Dataframe and Score

# put in reviews to calculate sentiments
def sentiment_score(review):

    # encode review into nums
    tokens = tokenizer.encode(review, return_tensors='pt')

    # pass tokens to model
    result = model(tokens)

    # logits represents probability of a specific class being sentiment
    # get largest value to represent the position that's the sentiment
    # value 1-5
    return(int(torch.argmax(result.logits))+1)

def main():

    # Step 2: Instantiate Model
    instantiate_model()

    # Step 4: Collect Reviews ("Scraper")
    collect_reviews()

    # Step 5: Load Review into Dataframe and Score
    # create df of reviews
    df = pd.DataFrame(np.array(reviews), columns = ['review'])

    # get sentiment of each review in df:
        # apply and lambda to loop through each review in column (as var x)
        # NLP is limited to how many tokens you can pass, which is == 512 (can improve in future)
    df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

if __name__ == "__main__":
    main()