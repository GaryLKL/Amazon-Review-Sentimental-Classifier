import os, json
import numpy as np, pandas as pd
import argparse
import contractions
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
from p_tqdm import p_map
import multiprocessing as mp
from multiprocessing import Process, Pool
import string

''' settings '''
# nltk.download('averaged_perceptron_tagger')
en_stop_words = set(stopwords.words('english'))
neg_set = {"no", "nor", "not"}
my_en_stop_words = {w for w in en_stop_words if w not in neg_set}
punctuations = string.punctuation

def read_line_json(path, name_list):
    json_contents = []
    for file_name in name_list:
        with open(os.path.join(path, file_name)) as file:
            for i, line in enumerate(file):
                json_dict = json.loads(line)
                json_dict["category"] = file_name[8:-7] # add a column denoting the category
                json_contents.append(json_dict)
    return json_contents

def json_to_df(selected_cols, json_data):
    data = pd.DataFrame(json_contents).loc[:, cols]
    '1'' Remove duplicated items if existing... '''
    # data.sort_values('asin').drop_duplicates(subset=['reviewerID','reviewText','unixReviewTime','summary','category'],keep='first',inplace=False)
    ''' Save the DataFrame into a csv file if needed... '''
    # data.to_csv()
    return data

def to_empty(texts):
    # represent NaN as ""
    if pd.isnull(texts) or texts == "":
        return ""
    else:
        return texts
        
def remove_tag(texts):
    return BeautifulSoup(texts,'lxml').get_text()

def remove_url(texts):
    return re.sub(r"http\S+", " ", texts) # \S non-space characters

def remove_emoji(texts):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', texts)

def find_emoji(texts):
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return bool(re.match(emoji_pattern, texts))

def decontracted(texts):
    return contractions.fix(texts)

def remove_punc(texts):
    return " ".join("".join([" " if ch in punctuations or not ch.isalpha() else ch for ch in texts]).split())

# POS_tag
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def to_lemma(texts):
    # tokenized and pos tag
    tokens = word_tokenize(texts)
    tagged_sent = pos_tag(tokens)
    
    # lemma
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return lemmas_sent

def clean_tokens(tokens):
    # remove stop words and then transform to lower case
    return [w.lower() for w in tokens if w.lower() not in my_en_stop_words]

def full_step_preprocessing(texts):
    # this function will do preprocessing on a string and then return a list of tokens
    token_result = clean_tokens(
                        to_lemma(
                            remove_punc(
                                decontracted(
                                    remove_emoji(
                                        remove_url(
                                            remove_tag(
                                                to_empty(
                                                    texts
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    
    return token_result

def simple_preprocessing(texts):
    # this function will do preprocessing on a string and then return a list of tokens
    token_result = decontracted(
                        remove_emoji(
                            remove_url(
                                remove_tag(
                                    to_empty(
                                        texts
                                    )
                                )
                            )
                        )
                    )
    return token_result
