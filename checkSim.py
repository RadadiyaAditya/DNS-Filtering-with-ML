from bs4 import BeautifulSoup
import requests, urllib
import urllib3
from collections import Counter
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
nlp = spacy.load('en_core_web_md')
# method to disable SSL warning from HTTP responses.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# a preprocessing step for keywords like case lowering, remove punctuation,
# remove stopwords, and repititions as these negatively effect the ML method.
def preprocess(keywords):
  sentence = ' '.join(keywords)
  sentence = sentence.lower()
  sentence = re.sub(r'\d+', '', sentence)
  translator = str.maketrans('', '', string.punctuation)
  sentence = sentence.translate(translator)
  stop_words = set(stopwords.words("english"))
  word_tokens = word_tokenize(sentence)
  filtered_text = [word for word in word_tokens if word not in stop_words]
  raw_keywords =  Counter(filtered_text).most_common()
  return raw_keywords[:10]

# method to collect appropriate keywords for the requested url which form as input for the ML model to find similarity between requested page and filter words
def get_keywords(domain_name):
  baseurl = "https://www.google.com/search?"
  params = {
    "q": domain_name
  }
  url = baseurl + urllib.parse.urlencode(params)
  res = requests.get(url, verify=False)
  # Parse Beautiful Soup and get keywords
  soup = BeautifulSoup(res.text, features="html.parser")
  start = soup.get_text().find('ALL')
  end = soup.get_text().find('Sign inSettingsPrivacyTerms')%10000000
  keywords = soup.get_text()[start:end]

  # eliminate empty or punctuation only strings
  def disallowed(string):
    # Do not allow empty strings
    if len(string) == 0:
      return False
    # Do not allow strings that are any combination of these letters ONLY
    if all(x in ['.', '|', '<', '>', '›', '-', '—', '_', '\n'] for x in list(set(string))):
      return False
    return True

  # Process every raw string - like eliminate new lines from them
  def process(string):
    return string.replace('\n', '')
  raw_keywords = list(filter(disallowed,  keywords.split(' ')[1:]))
  processed_keywords = map(process, raw_keywords)
  raw_keywords = preprocess(raw_keywords)
  # Result
  return raw_keywords

# this utility function where similarity between filterwords and keywords is found using the BERT embeddings of the words and finally return a similarity value between keywords and filterwords
def similarity_check_util(keywords, filterwords):
    avg_sim = []
    for filterword in filterwords:
        compare_vector = nlp(filterword).vector  # SpaCy word vector for the filter word
        similarities = []
        for (word, count) in keywords:
            key_vector = nlp(word).vector  # SpaCy word vector for each keyword
            # Compute cosine similarity
            similarity = np.dot(key_vector, compare_vector) / (np.linalg.norm(key_vector) * np.linalg.norm(compare_vector))
            similarities.append(similarity)
        avg_sim.append(np.max(similarities))
    return np.mean(avg_sim)

## Method
# @param servername name of the requested webpage
# @return similarity value [0,1] calculated by the model
#
# this is like a main function where required function calls are made to use the utility functions defined.
def similarity_check(servername):
  keywords = get_keywords(servername)
  ## List of some filter words to block pirated and torrent websites
  filterword = ['proxy','pirate','piracy','torrent', 'download']
  sim = similarity_check_util(keywords,filterword)
  print("\nSimilarity check on " + servername + ": " + str(sim) + "\n")
  return sim
