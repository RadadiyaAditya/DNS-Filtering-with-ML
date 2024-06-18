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
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import random
from sentence_transformers import SentenceTransformer

# method to disable SSL warning from HTTP responses.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""##Collecting Keywords
get the keywords from given URL, which later get compared to filter words
"""

def get_keywords(domain_name):
  baseurl = "https://www.bing.com/search?"
  search_name=domain_name
  params = {
    "q": search_name
  }
  url = baseurl + urllib.parse.urlencode(params)
  try:
        res = requests.get(url, verify=False)
        res.raise_for_status()
  except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
  # Parse Beautiful Soup and get keywords
  soup = BeautifulSoup(res.text, features="html.parser")
  text = soup.get_text()
  start = text.find('results')
  end = text.find('© 2024 Microsoft')
  keywords = text[start:end]

  # eliminate empty or punctuation only strings
  def disallowed(string):
    if len(string) == 0:
      return False
    if all(x in ['.', '|', '<', '>', '›', '-', '—', '_', '\n'] for x in list(set(string))):
      return False
    return True

  def process(string):
    return string.replace('\n', '')
  raw_keywords = list(filter(disallowed,  keywords.split(' ')[1:]))
  processed_keywords = map(process, raw_keywords)
  raw_keywords = preprocess(raw_keywords)
  return raw_keywords

"""##Word Preprocessing
a preprocessing step for keywords like case lowering, remove punctuation,
remove stopwords, and repititions as these negatively effect the ML method.
"""

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

"""##Checking Similarity
This function find the similarity between filterwords and keywords using the BERT embedding of the words and return similarity value.

"""

def similarity_check_util(keywords,filterwords):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    avg_sim = []
    for filterword in filterwords:
        compare_value = model.encode([filterword])[0]
        similarities = []
        for (word, count) in keywords:
            key_value = model.encode([word])[0]
            similarity = np.dot(key_value, compare_value) / ((math.sqrt(np.dot(key_value, key_value))) * (math.sqrt(np.dot(compare_value, compare_value))))
            similarities.append(similarity)
        avg_sim.append(np.max(similarities))
    return np.mean(avg_sim)

# this is like a main function where required function calls are made to use the utility functions defined.
def similarity_check(servername):
  keywords = get_keywords(servername)
  ## List of some filter words to block pirated and torrent websites
  filterword = ['Forest','buy','order','shop', 'cart']
  sim = similarity_check_util(keywords,filterword)
  print("\nSimilarity check on " + servername + ": " + str(sim) + "\n")
  return sim