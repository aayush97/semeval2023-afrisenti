import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os
import re



def combine_pos_neg_lexica(dict1, dict2):
  set1 = set(dict1.keys())
  set2 = set(dict2.keys())
  set3 = set1.intersection(set2)
  if len(set3)==0:
    dict1.update(dict2)
  else:
    for k in list(set3):
      dict1.pop(k)
      dict2.pop(k)
      dict1.update(dict2)
  return dict1



def lexica_df(path):
  df = pd.read_csv(path)
  return df



def lexica_dictionary(lexica_df):
  return lexica_df.set_index('Word').T.to_dict('list')



def predict_by_lexica(dictionary, review):
  pos_score = 0
  neg_score = 0
  for k,v in dictionary.items():
    k = ' '+k+' '
    if k in review and v == 'NEGATIVE': 
      neg_score += 1
    elif k in review and v == 'POSITIVE':
      pos_score += 1
    else:
      continue

  predicted_label = '0000!!!!'

  if pos_score<0 or neg_score<0:
    print('something wrong')
    return
  else:
    if pos_score==0 and neg_score != 0:
      predicted_label = 'negative' 
    elif pos_score != 0 and neg_score==0:
      predicted_label = 'positive'
    elif pos_score==0 and neg_score==0:
      predicted_label = 'neutral'
    else:
      ratio = pos_score/neg_score
      if ratio > 1:
        predicted_label = 'positive'
      elif ratio < 1:
        predicted_label = 'negative'
      else:
        predicted_label = 'neutral'

  return predicted_label



def predict_by_lexica_2(dictionary, review): # very slow
  pos_score = 0
  neg_score = 0
  for k,v in dictionary.items():
    l = re.findall(" "+k+" |"+k+" | "+k+"|^"+k+"$", review)
    length = len(l)
    if length > 0 and v == 'NEGATIVE': 
      neg_score += 1*length
    elif length > 0 and v == 'POSITIVE':
      pos_score += 1*length
    elif length < 0:
      print('something wrong!')
    else:
      continue

  predicted_label = '0000!!!!'

  if pos_score<0 or neg_score<0:
    print('something wrong')
    return
  else:
    if pos_score==0 and neg_score != 0:
      predicted_label = 'negative' 
    elif pos_score != 0 and neg_score==0:
      predicted_label = 'positive'
    elif pos_score==0 and neg_score==0:
      predicted_label = 'neutral'
    else:
      ratio = pos_score/neg_score
      if ratio > 1:
        predicted_label = 'positive'
      elif ratio < 1:
        predicted_label = 'negative'
      else:
        predicted_label = 'neutral'

  return predicted_label



def filter_stop_words(stopwords_file, review):
  words = pd.read_csv(stopwords_file).word
  words = list(words)
  processed_review = " ".join([w for w in review.split() if not w in words])
  return processed_review



def strip_whitespace_like_unicode(string):
  l1 = string.split()
  l2 = []
  for j in l1:
    if not re.match('\W', j):
      l2.append(j)
  return ' '.join(l2)



def dictionary_processing(dictionary):
  new_dict = {}
  for k,v in dictionary.items():
    if re.findall('\\u200d', k) != []:
      k = re.sub('\\u200d', "", k)
      k = k.strip()
      k = strip_whitespace_like_unicode(k)
      new_dict[k] = v[0]
    else:
      k = k.strip()
      k = strip_whitespace_like_unicode(k)
      new_dict[k] = v[0]
  return new_dict
  
  

def preprocessing_review(review):
  review = review.lower()
  return review



def preprocessing_review_2(review):
  review = review.lower()

  # review = re.sub('-', " ", review) # these two lines effect decrease acc?
  # review = re.sub('\'', " ", review)

  review = re.sub('!', " ", review)
  review = re.sub('\?', " ", review)
  review = re.sub(',', " ", review)
  review = re.sub('`', " ", review)
  review = re.sub('~', " ", review)
  review = re.sub('_', " ", review)
  review = re.sub('#', " ", review)
  review = re.sub('@user', " ", review)
  review = re.sub('[\W]https?://.*\s|[\W]https?://.*\Z|^https?://.*$', " ", review)
  review = re.sub('\.', " ", review)
  return review



def evaluate(file_path, dictionary, fsw=False, stopwords_file=None, pr='1', predict2=False): 
  # fsw: filter stop words
  # stopwords_file: path to stopwords file
  # pr: processing review, 1 means preprocessing_review, 2 means preprocessing_review_2
  # predict2: use predict_by_lexica_2 if true, otherwise use predict_by_lexica
  data_and_labels = pd.read_csv(file_path, sep='\t') 
  data = data_and_labels.text
  trues = data_and_labels.label 
  preds = []
  l = list(trues)
  for review in data: 
    if pr=='1':
      review = preprocessing_review(review) 
    
    if pr=='2':
      review = preprocessing_review_2(review)

    if fsw==True:
      review = filter_stop_words(stopwords_file, review) 
      
    if predict2==True:
      predicted_label = predict_by_lexica_2(dictionary, review) 
      preds.append(predicted_label)
    else:
      predicted_label = predict_by_lexica(dictionary, review) 
      preds.append(predicted_label)
    
  
  return classification_report(l, preds)






# =============================================================================
def test():
  print(':)')
  # return str(['NEGATIVE'])
  # a = {1:'a', 2:'b', 3:'c', 4:'d'}
  # print(set(a.keys())) # {1, 2, 3, 4}
  # a = {}
  # print(a)
  # print(set()=={})
  # a = set()
  # print(a=={})
  # print(len(a) == 0)
  # x = {"apple", "banana", "cherry"}
  # y = {"google", "microsoft", "apple"}
  # z = x.intersection(y) 
  # print(z)
  # print(list(z)[0])
  # d1 = {"apple":1, "banana":2, "cherry":3}
  # d2 = {"google":1, "microsoft":2, "apple":3}
  # d1.pop(list(z)[0])
  # print(d1)
  # d2.pop(list(z)[0])
  # print(d2)
  # d3 = {**d1, **d2}
  # print(d3)
  # d5 = {"google":1, "microsoft":2, "a":3}
  # d4 = d1.update(d5)
  # print(d4)
  # print(d1)
  # print(combine_pos_neg_lexica(d1, d2))
  # path1 = os.path.abspath("data/external/sentiment_lexicon/ig/igbo_negative.csv") 
  # path2 = os.path.abspath("data/external/sentiment_lexicon/ig/igbo_positive.csv")
  # df1 = lexica_df(path1)
  # dict1 = lexica_dictionary(df1)
  # dict1 = dictionary_processing(dict1)
  # df2 = lexica_df(path2)
  # dict2 = lexica_dictionary(df2)
  # dict2 = dictionary_processing(dict2)
  # print(len(combine_pos_neg_lexica(dict1, dict2)))
  # for k,v in combine_pos_neg_lexica(dict1, dict2).items():
  #   print(k, v, sep=',')
  # d = combine_pos_neg_lexica(dict1, dict2)
  # str = 'The queen 👑 has said it all !! Mercy Nwa Mara Mma anyi fu gi na anya❤ https://t.co/oblyK4NjMw'
  # print(predict_by_lexica(d, str))
  # str = 'Kee ka agadi ga emeghe onu waaaaa, na-ekwu tuu nnoo? Animal talk. Animal talk https://t.co/bO8HplNYvq'
  # str = 'Happy Sunday ndi nkem December with grace... #shuga #Jesusdeysugarmybody @user Abakiliki,Ebonyi state https://t.co/p0696wKUvA'
  # str = 'https://t.co/p0696wKUvA'
  # print(preprocessing_review(str))



def main():
  # get sentiment lexicons dictionaries paths
  path1 = os.path.abspath("data/external/sentiment_lexicon/ig/igbo_negative.csv") 
  path2 = os.path.abspath("data/external/sentiment_lexicon/ig/igbo_positive.csv")
  # read from csv and store in df, create dictionary and clean dictionary
  df1 = lexica_df(path1)
  dict1 = lexica_dictionary(df1)
  dict1 = dictionary_processing(dict1)
  df2 = lexica_df(path2)
  dict2 = lexica_dictionary(df2)
  dict2 = dictionary_processing(dict2)
  # combine the pos and neg sentiment lexicons dictionaries
  d = combine_pos_neg_lexica(dict1, dict2)
  # get datasets paths
  test_path = os.path.abspath("data/raw/train/splitted-train-dev-test/ig/test.tsv")
  train_path = os.path.abspath("data/raw/train/splitted-train-dev-test/ig/train.tsv")
  dev_path = os.path.abspath("data/raw/train/splitted-train-dev-test/ig/dev.tsv")
  # get stopwords path
  stop_path = os.path.abspath("data/external/stopwords/igbo_stopwords.csv")


  # experiment 1 -- baseline (review lowercased only)
  print('EXPERIMENT 1')
  print('====================================================')
  print('splitted-train-dev-test/ig/test.tsv --baseline (review lowercased only)')
  print(evaluate(test_path, d))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/train.tsv --baseline (review lowercased only)')
  print(evaluate(train_path, d))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/dev.tsv --baseline (review lowercased only)')
  print(evaluate(dev_path, d))
  print('----------------------------------------------------')
  print('\n')


  # experiment 2 -- baseline + stopwords filtered
  print('EXPERIMENT 2')
  print('====================================================')
  print('splitted-train-dev-test/ig/test.tsv --baseline + stopwords filtered')
  print(evaluate(test_path, d, fsw=True, stopwords_file=stop_path))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/train.tsv --baseline + stopwords filtered')
  print(evaluate(train_path, d, fsw=True, stopwords_file=stop_path))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/dev.tsv --baseline + stopwords filtered')
  print(evaluate(dev_path, d, fsw=True, stopwords_file=stop_path))
  print('----------------------------------------------------')
  print('\n')


  # experiment 3 -- baseline + review lowercased and punctuations removed
  print('EXPERIMENT 3')
  print('====================================================')
  print('splitted-train-dev-test/ig/test.tsv --baseline + review lowercased and punctuations removed')
  print(evaluate(test_path, d, pr='2'))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/train.tsv --baseline + review lowercased and punctuations removed')
  print(evaluate(train_path, d, pr='2'))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/dev.tsv --baseline + review lowercased and punctuations removed')
  print(evaluate(dev_path, d, pr='2'))
  print('----------------------------------------------------')
  print('\n')


  # experiment 4 -- baseline + predicted with frequency
  print('EXPERIMENT 4')
  print('====================================================')
  print('splitted-train-dev-test/ig/test.tsv --baseline + predicted with frequency')
  print(evaluate(test_path, d, predict2=True))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/train.tsv --baseline + predicted with frequency')
  print(evaluate(train_path, d, predict2=True))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/dev.tsv --baseline + predicted with frequency')
  print(evaluate(dev_path, d, predict2=True))
  print('----------------------------------------------------')
  print('\n')


  # experiment 5 -- baseline + stopwords filtered + review lowercased and punctuations removed
  print('EXPERIMENT 5')
  print('====================================================')
  print('splitted-train-dev-test/ig/test.tsv --baseline + stopwords filtered + review lowercased and punctuations removed')
  print(evaluate(test_path, d, fsw=True, stopwords_file=stop_path, pr='2'))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/train.tsv --baseline + stopwords filtered + review lowercased and punctuations removed')
  print(evaluate(train_path, d, fsw=True, stopwords_file=stop_path, pr='2'))
  print('----------------------------------------------------')
  print('splitted-train-dev-test/ig/dev.tsv --baseline + stopwords filtered + review lowercased and punctuations removed')
  print(evaluate(dev_path, d, fsw=True, stopwords_file=stop_path, pr='2'))
  print('----------------------------------------------------')
  print('\n')



if __name__ == "__main__":
  main()
