
"""
Webscrape IMDB for movie info
Author: Mei Yong

"""


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re 

# Import movies
movies = pd.read_csv("movies.csv")

# Get html from an imdb page and parse it 
def get_imdb_page(imdb_id):
    url = "https://www.imdb.com/title/" + imdb_id
    response = requests.get(url) 
    response_parsed = BeautifulSoup(response.text,'html.parser')  
    return response_parsed



'''
# testing

# iron man
imdb_page = get_imdb_page('tt0371746')
imdb_page_preview = str(imdb_page)

# get summary text
imdb_page.find('div', class_ = 'summary_text').text.strip('\n').strip()

# get director and cast names - see items with div tag or credit_summary_item class
for div in imdb_page.findAll('div', attrs={'class':'credit_summary_item'}):
    if div.find('h4').contents[0]=='Director:':
      print(div.find('a').contents[0])
    elif div.find('h4').contents[0]=='Stars:':
      print([actor.contents[0] for actor in div.findAll('a')[0:3]])
      
# get genres
for div in imdb_page.findAll('div', attrs={'class':'subtext'}):
    print([genre.contents[0] for genre in div.findAll('a')[0:3]]) 
    
'''

# Extract summary, director, cast, genres
def get_imdb_info(imdb_id):
    
    imdb_page = get_imdb_page(imdb_id)
    imdb_info = {}
  
    # Get summary
    imdb_info['summary_text'] = \
    imdb_page.find('div', class_ = 'summary_text').text.strip('\n').strip()
  
    # Get main director and top 3 cast members
    for div in imdb_page.findAll('div', attrs={'class':'credit_summary_item'}):
        if div.find('h4').contents[0]=='Director:':
            imdb_info['director'] = div.find('a').contents[0]
        elif div.find('h4').contents[0]=='Stars:':
            imdb_info['actors'] = [actor.contents[0] for actor in div.findAll('a')[0:3]]
  
    # Get genres
    for div in imdb_page.findAll('div', attrs={'class':'subtext'}):
        imdb_info['genres'] = [genre.contents[0] for genre in div.findAll('a')[0:3]]
  
    return imdb_info


'''
# test - 5 mins for 5 movies
import time
time_start = time.time()
imdb_info = list(movies.head()['imdb_id'].apply(get_imdb_info))
print('time elapsed:',(time.time() - time_start))

'''


# here's one I prepared earlier lol
import json
with open('imdb_info.txt') as f:
    imdb_info = []
    for line in f:
        item = json.loads(line)
        imdb_info.append(item)
    
imdb_df = pd.DataFrame(imdb_info)



# Fill lists with nulls so that there are 3 items in the list - for later column splitting
def add_nans(imdb_list):
    if len(imdb_list)==2:
        imdb_list.append(np.nan)
    elif len(imdb_list)==1:
        imdb_list.append([np.nan,np.nan])
    return imdb_list


imdb_df['actors'] = imdb_df['actors'].apply(add_nans)
imdb_df['genres'] = imdb_df['genres'].apply(add_nans)

# Some of the genres have dates - replace these with nulls
def remove_dates(genres):
    new_genres = []
    for genre in genres:
        genre = str(genre)
        if any(char.isdigit() for char in genre):
            new_genres.append(np.nan)
        else:
            new_genres.append(genre)
    return new_genres

imdb_df['genres'] = imdb_df['genres'].apply(remove_dates)



# Split a column containing lists (of length 3) into 3 columns
def explode_cols(df, input_col, output_cols):
    df[output_cols] = pd.DataFrame(df[input_col].values.tolist(), index=df.index)
    df = df.drop(input_col, axis=1)
    return df

imdb_df = explode_cols(df=imdb_df, input_col='actors', output_cols=['actor_1','actor_2','actor_3'])
imdb_df = explode_cols(df=imdb_df, input_col='genres', output_cols=['genre_1','genre_2','genre_3'])
    


# Extract top 3 descriptors from the summary - TF-IDF
# http://www.tfidf.com/



# Cleanse summary
def pre_process(text):
    text = text.lower() # lowercase
    text = re.sub("<!--?.*?-->", "", text) # remove tags
    text = re.sub("(\\d|\\W)+", " ", text) # remove special characters and digits
    return text
 
imdb_df['summary_text_new'] = imdb_df['summary_text'].apply(lambda x: pre_process(x))



# Create word count vectors - i.e. number of occurences of words in each summay text item
from sklearn.feature_extraction.text import CountVectorizer
summaries = imdb_df['summary_text_new'].tolist()
cv = CountVectorizer(stop_words='english')
word_count_vector = cv.fit_transform(summaries)

# Convert the word count vectors into TF-IDF vectors
# http://www.tfidf.com/
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)



# Helper functions to get the top 3 descriptive words from summary text
	
# Sort the tf-idf vectors by descending order of scores
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# Extract only the top n words and their tf-idf scores - default is top 10
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    
    # Get the topn items from the vector
    sorted_items = sorted_items[:topn]
 
	# Convert into a dictionary and round the scores
	results = {}
	for idx, score in sorted_items:
		results[feature_names[idx]] = round(score,3)
	
	return results
 
# Get list of top 3 key words from the summary text
def get_top_words(doc,num_words=3):

    # Get feature names (i.e. words)
    feature_names=cv.get_feature_names()
 
    # Generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
 
    # Sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
 
    # Extract only the top n words and their tf-idf scores
    keywords=extract_topn_from_vector(feature_names,sorted_items,num_words)

    return list(keywords.keys())


# Get top 3 words and put into a new column
imdb_df['top_descriptors'] = imdb_df['summary_text_new'].apply(lambda x: get_top_words(x,num_words=3))

# Same data prep as before
imdb_df['top_descriptors'] = imdb_df['top_descriptors'].apply(add_nans)
imdb_df = explode_cols(df=imdb_df, input_col='top_descriptors', output_cols=['descriptor_1','descriptor_2','descriptor_3'])



# Join the datasets together and export to CSV
movies_imdb = pd.concat([movies,imdb_df],join='inner',axis=1)
movies_imdb.to_csv("movies_imdb.csv", index=False)





