
"""
Prepping People & Genre Node Data
Author: Mei Yong

"""

import pandas as pd
import numpy as np


df = pd.read_csv("movies_imdb.csv")


# Get clean, distinct people names
persons = list(df['director']) + list(df['actor_1']) + list(df['actor_2']) + list(df['actor_3'])
persons = list(set(persons))
persons = [person for person in persons if str(person).lower() != 'nan']
persons_df = pd.DataFrame({'id': range(1, len(persons)+1), 
                           'name': persons})
persons_df.to_csv("movie_persons.csv", index=False)


# Get clean, distinct genre names
genres = list(df['genre_1']) + list(df['genre_2']) + list(df['genre_3'])
genres = list(set(genres))
genres = [genre for genre in genres if str(genre).lower() != 'nan']
genres_df = pd.DataFrame({'id': range(1, len(genres)+1), 
                           'name': genres})
genres_df.to_csv("movie_genres.csv", index=False)
