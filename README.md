# RecomMovies
Building a recommendation model in a graph database using movies data webscraped from IMDB - Python, Neo4j, natural language processing

### Contents
* recom_movies_get_imdb_info.py - webscrape IMDB for movie summary, director, actors + data cleanse + some NLP to get key descriptor words
* recom_movies_prep_persons_genres.py - further data prep for easier Neo4j import (easier to perform data prep in Pythan than Neo4j)
* recom_movies_build_graphdb.txt - Cypher queries to be run in the Neo4j Browser - later to be automated

### Example movie node & its relationships in Neo4j Browser
![alt text](https://raw.githubusercontent.com/mei-yong/RecomMovies/master/images/pacifier.JPG)


