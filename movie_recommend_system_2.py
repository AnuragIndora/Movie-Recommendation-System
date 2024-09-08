# This code is based on Tfidf vectorizer and Cosine Similarity 
# Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space.
# It is often used to measure how similar two documents or texts are based on their word vectors.
# The value of cosine similarity ranges from -1 to 1:
# - A value of 1 means the two vectors are identical.
# - A value of 0 means they are orthogonal (no similarity).
# - A value of -1 means they are completely opposite.

# Cosine similarity formula:
# cos(θ) = (A · B) / (||A|| * ||B||)
# Where:
# - A · B is the dot product of vectors A and B.
# - ||A|| and ||B|| are the magnitudes (lengths) of the vectors A and B.

# Import Libraries
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load Data
movie_data = pd.read_csv("MovieData1/tmdb_5000_movies.csv")
credit_data = pd.read_csv("MovieData1/tmdb_5000_credits.csv")

# Merge Datasets on Movie Title
movie_data = movie_data.merge(credit_data, on='title')

# Select Relevant Columns
movie_data = movie_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop Rows with Missing Values
movie_data.dropna(inplace=True)

# Function to Extract Names from JSON-like Strings
def fetch_name(text):
    """
    Extracts names from a JSON-like string of dictionaries.
    """
    names = []
    try:
        data = ast.literal_eval(text)
        for item in data:
            names.append(item['name'])
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing JSON-like string: {e}")
    return names

# Apply to Extract Movie Genres and Keywords
movie_data['genres'] = movie_data['genres'].apply(fetch_name)
movie_data['keywords'] = movie_data['keywords'].apply(fetch_name)

# Function to Extract Top 3 Cast Members
def fetch_hero(text):
    """
    Extracts the top 3 cast members from a JSON-like string.
    """
    heroes = []
    try:
        data = ast.literal_eval(text)
        for i, item in enumerate(data):
            if i < 3:
                heroes.append(item['name'])
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing JSON-like string: {e}")
    return heroes

# Apply to Extract Top 3 Cast Members
movie_data['cast'] = movie_data['cast'].apply(fetch_hero)

# Function to Extract Directors
def fetch_director(text):
    """
    Extracts the director's name from a JSON-like string.
    """
    directors = []
    try:
        data = ast.literal_eval(text)
        for item in data:
            if item.get("job") == "Director":
                directors.append(item['name'])
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing JSON-like string: {e}")
    return directors

# Apply to Extract Directors
movie_data['crew'] = movie_data['crew'].apply(fetch_director)

# Convert Lists to Lowercase and Remove Spaces
def space_eraser(list_with_spacebar):
    """
    Removes spaces from elements in a list of strings.
    """
    return [i.replace(" ", "") for i in list_with_spacebar]

movie_data['genres'] = movie_data['genres'].apply(lambda x: [i.lower() for i in x]).apply(space_eraser)
movie_data['keywords'] = movie_data['keywords'].apply(lambda x: [i.lower() for i in x]).apply(space_eraser)
movie_data['cast'] = movie_data['cast'].apply(lambda x: [i.lower() for i in x]).apply(space_eraser)
movie_data['crew'] = movie_data['crew'].apply(lambda x: [i.lower() for i in x]).apply(space_eraser)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to Stem Words in Text
def fn_stem(text):
    """
    Applies stemming to a string of text.
    """
    return ' '.join([ps.stem(word) for word in text.split()])

# Apply Stemming to Overviews and Split into Lists
movie_data['overview'] = movie_data['overview'].apply(fn_stem).apply(lambda x: x.split())

# Combine Features into a Single 'tags' Column
movie_data['tags'] = movie_data['overview'] + movie_data['genres'] + movie_data['keywords'] + movie_data['cast'] + movie_data['crew']
df = movie_data.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Join Lists into a Single String for Vectorization
df['tags'] = df['tags'].apply(lambda x: ' '.join(x))

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Vectorize Tags Using TF-IDF
vector = tfidf.fit_transform(df['tags']).toarray()

# Compute Cosine Similarity Matrix
similarity = cosine_similarity(vector)

# Improved Recommendation Function
def recommend_movie(movie):
    """
    Recommends movies similar to the given movie using TF-IDF vectorization.
    """
    try:
        # Find index of the given movie
        movie_index = df[df['title'].str.lower() == movie.lower()].index[0]
        
        # Get similarity scores
        distance = similarity[movie_index]
        
        # Get indices of the top 5 similar movies
        movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Return recommended movie titles
        recommended_movies = [df.iloc[i[0]]['title'] for i in movie_list]
        return recommended_movies
    except IndexError:
        return ["Movie not found in the database. Please check the movie title and try again."]

# Save Model
with open('movie_dict.pkl', 'wb') as f:
    pickle.dump(df.to_dict(), f)
    
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

# Example Usage
# movie_name = input("Enter Movie Name: ")
# recommendations = recommend_movie(movie_name)
# print("Recommended Movies:")
# for movie in recommendations:
#     print(movie)
