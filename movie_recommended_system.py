# Import Libraries
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


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

# Apply to Extract Movie Genres
movie_data['genres'] = movie_data['genres'].apply(fetch_name)

# Apply to Extract Movie Keywords
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

# Convert Lists to Lowercase
movie_data['genres'] = movie_data['genres'].apply(lambda x: [i.lower() for i in x])
movie_data['keywords'] = movie_data['keywords'].apply(lambda x: [i.lower() for i in x])
movie_data['cast'] = movie_data['cast'].apply(lambda x: [i.lower() for i in x])
movie_data['crew'] = movie_data['crew'].apply(lambda x: [i.lower() for i in x])

# Function to Remove Spaces from Strings
def space_eraser(list_with_spacebar):
    """
    Removes spaces from elements in a list of strings.
    """
    return [i.replace(" ", "") for i in list_with_spacebar]

# Apply SpaceEraser to All Relevant Columns
movie_data['genres'] = movie_data['genres'].apply(space_eraser)
movie_data['keywords'] = movie_data['keywords'].apply(space_eraser)
movie_data['cast'] = movie_data['cast'].apply(space_eraser)
movie_data['crew'] = movie_data['crew'].apply(space_eraser)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to Stem Words in Text
def fn_stem(text):
    """
    Applies stemming to a string of text.
    """
    return ' '.join([ps.stem(word) for word in text.split()])

# Apply Stemming to Overviews and Split into Lists
movie_data['overview'] = movie_data['overview'].apply(fn_stem)
movie_data['overview'] = movie_data['overview'].apply(lambda x: x.split())

# Combine Features into a Single 'tags' Column
movie_data['tags'] = movie_data['overview'] + movie_data['genres'] + movie_data['keywords'] + movie_data['cast'] + movie_data['crew']
df = movie_data.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Join Lists into a Single String for Vectorization
df['tags'] = df['tags'].apply(lambda x: ' '.join(x))

# Vectorize Tags Using CountVectorizer
cv = CountVectorizer(stop_words='english', max_features=5000)
vector = cv.fit_transform(df['tags']).toarray()

# Compute Cosine Similarity Matrix
similarity = cosine_similarity(vector)

# Function to Recommend Movies
def recommend_movie(movie):
    """
    Recommends movies similar to the given movie.
    """
    try:
        # Find index of the given movie
        movie_index = df[df['title'] == movie].index[0]
        
        # Get similarity scores
        distance = similarity[movie_index]
        
        # Get indices of the top 5 similar movies
        movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        
        # Print recommended movie titles
        if movie_list:
            print("Recommended Movies:")
            for i in movie_list:
                print(df.iloc[i[0]]['title'])
        else:
            print("No similar movies found.")
    except IndexError:
        print("Movie not found in the database.")

# Get User Input for Movie Recommendation
movie_name = input("Enter Movie Name: ")
recommend_movie(movie_name)
