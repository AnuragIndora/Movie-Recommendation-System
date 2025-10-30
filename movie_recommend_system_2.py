# This code builds a TF-IDF + Cosine Similarity movie recommender
# and saves compressed sparse matrices to keep file size small.

import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import ast
import pickle, gzip

# -------------------------------
# Load and merge datasets
# -------------------------------
movie_data = pd.read_csv("MovieData1/tmdb_5000_movies.csv")
credit_data = pd.read_csv("MovieData1/tmdb_5000_credits.csv")
movie_data = movie_data.merge(credit_data, on='title')

movie_data = movie_data[[
    'movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew'
]]
movie_data.dropna(inplace=True)


# -------------------------------
# Helper functions for cleaning
# -------------------------------
def fetch_name(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except Exception:
        return []


def fetch_hero(text):
    try:
        data = ast.literal_eval(text)
        return [i['name'] for i in data[:3]]
    except Exception:
        return []


def fetch_director(text):
    try:
        data = ast.literal_eval(text)
        return [i['name'] for i in data if i.get('job') == 'Director']
    except Exception:
        return []


def clean_list(lst):
    return [i.replace(" ", "").lower() for i in lst]


ps = PorterStemmer()


def fn_stem(text):
    return ' '.join([ps.stem(word) for word in text.split()])


# -------------------------------
# Apply transformations
# -------------------------------
movie_data['genres'] = movie_data['genres'].apply(fetch_name).apply(clean_list)
movie_data['keywords'] = movie_data['keywords'].apply(fetch_name).apply(
    clean_list)
movie_data['cast'] = movie_data['cast'].apply(fetch_hero).apply(clean_list)
movie_data['crew'] = movie_data['crew'].apply(fetch_director).apply(clean_list)
movie_data['overview'] = movie_data['overview'].apply(fn_stem).apply(
    lambda x: x.split())

movie_data['tags'] = (movie_data['overview'] + movie_data['genres'] +
                      movie_data['keywords'] + movie_data['cast'] +
                      movie_data['crew'])

df = movie_data.drop(
    columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
df['tags'] = df['tags'].apply(lambda x: ' '.join(x))

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
vector = tfidf.fit_transform(df['tags'])  # already sparse

# Compute cosine similarity as sparse matrix (saves space)
similarity_sparse = cosine_similarity(vector, dense_output=False)
similarity_sparse = sparse.csr_matrix(similarity_sparse, dtype='float32')

# -------------------------------
# Save compressed artifacts
# -------------------------------
# Save DataFrame dictionary
with gzip.open('movie_dict.pkl.gz', 'wb') as f:
    pickle.dump(df.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

# Save sparse similarity matrix
sparse.save_npz('similarity_sparse.npz', similarity_sparse)

print("Files saved: 'movie_dict.pkl.gz' and 'similarity_sparse.npz'")
print(f"Movies: {len(df)}, Matrix shape: {similarity_sparse.shape}")


# -------------------------------
# Recommendation function
# -------------------------------
def recommend_movie(movie):
    try:
        movie_index = df[df['title'].str.lower() == movie.lower()].index[0]
        distances = similarity_sparse[movie_index].toarray().ravel()
        indices = distances.argsort()[::-1][1:6]
        return [df.iloc[i]['title'] for i in indices]
    except IndexError:
        return ["Movie not found in the database."]
