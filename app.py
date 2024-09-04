import pickle
import streamlit as st
import pandas as pd
import requests
import os

# Load movie data and similarity matrix from pickle files
with open('movie_dict.pkl', 'rb') as file:
    movie_dict = pickle.load(file)

with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

# Convert the loaded dictionary to a DataFrame
movies = pd.DataFrame(movie_dict)

def fetch_poster(movie_id: int) -> str:
    """
    Fetch the poster URL for a movie from The Movie Database API.

    Args:
        movie_id (int): The ID of the movie to fetch the poster for.

    Returns:
        str: The URL of the movie poster.

    Raises:
        requests.RequestException: If the API request fails.
        KeyError: If the API response does not contain a poster path.
    """
    api_key = '20c72665a524df7b2f19090233398e4a'  # Use environment variable for API key
    if not api_key:
        raise ValueError("TMDB API key not set")

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=20c72665a524df7b2f19090233398e4a&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch poster: {e}")
        return None

    try:
        data = response.json()
        poster_path = data['poster_path']
    except KeyError:
        st.error("Poster path not found in API response")
        return None

    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

def recommended_movie(movie):
    """
    Recommend movies similar to the selected movie.

    Args:
        movie (str): The title of the movie to find recommendations for.

    Returns:
        list: A list of recommended movie titles.
    """
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the database.")
        return []

    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        movie_title = movies.iloc[i[0]].title
        movie_id = movies.iloc[i[0]].movie_id
        poster_url = fetch_poster(movie_id)
        recommended_movies.append((movie_title, poster_url))
    
    return recommended_movies

# Streamlit app
st.title('Movie Recommendation System')

# Create a dropdown menu with movie titles
selected_movie = st.selectbox("Select a movie from the dropdown", movies['title'].values)

# # Display recommendations when the button is clicked
# if st.button('Recommend'):
#     recommendations = recommended_movie(selected_movie)
    
#     if recommendations:
#         col1, col2, col3, col4, col5 = st.columns(5)
#         cols = [col1, col2, col3, col4, col5]
        
#         for i, (movie_title, poster_url) in enumerate(recommendations):
#             with cols[i]:
#                 st.text(movie_title)
#                 if poster_url:
#                     st.image(poster_url, use_column_width=True)
#                 else:
#                     st.image('https://via.placeholder.com/150', use_column_width=True)
#     else:
#         st.write("No recommendations available.")

# Display recommendations when the button is clicked
if st.button('Recommend'):
    recommendations = recommended_movie(selected_movie)
    
    # Show the selected movie
    selected_movie_data = movies[movies['title'] == selected_movie]
    selected_movie_id = selected_movie_data.iloc[0]['movie_id']
    selected_movie_poster = fetch_poster(selected_movie_id)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    cols = [col1, col2, col3, col4, col5, col6]
    
    # Display the selected movie
    with cols[0]:
        st.text(selected_movie)
        if selected_movie_poster:
            st.image(selected_movie_poster, use_column_width=True)
        else:
            st.image('https://via.placeholder.com/150', use_column_width=True)
    
    # Display recommended movies
    if recommendations:
        for i, (movie_title, poster_url) in enumerate(recommendations):
            with cols[i+1]:
                st.text(movie_title)
                if poster_url:
                    st.image(poster_url, use_column_width=True)
                else:
                    st.image('https://via.placeholder.com/150', use_column_width=True)
    else:
        st.write("No recommendations available.")