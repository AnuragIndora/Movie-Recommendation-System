import pickle
import streamlit as st
import pandas as pd
import requests

# TMDB API key (direct use)
TMDB_API_KEY = "082f9428255940bcfe0010f2e7f5d5a4"

# Load movie data and similarity matrix
with open('movie_dict.pkl', 'rb') as file:
    movie_dict = pickle.load(file)

with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

movies = pd.DataFrame(movie_dict)


def fetch_poster(movie_id: int) -> str | None:
    """Fetch the poster URL for a movie from The Movie Database API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return None
    except Exception as e:
        st.error(f"Failed to fetch poster for movie ID {movie_id}: {e}")
        return None


def recommend_movie(movie_title: str):
    """Recommend movies similar to the selected movie."""
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        st.error("Movie not found in the database.")
        return []

    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]

    recommendations = []
    for i, _ in movie_list:
        movie_data = movies.iloc[i]
        poster_url = fetch_poster(movie_data.movie_id)
        recommendations.append((movie_data.title, poster_url))
    return recommendations


# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie from the dropdown",
                              movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie)

    selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
    selected_poster = fetch_poster(selected_movie_data.movie_id)

    # --- Row 1: "You selected" text ---
    st.subheader("You selected:")

    # --- Row 2: Poster of the selected movie ---
    if selected_poster:
        st.image(selected_poster, width=250)
    else:
        st.image("https://via.placeholder.com/250")

    # --- Row 3: "Recommended movies" text ---
    st.subheader("Recommended Movies:")

    # --- Row 4: Posters of 5 recommended movies ---
    if recommendations:
        cols = st.columns(5)
        for i, (title, poster) in enumerate(recommendations):
            with cols[i]:
                st.image(poster or "https://via.placeholder.com/150",
                         use_container_width=True)
                st.text(title)
    else:
        st.warning("No recommendations found.")
