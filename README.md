# Movie Recommendation System

This is a movie recommendation system built using **TF-IDF Vectorizer** and **Cosine Similarity**. The system provides movie recommendations based on the textual similarity of various features such as the movie's overview, genres, keywords, cast, and crew. It also fetches movie posters from **The Movie Database (TMDb) API** and displays them using a **Streamlit** web interface.

## Features

- Recommends movies based on their content similarity using TF-IDF Vectorization.
- Fetches movie posters dynamically using the TMDb API.
- Displays movie recommendations in a visually appealing format using Streamlit.
- Easy to use dropdown menu to select a movie and receive recommendations.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- Requests
- dotenv (optional, for secure API key management)

You can install the required Python libraries using the following command:

```bash
pip install streamlit pandas scikit-learn nltk requests python-dotenv
```

## Setup Instructions

1. Clone the repository or download the project files.

2. Ensure you have the **movie_dict.pkl** and **similarity.pkl** files in the project directory. These files contain the preprocessed movie data and similarity matrix, respectively.

3. To fetch movie posters, you will need an API key from [The Movie Database (TMDb)](https://www.themoviedb.org/). Sign up for an account and obtain your API key.

4. (Optional) For better security, store your API key in an `.env` file:

    - Create a `.env` file in the project directory:
      ```
      TMDB_API_KEY=your_actual_api_key_here
      ```
    - Install `python-dotenv` for loading environment variables:
      ```bash
      pip install python-dotenv
      ```

## How to Run the Project

1. Make sure you have the required dependencies installed.
   
2. Run the Streamlit app using the following command in your terminal:
   
   ```bash
   streamlit run app.py
   ```

3. The app should open automatically in your browser. If not, go to `http://localhost:8501` in your web browser.

4. Use the dropdown menu to select a movie, and click the **Recommend** button to view movie recommendations along with their posters.

## Project Structure

- **app.py**: The main Streamlit app file that loads the movie data, computes recommendations, and fetches movie posters.
- **movie_dict.pkl**: The preprocessed movie dataset stored as a dictionary.
- **similarity.pkl**: The similarity matrix computed using TF-IDF Vectorization.
- **.env**: (Optional) Environment file for storing the TMDb API key.
- **README.md**: This file, containing instructions for running the project.

## Functions

- **fetch_poster(movie_id: int) -> str**: Fetches the poster URL for a movie using the TMDb API.
- **recommended_movie(movie: str) -> list**: Returns a list of recommended movies and their poster URLs based on the selected movie.

## TMDb API Integration

This project uses The Movie Database (TMDb) API to fetch movie posters. Ensure you add your API key to the project either by hardcoding it in the `fetch_poster()` function or by storing it in an `.env` file.

### Example

```python
api_key = os.getenv('TMDB_API_KEY')  # Load from environment variable
```

Or hardcode it:

```python
api_key = 'your_actual_api_key_here'
```

## Example Usage

1. Run the app using Streamlit.
2. Select a movie from the dropdown.
3. Click the **Recommend** button to view the recommended movies and their posters.
