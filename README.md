# Movie Recommendation System

A content-based movie recommendation engine built with Python and Streamlit. Get personalized movie suggestions based on movies you like!

## Features

- Smart Recommendations - Uses TF-IDF vectorization and cosine similarity to find movies similar to your favorites
- Multiple Filters - Filter by language, genres, year range, and minimum rating
- Genre Support - Automatic genre detection from TMDB genre IDs
- Posters - Fetches movie posters from TMDB API
- Dark Theme - Clean, modern dark UI

## Requirements

streamlit>=1.0
pandas
scikit-learn
requests
urllib3

## Installation

pip install streamlit pandas scikit-learn requests urllib3

## Run the App

streamlit run app.py

The app will open at http://localhost:8501

## Project Structure

app.py - Streamlit frontend UI
recommender.py - Recommendation engine (TF-IDF + similarity)
TMDB_Top_10k_Movies_2026.csv - Movie dataset
README.md - This file

## How It Works

1. Data Loading - Loads 10,000 movies from CSV with metadata (title, overview, release date, language, genres, ratings)
2. TF-IDF Vectorization - Converts movie overviews into numerical vectors using Term Frequency-Inverse Document Frequency
3. Cosine Similarity - Computes similarity scores between all movie pairs
4. Filtering - Applies user filters (language, genres, year, rating) to refine results
5. Recommendations - Returns top N most similar movies to the selected title

## API Key

The TMDB API key is hardcoded as a fallback. To use your own:
In recommender.py, line 15: self.api_key = os.environ.get("TMDB_API_KEY", "your_key_here")
Or set environment variable: set TMDB_API_KEY=your_key_here

## Tech Stack

Python - Core language
Streamlit - Web UI framework
Pandas - Data handling
Scikit-learn - TF-IDF and cosine similarity
Requests - TMDB API calls
urllib3 - HTTP retry logic

## License

MIT License