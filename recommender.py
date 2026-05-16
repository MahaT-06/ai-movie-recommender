import os
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class MovieRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        self.compute_similarity()
        self.api_key = os.environ.get("TMDB_API_KEY", "025c4b2b640bc9337cb84cbadc319980")
        
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def prepare_data(self):
        # Handle missing values
        self.df['Overview'] = self.df['Overview'].fillna('')

        # 🔹 Extract Year from Release_Date
        self.df['Release_Date'] = pd.to_datetime(self.df['Release_Date'], errors='coerce', dayfirst=True)
        self.df['Year'] = self.df['Release_Date'].dt.year

        # 🔹 Language mapping
        self.lang_map = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'ko': 'Korean',
            'ja': 'Japanese',
            'zh': 'Chinese',
            'it': 'Italian',
            'ru': 'Russian'
        }

        self.genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 10770: 'Western'
        }

        if 'Genre_IDs' in self.df.columns:
            def parse_genres(genre_str):
                try:
                    import ast
                    ids = ast.literal_eval(genre_str)
                    return [self.genre_map.get(i, 'Unknown') for i in ids]
                except:
                    return []
            self.df['Genres'] = self.df['Genre_IDs'].apply(parse_genres)
            all_genres = set()
            for g_list in self.df['Genres']:
                all_genres.update(g_list)
            self.genres = sorted([g for g in all_genres if g])
        else:
            self.genres = []

        if 'Original_Language' in self.df.columns:
            self.df['Language_Full'] = self.df['Original_Language'].map(self.lang_map).fillna(self.df['Original_Language'])
            self.languages = sorted(self.df['Language_Full'].dropna().unique())
        else:
            self.df['Language_Full'] = 'Unknown'
            self.languages = ['Unknown']

        self.titles = self.df['Title'].values

    def compute_similarity(self):
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.df['Overview'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, title, num_rec=6, language=None, year_range=None, min_rating=None, genres=None):
        try:
            idx = self.df[self.df['Title'] == title].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            sim_scores = sim_scores[1:]  # remove itself

            movie_indices = []

            # 🔹 SAFE conversion for typed inputs
            start_year, end_year = None, None
            if year_range is not None:
                try:
                    start_year = int(year_range[0])
                    end_year = int(year_range[1])
                except:
                    start_year, end_year = None, None

            if min_rating is not None:
                try:
                    min_rating = float(min_rating)
                except:
                    min_rating = None

            for i, score in sim_scores:
                movie = self.df.iloc[i]

                # 🔹 Language filter
                if language is not None and movie['Language_Full'] != language:
                    continue

                # 🔹 Year filter
                if start_year is not None and end_year is not None:
                    if pd.isna(movie['Year']) or not (start_year <= movie['Year'] <= end_year):
                        continue

                # 🔹 Rating filter
                if min_rating is not None:
                    if pd.isna(movie['Vote_Average']) or movie['Vote_Average'] < min_rating:
                        continue

                # 🔹 Genre filter
                if genres:
                    movie_genres = movie.get('Genres', [])
                    if not any(g in movie_genres for g in genres):
                        continue

                movie_indices.append(i)

                if len(movie_indices) >= num_rec:
                    break

            return self.df.iloc[movie_indices]

        except Exception:
            return None

    def fetch_poster(self, movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={self.api_key}&language=en-US"
        try:
            response = self.session.get(url, timeout=10)
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        except Exception:
            pass

        return "https://via.placeholder.com/500x750?text=No+Poster+Found"