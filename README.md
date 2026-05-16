# AI Movie Recommendation System

A premium content-based movie recommendation engine built with Python, Streamlit, and Machine Learning. The application utilizes natural language processing to analyze movie descriptions and provide real-time recommendations alongside an advanced filtering system and automated poster fetching.

## Live Demo

The application is fully deployed and can be accessed at:
https://ai-movie-recommender-bmgwuvakakdkbu5gcphhvz.streamlit.app

## Features

* Content-Based Recommendation Engine: Employs TF-IDF vectorization and cosine similarity to discover films semantically similar to a selected title.
* Real-Time Poster Generation: Integrates with the official TMDB API to pull high-resolution movie artwork dynamically.
* Multi-Criteria Filtering: Allows filtering by full language mapping, specialized genres, release year ranges, and minimum vote averages.
* Fallback Discovery Grid: Automatically populates a showcase of trending popular movies when no recommendation filter state is initialized.
* Tailored Dark UI: Features customized styling implemented through inline CSS injection for an optimized user experience.

## Technical Architecture and How It Works

1. Data Extraction and Preprocessing: Curates a dataset of 10,000 top TMDB titles, handles missing values, parses dates, and maps code keys to full language and genre matrices.
2. TF-IDF Vectorization: Text strings from the movie overviews are converted into numerical vectors using Term Frequency-Inverse Document Frequency.
3. Cosine Similarity Matrix: Computes linear kernel similarity values across all vector pairings to determine semantic closeness.
4. Adaptive Pipeline Filtering: Evaluates user parameters dynamically across data rows to omit non-matching results before delivering the top matching recommendations.

## Tech Stack

* Programming Language: Python
* User Interface Framework: Streamlit
* Mathematical and Data Processing: Pandas, AST
* Machine Learning and Natural Language Processing: Scikit-learn (TfidfVectorizer, linear_kernel)
* Networking and APIs: Requests, urllib3 (configured with HTTP adapter retries and adaptive backoff factors)

## Project Structure

* app.py: Handles the visual interface layout, session state tracking, and user option controls.
* recommender.py: Powers the underlying machine learning logic, data mapping functions, and TMDB API connectivity.
* TMDB_Top_10k_Movies_2026.csv: The dataset containing structured metadata profiles for 10,000 films.
* README.md: Technical documentation for the repository.

## Installation and Local Setup

### Prerequisites
Ensure you have a Python environment or an Anaconda distribution installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/MahaT-06/ai-movie-recommender.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-movie-recommender
   ```
3. Install the necessary build dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the web application server:
   ```bash
   streamlit run app.py
   ```

## API Configuration

The system includes a default fallback API key to authenticate TMDB requests. To supply custom environment configurations:
Set the environment variable via system configurations:
```bash
set TMDB_API_KEY=your_key_here
```
Alternatively, adjust the initialization parameter directly inside recommender.py on line 15.

## License

This project is licensed under the MIT License.