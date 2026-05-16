import streamlit as st
import pandas as pd
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_resource
def load_engine():
    return MovieRecommender('TMDB_Top_10k_Movies_2026.csv')

engine = load_engine()

@st.cache_data(ttl=3600)
def get_poster(movie_id):
    return engine.fetch_poster(movie_id)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 50%, #0f0f0f 100%);
        min-height: 100vh;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stSelectbox > div > div {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333;
        border-radius: 6px;
    }
    
    .stMultiSelect > div > div {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333;
    }
    
    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333;
    }
    
    .stButton button {
        background: #00d4aa !important;
        color: #0f0f0f !important;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background: #00e6b8 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Filters")
    
    selected_movie = st.selectbox("Select a movie:", engine.titles)
    
    st.markdown("**Language**")
    selected_language = st.selectbox("Language", ["All"] + list(engine.languages), label_visibility="collapsed")
    
    if engine.genres:
        st.markdown("**Genres**")
        selected_genres = st.multiselect("Genres", engine.genres, label_visibility="collapsed")
    else:
        selected_genres = []
    
    st.markdown("**Year Range**")
    year_col1, year_col2 = st.columns(2)
    with year_col1:
        from_year = st.text_input("From", "", key="from_year", placeholder="1900")
    with year_col2:
        to_year = st.text_input("To", "", key="to_year", placeholder="2026")
    
    st.markdown("**Minimum Rating**")
    min_rating = st.text_input("Rating", "", key="min_rating", placeholder="0-10")
    
    num_rec = st.slider("Results", 3, 12, 6)
    
    if st.button("Get Recommendations"):
        st.session_state.show_recs = True
        st.session_state.sel_movie = selected_movie
        st.session_state.sel_lang = None if selected_language == "All" else selected_language
        st.session_state.sel_genres = selected_genres if selected_genres else None
        st.session_state.sel_years = (from_year, to_year) if from_year and to_year else None
        st.session_state.sel_rating = min_rating if min_rating else None
        st.session_state.sel_num = num_rec

with col2:
    if 'show_recs' not in st.session_state:
        st.session_state.show_recs = False
    
    if st.session_state.show_recs:
        st.subheader(f"Because you liked \"{st.session_state.sel_movie}\"")
        
        with st.spinner("Finding similar movies..."):
            results = engine.get_recommendations(
                st.session_state.sel_movie,
                num_rec=st.session_state.sel_num,
                language=st.session_state.sel_lang,
                year_range=st.session_state.sel_years,
                min_rating=st.session_state.sel_rating,
                genres=st.session_state.sel_genres
            )
        
        if results is not None and not results.empty:
            cols = st.columns(3)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    poster = get_poster(row['ID'])
                    st.image(poster, use_container_width=True)
                    st.markdown(f"**{row['Title']}**")
                    
                    year = int(row['Year']) if pd.notna(row['Year']) else 'N/A'
                    st.caption(f"⭐ {row['Vote_Average']:.1f} | {year} | {row['Language_Full']}")
                    
                    with st.expander("Overview"):
                        st.write(row['Overview'])
                    
                    if row.get('Genres'):
                        st.caption(f"🎭 {', '.join(row['Genres'])}")
                    
                    st.markdown("---")
        else:
            st.warning("No matching movies found. Try changing your filters.")
    else:
        st.subheader("Popular Movies")
        
        cols = st.columns(4)
        popular = engine.df.nlargest(12, 'Popularity')
        
        for i, (_, row) in enumerate(popular.iterrows()):
            with cols[i % 4]:
                poster = get_poster(row['ID'])
                st.image(poster, use_container_width=True)
                st.markdown(f"**{row['Title']}**")
                
                year = int(row['Year']) if pd.notna(row['Year']) else 'N/A'
                st.caption(f"⭐ {row['Vote_Average']:.1f} | {year}")
                st.markdown("")

st.markdown("---")
st.caption("Powered by Python • TF-IDF Content-Based Filtering")