
import streamlit as st
from streamlit_image_select import image_select
import pandas as pd
import numpy as np
import ast
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dotenv import load_dotenv
import os
import requests
from concurrent.futures import ThreadPoolExecutor


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TMDB_API_KEY = st.secrets.get(
    "API_KEY", os.getenv("API_KEY", "")
)



# page layout settings...
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0b0b12 0%, #15101c 55%, #09090d 100%);
    color: #f5f5f7;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff !important;
    letter-spacing: -0.5px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #e50914, #b20710);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff2530, #d9000b);
    color: white;
    border: none;
    transform: translateY(-2px);
}

/* Movie search box */
.stSelectbox > div > div {
    border-radius: 10px;
}

/* Subtle separators */
hr {
    border-color: #30232c;
}

.hero {
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1.5rem;
}

.hero-badge {
    display: inline-block;
    color: #ff5964;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 0.8rem;
}

.hero-tagline {
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: #f2dce0 !important;
    margin: 0.5rem 0 !important;
}

/* Search section */
.stSelectbox label {
    color: #e8dfe9 !important;
    font-weight: 600;
    font-size: 1rem;
}

.stSelectbox > div > div {
    background-color: #191720;
    border: 1px solid #39303f;
    border-radius: 12px;
    min-height: 48px;
}

.stSelectbox > div > div:focus-within {
    border-color: #e50914;
}

/* Add breathing room around the search button */
.stButton {
    margin-top: 0.5rem;
}


.brand-title {
    display: block !important;
    font-size: 64px !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    letter-spacing: -2px !important;
    margin: 16px 0 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Recommendation card polish */
div[data-testid="column"] {
    padding: 10px;
    border: 1px solid #302733;
    border-radius: 14px;
    background: rgba(25, 23, 32, 0.65);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

div[data-testid="column"]:hover {
    transform: translateY(-4px);
    border-color: #e50914;
}

div[data-testid="column"] img {
    border-radius: 10px;
}

/* Recommendation section heading */
.recommendation-title {
    font-size: 1.8rem;
    font-weight: 750;
    color: #ffffff;
    border-left: 4px solid #e50914;
    padding-left: 14px;
    margin: 2rem 0 1.2rem 0;
}

</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="hero">
    <div class="hero-badge">YOUR PERSONAL MOVIE DISCOVERY</div>
    <div class="brand-title">🎬 CineMatch</div>
    <p class="hero-tagline">Discover films that match your taste.</p>
</div>
""", unsafe_allow_html=True)


# Load and prepare the datasets
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_top_10k_movies_2026.csv")
    credits = pd.read_csv("tmdb_full_credits.csv")
    metadata=pd.read_csv("tmdb_5000_movies.csv")
    posters=pd.read_csv("tmdb_poster_paths.csv")
    return df, credits,metadata,posters
df, credits,metadata,posters = load_data()


# Match each movie with its saved poster path
posters = posters.drop_duplicates(subset="id", keep="last")

df = df.merge(
    posters[["id", "poster_path"]],
    on="id",
    how="left"
)
df["poster_path"] = df["poster_path"].fillna("")


# Merge full TMDB credits into the movie dataset
credits = credits.drop_duplicates(subset="movie_id", keep="last")
df = df.merge(
    credits[["movie_id", "cast_names", "director"]],
    left_on="id",
    right_on="movie_id",
    how="left"
)
df["cast_names"] = df["cast_names"].fillna("")
df["director"] = df["director"].fillna("")

# Merge movie keywords and franchise information
metadata = metadata.drop_duplicates(subset="id", keep="last")

df = df.merge(
    metadata[["id", "keywords"]],
    on="id",
    how="left"
)
df["keywords"] = df["keywords"].fillna("")
df["keywords"] = df["keywords"].fillna("")



df.drop(columns=["movie_id"], inplace=True)


# Prepare movie metadata and recommendation features
genre_map = {
    12: "Adventure", 14: "Fantasy", 16: "Animation",
    18: "Drama", 27: "Horror", 28: "Action",
    35: "Comedy", 36: "History", 37: "Western",
    53: "Thriller", 80: "Crime", 878: "Science Fiction",
    9648: "Mystery", 10402: "Music", 10749: "Romance",
    10751: "Family", 10752: "War", 10770: "TV Movie"
}

@st.cache_data
def prepare_data(df,credits):
    df = df.copy()

    def parse_genres(value):
        try:
            ids = ast.literal_eval(value)
            return " ".join(genre_map.get(i, "") for i in ids)
        except (ValueError, TypeError):
            return ""


    df["genre"] = df["genre_ids"].apply(parse_genres)

    def parse_keywords(value):
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
            return " ".join(
                item["name"].lower().replace(" ", "")
                for item in items
                if isinstance(item, dict) and item.get("name")
            )
        except (ValueError, SyntaxError, TypeError):
            return ""

    df["keywords"] = df["keywords"].apply(parse_keywords)
    
    df["cast_names"] = df["cast_names"].fillna("")
    df["director"] = df["director"].fillna("")
    df["overview"] = df["overview"].fillna("")

    return df

df = prepare_data(df, credits)


# Build separate TF-IDF matrices
@st.cache_resource
def build_matrices(genres, overviews, keywords, casts, directors):
    genre_tv = CountVectorizer(binary=True,token_pattern=r"(?u)\b\w+\b")
    overview_tv = TfidfVectorizer(stop_words="english",ngram_range=(1, 2),max_features=100000,min_df=2,
    max_df=0.85,sublinear_tf=True)
    cast_tv = TfidfVectorizer()
    director_tv = TfidfVectorizer()
    keyword_tv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    genre_matrix = genre_tv.fit_transform(genres)
    overview_matrix = overview_tv.fit_transform(overviews)
    cast_matrix = cast_tv.fit_transform(casts)
    director_matrix = director_tv.fit_transform(directors)
    keyword_matrix = keyword_tv.fit_transform(keywords)

    return genre_matrix, overview_matrix, keyword_matrix,cast_matrix, director_matrix

genre_matrix, overview_matrix, keyword_matrix, cast_matrix, director_matrix = build_matrices(
    df["genre"].tolist(),
    df["overview"].tolist(),
    df["keywords"].tolist(),
    df["cast_names"].tolist(),
    df["director"].tolist()
)


@st.cache_data
def recommend_movies(movie_name, top_n=5):
    matches = df.index[
        df["title"].fillna("").str.lower() == str(movie_name).lower()
    ]

    if len(matches) == 0:
        return []

    idx = matches[0]

    genre_sim = (genre_matrix[idx] @ genre_matrix.T).toarray().ravel()
    overview_sim = (overview_matrix[idx] @ overview_matrix.T).toarray().ravel()
    cast_sim = (cast_matrix[idx] @ cast_matrix.T).toarray().ravel()
    director_sim = (director_matrix[idx] @ director_matrix.T).toarray().ravel()
    keyword_sim = (keyword_matrix[idx] @ keyword_matrix.T).toarray().ravel()

    # Check which features are available for the selected movie
    has_genre = bool(df.iloc[idx]["genre"].strip())
    has_overview = bool(df.iloc[idx]["overview"].strip())
    has_keywords = bool(df.iloc[idx]["keywords"].strip())
    has_cast = bool(df.iloc[idx]["cast_names"].strip())
    has_director = bool(df.iloc[idx]["director"].strip())

    # Base weights
    weights = {
        "genre": 0.20,
        "overview": 0.30,
        "keywords": 0.05,
        "cast": 0.35,
        "director": 0.10
    }

    available = {
        "genre": has_genre,
        "overview": has_overview,
        "keywords": has_keywords,
        "cast": has_cast,
        "director": has_director
    }

    # Redistribute weights proportionally
    total_weight = sum(
        weights[name] for name in weights if available[name]
    )

    if total_weight == 0:
        return []

    weights = {
        name: weight / total_weight if available[name] else 0
        for name, weight in weights.items()
        }

    # Calculate final recommendation scores
    scores = (
    weights["genre"] * genre_sim
    + weights["overview"] * overview_sim
    + weights["keywords"] * keyword_sim
    + weights["cast"] * cast_sim
    + weights["director"] * director_sim
    )

    scores[idx] = -1
    top_indices = np.argpartition(scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
    {
        "title": df.iloc[i]["title"],
        "id": int(df.iloc[i]["id"]),
        "score": round(float(scores[i]), 4)
    }
    for i in top_indices
]

def test_recommendations(movie_name):
    recommendations = recommend_movies(movie_name)

    print(f"\nRecommendations for: {movie_name}")

    for number, movie in enumerate(recommendations, start=1):
        print(
            f"{number}. {movie['title']} "
            f"| Similarity score: {movie['score']}"
        )


# fetch movie posters...
@st.cache_data(show_spinner=False)
def get_movie_poster(movie_id):
    movie = df.loc[df["id"] == int(movie_id), "poster_path"]

    if movie.empty or not movie.iloc[0]:
        return None

    return f"https://image.tmdb.org/t/p/w500{movie.iloc[0]}"



# Streamlit interface
movie_titles = sorted(df["title"].dropna().unique())

selected_movie = st.selectbox(
    "Choose a movie",
    movie_titles,
    index=None,
    placeholder="Search for a movie...",
)

if selected_movie:
    selected_id = int(
        df.loc[df["title"] == selected_movie, "id"].iloc[0]
    )
    selected_poster = get_movie_poster(selected_id)
    st.markdown("##### 🎬 Selected movie")

    if selected_poster:
        st.image(selected_poster, width=200)
    else:
        st.caption("Poster unavailable for this movie.")



import time

if st.button("Find Recommendations"):
    start = time.perf_counter()

    recommendations = recommend_movies(selected_movie)

    if recommendations:
        st.markdown(
    f'<h2 class="recommendation-title">🎯 Movies similar to: {selected_movie}</h2>',
    unsafe_allow_html=True
)

        
        columns = st.columns(5)

        movie_ids = [
            int(df.loc[df["title"] == movie["title"], "id"].iloc[0])
            for movie in recommendations
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            poster_urls = list(executor.map(get_movie_poster, movie_ids))

        for number, (column, movie, poster_url) in enumerate(
            zip(columns, recommendations, poster_urls), start=1
        ):
            with column:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.write("Poster unavailable")

                st.markdown(f"**{number}. {movie['title']}**")


    else:
        st.warning("No recommendations found for this movie.")


