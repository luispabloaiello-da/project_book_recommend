# ==============================================
# 📚 Simple Content-Based Book Recommender (Streamlit)
# ==============================================
# This app uses:
# - books_merged_cleaned.csv (from previous notebooks)
# - TF-IDF (text_for_keywords)
# - Genre dummy variables
# - Cosine similarity
# - K-Means cluster labels
#
# We keep it:
# - Simple
# - No custom functions
# - Top-to-bottom logic (like in class notebooks)
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.sparse import hstack
from collections import Counter
import ast
import re

# -------------------------
# 1. LOAD DATA
# -------------------------

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.title("📚 Content-Based Book Recommender")
st.write("This app completes our pipeline: **Scraping → Cleaning → Clustering → Recommendation → Web App**.")

# Get the current file's directory and construct the path to the data file
current_dir = Path(__file__).parent
data_path = current_dir.parent / "data" / "books_merged_cleaned.csv"

# Check if file exists and provide helpful error message
if not data_path.exists():
    st.error(f"Dataset not found at: {data_path}")
    st.write("Expected file location:", str(data_path))
    st.write("Make sure the books_merged_cleaned.csv file exists in the project_book_recommend/data/ directory")
    st.stop()

# Load the cleaned dataset created earlier
df = pd.read_csv(data_path)

st.write("### Dataset loaded")
st.write("Shape:", df.shape)

# -------------------------
# 2. BASIC PREP (TITLE, AUTHOR, GENRES LIST)
# -------------------------

# Ensure title and author are clean and consistent with previous notebooks
df["title"] = df["title"].astype(str).str.strip()
df["author"] = df["author"].astype(str).str.strip()
df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)
df["author"] = df["author"].str.replace(r"\s+", " ", regex=True)

# Lowercase helper for search
df["title_lower"] = df["title"].str.lower()
df["author_lower"] = df["author"].str.lower()

# Handle genres_list, which might be saved as string in the CSV
if "genres_list" in df.columns:
    fixed_genres_list = []
    for val in df["genres_list"]:
        if isinstance(val, list):
            fixed_genres_list.append(val)
        elif isinstance(val, str):
            val = val.strip()
            # try to parse list-like strings: "['fantasy', 'young adult']"
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        fixed_genres_list.append([str(x).strip().lower() for x in parsed])
                    else:
                        fixed_genres_list.append([])
                except:
                    parts = [p.strip().lower() for p in val.split(",") if p.strip() != ""]
                    fixed_genres_list.append(parts)
            else:
                parts = [p.strip().lower() for p in val.split(",") if p.strip() != ""]
                fixed_genres_list.append(parts)
        else:
            fixed_genres_list.append([])
    df["genres_list"] = fixed_genres_list
else:
    df["genres_list"] = [[] for _ in range(len(df))]

# Optional: create a readable genres string
df["genres_clean"] = df["genres_list"].apply(lambda x: ", ".join(sorted(set(x))))

# -------------------------
# 3. GENRE FEATURES (ONE-HOT STYLE)
# -------------------------

genre_counter = Counter()
for glist in df["genres_list"]:
    for g in glist:
        if isinstance(g, str):
            g_clean = g.strip().lower()
            if g_clean != "" and len(g_clean) > 2:
                genre_counter[g_clean] += 1

# Keep top N genres (to keep matrix small)
top_n_genres = 20
top_genres = [g for g, c in genre_counter.most_common(top_n_genres)]

st.write("**Top genres used as features:**", top_genres)

# Create binary columns for these top genres
for g in top_genres:
    col_name = "genre_" + g.replace(" ", "_")
    df[col_name] = df["genres_list"].apply(
        lambda genres: 1 if isinstance(genres, list) and g in genres else 0
    )

genre_cols = [c for c in df.columns if c.startswith("genre_")]

# -------------------------
# 4. TEXT FEATURES WITH TF-IDF
# -------------------------

# Use text_for_keywords from previous notebook (title + genres cleaned)
if "text_for_keywords" not in df.columns:
    # fallback: simple version if missing
    df["text_for_keywords"] = (
        df["title"].fillna("") + " " + df["genres_clean"].fillna("")
    )

df["text_for_keywords"] = df["text_for_keywords"].fillna("").astype(str).str.lower()

max_features = 100  # small number for speed in Streamlit demo

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(df["text_for_keywords"])

# -------------------------
# 5. COMBINE FEATURES (GENRES + TEXT)
# -------------------------

X_genres = df[genre_cols].values
X = hstack([X_genres, tfidf_matrix])

# -------------------------
# 6. K-MEANS CLUSTERING (FOR EXPLANATION / DISPLAY)
# -------------------------

# We'll use a small k just to show in which cluster a book lands.
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

df["cluster_kmeans"] = cluster_labels

# -------------------------
# 7. COSINE SIMILARITY MATRIX (FOR RECOMMENDATIONS)
# -------------------------

cosine_sim = cosine_similarity(X, X)

# -------------------------
# 8. SIDEBAR: CONTROLS
# -------------------------

st.sidebar.header("🔍 Search & Filters")

search_mode = st.sidebar.selectbox(
    "Search by",
    ["Title", "Author", "Genre", "Keyword"]
)

query = st.sidebar.text_input("Type your search:")

top_n = st.sidebar.slider(
    "How many recommendations?",
    min_value=1,
    max_value=10,
    value=5
)

min_rating = 0.0
if "rating_0_5" in df.columns:
    min_rating = st.sidebar.slider(
        "Minimum rating (0-5)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )

genre_filter = st.sidebar.multiselect(
    "Filter recommendations by genres (optional)",
    options=sorted(top_genres)
)

st.sidebar.write("---")
st.sidebar.write("Tip: Try searching 'harry potter', 'hunger games', 'tolkien', etc.")

# -------------------------
# 9. FIND CANDIDATE BOOKS BASED ON SEARCH
# -------------------------

filtered_df = df.copy()

if query:
    q = query.lower().strip()

    if search_mode == "Title":
        mask = filtered_df["title_lower"].str.contains(q, na=False)

    elif search_mode == "Author":
        mask = filtered_df["author_lower"].str.contains(q, na=False)

    elif search_mode == "Genre":
        # Check if query appears inside genres_clean
        mask = filtered_df["genres_clean"].str.contains(q, na=False)

    elif search_mode == "Keyword":
        mask = filtered_df["text_for_keywords"].str.contains(q, na=False)

    filtered_df = filtered_df[mask]

# Show user the matching base books to choose from
st.write("### 1️⃣ Choose a book as starting point")

if query and len(filtered_df) > 0:
    # Build display labels
    options = []
    for i, row in filtered_df.iterrows():
        label = f"{row['title']} — {row['author']}"
        options.append((label, i))

    labels = [x[0] for x in options]
    selected_label = st.selectbox(
        "Select one of the matching books:",
        options=labels
    )

    # Find the corresponding index
    selected_index = None
    for label, idx in options:
        if label == selected_label:
            selected_index = idx
            break
else:
    st.write("Type something in the sidebar to search by title, author, genre, or keyword.")
    selected_index = None

# -------------------------
# 10. SHOW BASE BOOK INFO + CLUSTER
# -------------------------

if selected_index is not None:
    base_row = df.loc[selected_index]

    st.write("### 2️⃣ Selected book")
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"**Title:** {base_row['title']}")
        st.markdown(f"**Author:** {base_row['author']}")
        if "rating_0_5" in df.columns:
            st.markdown(f"**Rating (0-5):** {base_row['rating_0_5']:.2f}")
        if "num_ratings" in df.columns:
            st.markdown(f"**# Ratings:** {int(base_row['num_ratings'])}")
        st.markdown(f"**Cluster (K-Means):** {int(base_row['cluster_kmeans'])}")

    with col2:
        st.markdown("**Genres:**")
        st.write(base_row["genres_clean"] if pd.notna(base_row["genres_clean"]) else "No genres available")
        if "book_url" in df.columns and pd.notna(base_row.get("book_url", None)):
            st.markdown(f"[Open book page]({base_row['book_url']})")
        if "author_url" in df.columns and pd.notna(base_row.get("author_url", None)):
            st.markdown(f"[Open author's page]({base_row['author_url']})")

       # -------------------------
    # 11. COMPUTE RECOMMENDATIONS (WITH SIMILARITY SCORE)
    # -------------------------

    st.write("### 3️⃣ Recommended similar books")

    # Get similarity scores for this book
    sim_scores = list(enumerate(cosine_sim[selected_index]))

    # Sort by similarity, highest first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Collect recommendations as (idx, score)
    rec_items = []

    for idx, score in sim_scores:
        if idx == selected_index:
            continue  # skip the selected book itself

        # Apply minimum rating filter (if column exists)
        if "rating_0_5" in df.columns:
            if pd.isna(df.loc[idx, "rating_0_5"]) or df.loc[idx, "rating_0_5"] < min_rating:
                continue

        # Apply genre filter (if any selected)
        if len(genre_filter) > 0:
            book_genres = df.loc[idx, "genres_list"]
            if not isinstance(book_genres, list):
                continue
            # require at least one selected genre to be present
            if not any(g in book_genres for g in genre_filter):
                continue

        rec_items.append((idx, score))

        if len(rec_items) >= top_n:
            break

    if len(rec_items) == 0:
        st.write("No recommendations found with the current filters. Try lowering minimum rating or clearing genre filter.")
    else:
        # Columns to display
        rec_cols = ["title", "author", "genres_clean"]
        if "rating_0_5" in df.columns:
            rec_cols.append("rating_0_5")
        if "num_ratings" in df.columns:
            rec_cols.append("num_ratings")
        if "cluster_kmeans" in df.columns:
            rec_cols.append("cluster_kmeans")

        # Build a small table including similarity_score
        rows = []
        for idx, score in rec_items:
            row = df.loc[idx, rec_cols].copy()
            row["similarity_score"] = round(float(score), 3)
            rows.append(row)

        rec_df = pd.DataFrame(rows).reset_index(drop=True)

        st.dataframe(rec_df)

# End of app
