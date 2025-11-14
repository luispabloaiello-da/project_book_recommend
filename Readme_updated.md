# 📚 Book Recommendation Project — Web Scraping & Content-Based Filtering

> **Context**
> 9-week Data Analytics Bootcamp · Project
> Focus: Web scraping, REST APIs, data cleaning, unsupervised learning,
and a simple content-based recommender.

This project walks through a full end-to-end pipeline:

**Scraping → Cleaning → Feature Engineering → Clustering →
Recommendation → Streamlit Web App**

## Slides

- [Book Recommendation System](https://www.canva.com/design/DAG4Xyd8ELI/92lvv1QURGM2MgTsUDFjRw/edit?utm_content=DAG4Xyd8ELI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 1) Project Overview**

This project builds a small **Book Recommendation dataset** by combining:

1. ~**600 books** from **Goodreads – Best Books Ever** (web scraping).
2. ~**600 books** from **Open Library – Trending** (Search API + work pages).

**We collect:**

-   Title, Author & Author URL

-   Book URL

-   Average rating & Number of ratings

-   Score / Votes (Goodreads)

-   Trending score (Open Library)

-   Genres / Subjects

-   Publication years (when available)

Then we:

-   Clean & normalize

-   Engineer features (TF-IDF keywords + genre dummies)

-   Explore the data (EDA)

-   Use **unsupervised learning** (similarity + clustering) for a
    **content-based recommender**

-   Ship a simple **Streamlit** app

**2) Repository Structure**

```bash
project_book_recommend/
├── data/
│   ├── goodreads_best_books_600.csv
│   ├── openlibrary_trending_600.csv
│   └── books_merged_cleaned.csv
├── figures/
│   ├── book_clusters_k_means_genres_text_visualized_pca.png
│   ├── eda_rating_distribution.png
│   ├── eda_top_authors.png
│   ├── eda_top_genres.png
│   ├── eda_rating_vs_num_ratings.png
│   ├── eda_wordcloud_titles_genres.png
│   └── hierarchical_clustering_dendrogram.png
├── my_stremlit_app/
│   └── app.py
├── notebooks/
│   ├── books_data_cleaning_eda.ipynb
│   ├── content_based_recommender.ipynb
│   ├── scrape_goodreads_best_books_ever.ipynb
│   └── scrape_openLibrary_trending_api.ipynb
├── slides/
│   └── 
└── .gitattributes
└── README.md
```

**Key notebooks**

-   scrape_goodreads_best_books_ever.ipynb --- scrape Goodreads list
    pages + book pages (genres, published year).

-   scrape_openLibrary_trending_api.ipynb --- Open Library Search API +
    book page Subjects as genres.

-   books_data_cleaning_eda.ipynb --- concatenate, clean, normalize,
    TF-IDF, and EDA.

-   content_based_recommender.ipynb --- build features, cosine
    similarity, K-Means clusters, recommendations.

**3) Data Sources**

**Goodreads --- Best Books Ever**

-   List: https://www.goodreads.com/list/show/1.Best_Books_Ever

-   Techniques: requests, BeautifulSoup, polite pauses, absolute URLs

-   From list page: Rank, Title, Book URL, Author & Author URL, Avg
    rating, #ratings, Score, Votes

-   From book page: **Genres** (new button layout), **First published
    year** (regex from "First published \...")

**Open Library --- Trending via API**

-   Endpoint: https://openlibrary.org/search.json

-   Example query (simplified):\
    trending_score_hourly_sum:\[1 TO \*\]
    -subject:\"content_warning:cover\" language:eng

-   From API: title, author_name, author_key→author URL, key→work URL,
    first_publish_year, ratings_average, ratings_count,
    trending_score_hourly_sum

-   From work page: **Subjects** under \<h3\>Subjects\</h3\> as genres

**4) Setup**

Install requirements:

pip install pandas numpy requests beautifulsoup4 matplotlib scikit-learn
wordcloud streamlit

Run notebooks in order:

1.  scrape_goodreads_best_books_ever.ipynb

2.  scrape_openLibrary_trending_api.ipynb

3.  books_data_cleaning_eda.ipynb

4.  content_based_recommender.ipynb

Run the app:

cd my_streamlit_app

streamlit run app.py

**5) Data Cleaning & Preprocessing (notebook:
books_data_cleaning_eda.ipynb)**

**5.1 Titles & Authors**

-   Convert to string; strip leading/trailing spaces.

-   Collapse internal multiple spaces: \"stephen king\" → \"stephen
    king\".

-   Remove non-letter/number chars when needed; keep parentheses.

-   Create title_clean, author_clean (lowercased).

-   **Drop rows** where title_clean or author_clean is NaN/empty.

**5.2 Genres --- Normalization**

Goals:

-   Keep only **letters** (remove numbers/symbols)

-   Consistent separators

-   No duplicates per row

Process (cell-by-cell, no functions):

1.  Ensure genres exists → \"\" if missing.

2.  Lowercase.

3.  Replace / & = ; \| with commas; remove brackets/quotes/parentheses.

4.  Keep only letters, commas, spaces → regex.

5.  Collapse spaces; split by comma.

6.  Filter empty/short junk tokens (len ≤ 2), drop known junk prefixes
    (nyt, pz, loc, ol).

7.  Remove duplicates **within the row** (keep first).

Outputs:

-   genres_list --- list of clean genres per book

-   genres_clean --- joined string for display

**5.3 Ratings**

-   avg_rating → numeric

-   Standardize to rating_0\_5 (divide by 2 if in 0--10 range; clip to
    \[0,5\]).

-   num_ratings → numeric; fill missing with 0 (int).

**5.4 Text for TF-IDF**

-   text_for_keywords = (title + \" \" + genres) (lowercase)

-   Keep it simple and visible (no functions).

**6) TF-IDF Keyword Extraction**

-   TfidfVectorizer(max_features=N, stop_words=\"english\")

-   Fit on text_for_keywords; get tfidf_matrix and feature_names.

-   For each book: sort TF-IDF row, take top \~5 tokens → top_keywords.

-   Unsupervised representation of content (no labels).

**7) EDA (notebook: books_data_cleaning_eda.ipynb)**

-   Rating distribution (histogram, 0--5).

-   Top authors by count.

-   Genre distribution from genres_list.

-   Scatter: log(#ratings) vs rating.

-   Correlations: rating_0\_5, num_ratings, score, votes,
    trending_score_hourly_sum.

-   Optional word cloud (titles + genres).

**8) Content-Based Recommender (notebook:
content_based_recommender.ipynb)**

**8.1 Features**

-   **Genres** → top-N genre dummies (genre_fantasy, genre_romance,
    ...).

-   **Text** → TF-IDF on text_for_keywords (e.g., 100 features).

-   **Combined matrix**: X = \[genres_dummies \| tfidf_matrix\] (use
    hstack).

**8.2 Similarity**

-   **Cosine similarity** on X → cosine_sim\[i, j\] ∈ \[0,1\].

-   For a selected book:

    1.  Find row index

    2.  Sort others by similarity

    3.  Return top-N similar books

-   Fully **unsupervised** (no labels).

**8.3 Clustering & Visualization**

-   **K-Means** on X with small k (e.g., 5) → cluster_kmeans per book.

-   **PCA (2D)** to plot book clusters in a scatterplot.

-   **Hierarchical Dendrogram** (optional): tree of merges; choose cut
    height to form clusters.

**9) Streamlit Web App (folder: my_streamlit_app/)**

app.py implements a simple interface:

-   **Search**: Title / Author / Genre / Keyword

-   **Controls**: \# of recommendations (1--10), minimum rating,
    optional genre filter

-   **Display**:

    -   Selected book info (title, author, rating, genres, URL)

    -   **Cluster (K-Means)** the book belongs to

    -   **Top similar books** with **similarity_score** (cosine)

-   Internally (top-to-bottom cells):

    1.  Load books_merged_cleaned.csv

    2.  Rebuild features (genre dummies + TF-IDF)

    3.  Fit small K-Means for display

    4.  Compute cosine similarity

    5.  Filter + sort recommendations

Run:

cd my_streamlit_app

streamlit run app.py

**10) How This Aligns with the Bootcamp (ML-Unsupervised)**

-   **Same mindset**: no labels → rely on **distance/similarity**.

-   **Feature engineering first**: one-hot/multi-hot for categories;
    TF-IDF for text.

-   **Clustering**: K-Means to group similar books; PCA to visualize.

-   **Dendrogram**: alternative view of structure (merge history).

-   **Recommender**: "nearest neighbors" concept using cosine
    similarity.

Clustering groups *all* items; the recommender finds the *closest*
neighbors to one item. Same geometry, different usage.

**11) Notes & Tips**

-   Be polite with scraping (headers, delays).

-   If Open Library or Goodreads layout changes, update selectors
    carefully.

-   Keep max_features (TF-IDF) small for transparency and speed.

-   Always sanity-check genres_list (lists, not strings) before counting
    or dummies.

**12) Author**

-   **Luis Pablo Aiello** --- Data Analytics Bootcamp Student

---

## License
This project is intended for **educational use within the bootcamp cohort**.
The dataset is built from publicly available book information (Goodreads and Open Library) and is used solely for learning, exploration, and demonstration purposes.
