# 📚 Book Recommendation Project — Web Scraping & Content-Based Filtering

> **Bootcamp context**  
> 9-week Data Analytics Bootcamp · Project 2  
> Focus: Web scraping, APIs, data cleaning, basic NLP, and a simple content-based recommender using real-world book data.

---

## 1. Overview

This project builds a small **Book Recommendation dataset** by combining:

1. ~**600 books** from **Goodreads – Best Books Ever** (web scraping).
2. ~**600 books** from **Open Library – Trending** (Search API + work pages).

From these sources we:

- Collect **titles, authors, URLs, ratings, votes, genres, subjects, and years**.
- Clean and normalize the data into a single dataset.
- Engineer features like **genres, keyword representations, and TF-IDF-based keywords**.
- Perform **Exploratory Data Analysis (EDA)**.
- Prepare the ground for a **content-based recommendation system**.

---

## 2. What You’ll Find

- **Web Scraping (Goodreads)**  
  Using `requests + BeautifulSoup` to:
  - Scrape list pages from:  
    `https://www.goodreads.com/list/show/1.Best_Books_Ever`
  - Extract (if available): rank, title, author, author URL, average rating, number of ratings, score, votes, book URL, genres, and first published year.

- **API Integration (Open Library)**  
  Using the **Open Library Search API**:
  - Query:  
    `trending_score_hourly_sum:[1 TO *] -subject:"content_warning:cover" language:eng -subject:"content_warning:cover" -subject:"content_warning:cover"`
  - Extract from API: title, authors, ratings, rating count, trending scores, work keys.
  - Enrich with **book page scraping** to pull **Subjects** as genres.

- **Data Cleaning & Preprocessing**
  - Remove duplicates and inconsistent records.
  - Normalize text (titles, authors, genres).
  - Clean noisy genres (symbols, codes, brackets, etc.).
  - Standardize ratings to a **0–5 scale**.
  - Prepare a combined dataset for modeling and recommendations.

- **Keyword & Feature Engineering**
  - Build a simple text field from titles and genres.
  - Use **TF-IDF** to generate **top keywords per book**.

- **EDA**
  - Top-rated books.
  - Most frequent authors.
  - Genre distribution.
  - Rating vs number of ratings.
  - Simple correlations.
  - Optional word cloud of titles + genres.

---

## 3. Data Sources

1. **Goodreads — Best Books Ever**
   - Public list pages.
   - Scraped with polite delays and User-Agent header.

2. **Open Library — Search & Work Pages**
   - `https://openlibrary.org/search.json`
   - Sorted by `trending`.
   - Additional genres/subjects scraped from each work page under **Subjects**.

> All scraping/API usage follows the educational intent of the bootcamp project.

---

## 4. Repo Structure (Suggested)

```bash
project_book_recommend/
├── data/
│   ├── goodreads_best_books_600.csv
│   ├── openlibrary_trending_600.csv
│   └── books_merged_cleaned.csv
├── figures/
│   ├── eda_rating_distribution.png
│   ├── eda_top_authors.png
│   ├── eda_top_genres.png
│   ├── eda_rating_vs_num_ratings.png
│   └── eda_wordcloud_titles_genres.png
├── notebooks/
│   ├── scrape_goodreads_best_books_ever.ipynb
│   ├── scrape_openLibrary_trending_api.ipynb
│   └── books_data_cleaning_eda.ipynb
└── README.md
```

---

## 5. Environment & Setup

**Recommended packages:**

- `pandas`
- `numpy`
- `requests`
- `beautifulsoup4`
- `matplotlib`
- `scikit-learn`
- `wordcloud` (optional, for EDA word cloud)

Install (example):

```bash
pip install pandas numpy requests beautifulsoup4 matplotlib scikit-learn wordcloud
```

Then run the notebooks in order:

1. `scrape_goodreads_best_books_ever.ipynb`
2. `scrape_openLibrary_trending_api.ipynb`
3. `books_data_cleaning_eda.ipynb`

---

## 6. Workflow Summary

### Step 1 — Scraping Goodreads

- Loop over the **Best Books Ever** list pages.
- Extract:
  - Book title & URL
  - Author name & URL
  - Average rating
  - Number of ratings
  - Score & votes
  - List rank
- For each book page:
  - Extract **Genres** (new layout tags).
  - Extract **First published year** from book details.

### Step 2 — Open Library via API + Subjects

- Call `search.json` with the trending query.
- Extract:
  - Title, author(s)
  - Ratings average & count (if available)
  - `trending_score_hourly_sum`
  - Work key → build book URL.
- For each work URL:
  - Parse the **Subjects** block as genres.

### Step 3 — Merge & Clean

- Add a `source` column (`goodreads` / `openlibrary`).
- Align columns between both datasets.
- Concatenate into a single DataFrame.

### Step 4 — Data Cleaning, Feature Engineering & EDA

---

## 7. Key Steps Explained

### 5. Normalize genres

**What problem are we solving?**

After scraping, the `genres` column is messy:

- Some rows are empty.
- Some look like: `"Fantasy, Young Adult, Adventure"`.
- Others include subjects, noise, codes (`nyt:...`, `[fic]`, `pz7.1...`), or weird punctuation.
- We also saw strange list-like strings with brackets and repeated values.

That makes it hard to count genres or use them for recommendations.

**What do we do?**

For each row:

1. Make sure `genres` exists and convert to lowercase text.
2. Remove brackets `[]`, quotes `'`, and other list-like artifacts.
3. Replace separators like `/`, `&`, `=`, `--`, `-` with spaces so they don’t break words.
4. Split mainly by commas to get candidate genres/subjects.
5. Clean each candidate:
   - Keep only letters and spaces.
   - Remove very short or junk tokens.
   - Drop obvious technical codes like tags starting with `nyt`, `collectionid`, `pz`, etc.
6. Remove duplicates inside each book’s genre list.

**End result**

Each book gets a **clean list of genres**, for example:

```text
"Fantasy, Young Adult, Adventure"
→ ["fantasy", "young adult", "adventure"]
```

This makes it much easier to:

- Count how many books belong to each genre.
- Plot genre distributions.
- Use genres as features in a recommender.

---

### 8. Text for TF-IDF Keywords

**What problem are we solving?**

We want to extract “keywords” that describe each book, but:

- We don’t always have rich descriptions.
- We do have **titles** and **genres/subjects**.

So we build a simple combined text field to represent each book.

**What do we do?**

For each book:

1. Take the title (or `""` if missing).
2. Add a space.
3. Add the genres string.
4. Convert everything to lowercase.

Example:

```text
title:  "The Hunger Games"
genres: "young adult, dystopia, fiction"

→ "the hunger games young adult, dystopia, fiction"
```

We store this in a column like `text_for_keywords`.

**Why this is useful**

- This gives us one compact “summary text” per book.
- We can feed this into TF-IDF to find meaningful words.
- It’s a simple and realistic approach for a bootcamp-level project.

---

### 9. TF-IDF Keyword Extraction

Now we use **TF-IDF** to turn that text into keywords.

**What is TF-IDF (simple version)?**

- **TF (Term Frequency)**: Words that appear more often in a book’s text are more important for that book.
- **IDF (Inverse Document Frequency)**: Words that appear in many books (like “book”, “novel”) are less special.
- **TF-IDF** is high when:
  - A word is frequent in one book’s text.
  - But not so common across all books.

So TF-IDF helps us find **specific, meaningful words** for each book.

**What does the code do?**

1. Use `TfidfVectorizer` on `text_for_keywords`:
   - Limit to a small number of features (e.g. 50) to keep it simple.
   - Use English stopwords to ignore very common words.

2. For each book:
   - Look at its TF-IDF scores.
   - Sort words from highest to lowest score.
   - Pick the top few words (e.g. 5) with TF-IDF > 0.

3. Save them as `top_keywords`, for example:

```text
"dystopia, survival, rebellion, future, young"
```

**Why this is useful**

- These `top_keywords` act like a mini “fingerprint” of each book.
- We can later:
  - Compare books by overlapping keywords.
  - Build a simple content-based recommender:
    - “If you liked this book, here are others with similar genres/keywords.”

---

## 8. EDA Highlights

Using the cleaned dataset, we:

- Plot **rating distribution (0–5)**.
- Show **Top 10 most frequent authors**.
- Plot **Top 15 genres** (after genre normalization).
- Visualize **Rating vs Number of Ratings** (with log-scale on counts).
- Compute basic **correlations** between ratings, votes, and trending scores.
- Generate a **word cloud** from `text_for_keywords`.

All plots are also saved into the `figures/` directory for easy use in the presentation or report.

---

## 9. Next Steps (Future Work)

- Implement a **Content-Based Recommender** using:
  - Cleaned genres.
  - `top_keywords` from TF-IDF.
  - Similarity measures (e.g. cosine similarity).
- Add simple search & filter:
  - By genre, rating, popularity, etc.
- (Optional) Wrap the recommender in:
  - A Jupyter demo notebook, or
  - A small Streamlit app.

---

## 10. Authors

- **Luis Pablo Aiello** — Data Analytics Bootcamp Student
