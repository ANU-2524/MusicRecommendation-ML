# Music Recommendation System (Content-Based Filtering)

## Overview

This project implements a content-based music recommendation system using Natural Language Processing (NLP) techniques. The system recommends songs based on lyrical similarity by analyzing and comparing song lyrics.

Unlike collaborative filtering approaches that depend on user interaction data, this system relies purely on textual content from song lyrics.

---

## Dataset

**Source:** Kaggle  
**Dataset:** Spotify Million Song Dataset  
**Link:** https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset  
**License:** CC0-1.0  

### Dataset Details

- Total records: 57,650 songs  
- Columns:
  - `artist`
  - `song`
  - `link`
  - `text` (lyrics)

For computational efficiency, a random subset of 10,000 songs was used during model development.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- NLTK
- WordCloud
- Kaggle API

---

## Methodology

### 1. Data Collection
The dataset was downloaded using the Kaggle API and extracted into the working directory.

### 2. Data Preprocessing
- Removed non-alphabetic characters
- Converted text to lowercase
- Tokenized lyrics using NLTK
- Removed English stopwords
- Generated a cleaned text column for modeling

### 3. Exploratory Data Analysis
- Identified top contributing artists
- Generated a word cloud to visualize frequent words in lyrics

### 4. Feature Engineering
- Applied TF-IDF Vectorization (`max_features=5000`)
- Transformed cleaned lyrics into numerical feature vectors

### 5. Similarity Computation
- Computed cosine similarity between TF-IDF vectors
- Built a similarity matrix for all songs

### 6. Recommendation Engine
Given a song name:
- Locate its index in the dataset
- Retrieve similarity scores
- Sort scores in descending order
- Return the top N most similar songs

---

## Core Function

```python
def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset!"
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    return df[["artist", "song"]].iloc[song_indices]
