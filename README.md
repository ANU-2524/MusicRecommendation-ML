Music Recommendation System (Content-Based Filtering)
Overview

This project implements a content-based music recommendation system using Natural Language Processing (NLP) techniques. The system recommends songs based on lyrical similarity by analyzing and comparing song lyrics.

Unlike collaborative filtering approaches that depend on user interaction data, this system relies purely on textual content from song lyrics to generate recommendations.

Dataset

Source: Kaggle
Dataset: Spotify Million Song Dataset
Link: https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

License: CC0-1.0

Dataset Details

Total records: 57,650 songs

Columns:

artist

song

link

text (lyrics)

For computational efficiency, a random subset of 10,000 songs was used during model development.

Tech Stack

Python

NumPy

Pandas

Matplotlib

scikit-learn

NLTK

WordCloud

Kaggle API

Methodology
1. Data Collection

The dataset was downloaded using the Kaggle API and extracted into the working directory.

2. Data Preprocessing

Removed non-alphabetic characters

Converted text to lowercase

Tokenized lyrics using NLTK

Removed English stopwords

Generated a cleaned text column for further processing

3. Exploratory Data Analysis

Identified top contributing artists

Generated a word cloud to visualize the most frequent words in lyrics

4. Feature Engineering

Applied TF-IDF Vectorization (max_features=5000)

Converted cleaned lyrics into numerical feature vectors

5. Similarity Computation

Computed cosine similarity between TF-IDF vectors

Created a similarity matrix for all songs

6. Recommendation Engine

Given a song name:

Locate its index in the dataset

Retrieve similarity scores

Sort scores in descending order

Return the top N most similar songs

Core Function
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
Example

Input:

recommend_songs("Pigeon Song")

Output:
A list of top 5 songs with similar lyrical content.

Project Structure
├── kaggle.json
├── spotify-million-song-dataset.zip
├── spotify_millsongdata.csv
├── notebook.ipynb
└── README.md
Limitations

Uses only lyrical content (no audio features).

Does not consider user preferences.

Exact string matching is required for song search.

Large similarity matrix may consume significant memory for full dataset usage.

Future Improvements

Implement fuzzy matching for song search.

Add artist-based filtering.

Integrate audio features (tempo, energy, etc.).

Deploy as a web application using Flask or FastAPI.

Optimize similarity computation using approximate nearest neighbors (ANN).

Conclusion

This project demonstrates the application of NLP and vector space models to build a scalable, content-based music recommendation system. It provides a strong foundation for extending into hybrid or production-grade recommender systems.
