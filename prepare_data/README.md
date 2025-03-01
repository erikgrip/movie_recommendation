# Movie Recommendation Data Preparation

This directory contains scripts for downloading, cleaning, and preparing data for the movie recommendation system.

## Overview

The data preparation pipeline consists of the following steps:

1. **Download**: Fetches the MovieLens dataset from GroupLens Research
2. **Clean**: Processes raw data by cleaning movie titles, extracting release years, and formatting ratings
3. **Feature Engineering**: Creates features for the recommendation model including:
   - Movie title embeddings using SentenceTransformer
   - Genre dummy variables
   - User-genre average ratings

## Files

- `main.py`: Entry point that runs the complete data preparation pipeline
- `config.py`: Configuration settings including file paths and URLs
- `download.py`: Downloads and extracts the MovieLens dataset
- `clean.py`: Cleans and preprocesses the raw data
- `features.py`: Generates features for the recommendation model

## Usage

To run the complete data preparation pipeline, use one of the following methods:

### Using Poetry (recommended)

If you've installed the project with Poetry:

```bash
# From project root
PYTHONPATH=. poetry run python -m prepare_data.main
```

## Data Structure

The pipeline creates the following directory structure:

```
data/
├── raw/                  # Raw downloaded zip file
├── extracted/            # Extracted CSV files
├── clean/                # Cleaned parquet files
└── features/             # Generated features
    ├── movie_title_embeddings.parquet
    ├── movie_genre_dummies.parquet
    └── user_genre_avg_ratings.parquet
```

## Dependencies

- pandas: Data manipulation
- polars: Fast data processing
- sentence-transformers: Text embedding generation 