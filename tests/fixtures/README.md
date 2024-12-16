# Test Datasets

This directory contains a zip file for testing the RatingsDataModule and the training loop.  
The zip file contains two CSV files: `ml-latest/ratings_sample_100.csv` and `ml-latest/movies_sample_100.csv`.

## Datasets

1. **ratings_sample_100.csv**:  
A 100 row sample from the actual MovieLens dataset. For testing that the training loop works.  
Format:

```python
userId,movieId,rating,timestamp
1,1,4.0,964982703
```

2. **movies_sample_100.csv**:  
A movie metadata file that corresponds to the `movieId`'s in the `ratings_sample_100.csv` file. It does not contain 100 unique movies because some movies have multiple ratings.  
Format:

```python
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
```