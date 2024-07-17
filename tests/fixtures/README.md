# Test Datasets

This directory contains datasets used for testing the functionality of the
movie recommendation system.

## Datasets

1. **ratings.csv**:  
Three rows of mock data that has the same format as the output from the MovieLensDataModule's `prepare_data()` (src/data/data_module.py) method.

2. **ratings_sample_100.csv**:  
A 100 row sample from the actual MovieLens dataset. For testing that the training loop works.