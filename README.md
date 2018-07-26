## recommendationsystem
[![Build Status](https://travis-ci.org/uwescience/recommendationsystem.svg?branch=master)](https://travis-ci.org/uwescience/recommendationsystem)

## Project

This is an implementation Movie Recommendation System with following features:
1. Popularity based recommendations
2. Item based recommendations
3. User similarity based recommendations(Collaborative Filtering)
4. Hybrid of Item and User similarity based recommendations

## Algorithms implemented:

1. Cosine, Euclieadean and Jaccard similarity
2. Clustering-kmeans, affinity propagation 
3. k-nearest neighbors
4. Singlular Value Decomposition(SVDs)
5. Artificial Neural Networks/Long Short Term Memory Network(LSTM) (Optimization techniques: Gradient Desecent/Adam loss)

## Optimization Metrics:
1. RMSE, MAE
2. Precision and recall
3. Mean Average Precision
4. Normalized Discounted Cumulative Gain(nDCG)


## Project data

Following is the description of varied data sources used for building this project. The source of data is Movie Lens which a publicly available data set and have been used by many prominent industry players to train their recommendation engines.

1. [Movie Lens Dataset 1](http://grouplens.org/datasets/movielens/latest/)

Description: 26,000,000 ratings and 750,000 tag applications applied to 45,000 movies by 270,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags. Last updated 8/2017.

2. [Movie Lens Dataset 2](http://grouplens.org/datasets/movielens/latest/)

Description: 100,000 ratings and 1,300 tag applications applied to 9,000 movies by 700 users. Last updated 10/2016.

3. [Movie Lens 100k Dataset](http://grouplens.org/datasets/movielens/100k/)

Description: 100,000 ratings and 1,300 tag applications applied to 1682 movies by 943 users. Released 4/1998.






