#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

## Import first dataset and define columns
ratings = pd.read_csv(r'~\Documents\_Useful_Things\Programming\Data_Sets\book_ratings.csv')
columns = ['book_id', 'user_id', 'rating']

print("book_ratings.csv loaded.\n")

## Import second dataset and define columns
books = pd.read_csv(r'~\Documents\_Useful_Things\Programming\Data_Sets\books.csv')
columns = ['id', 'book_id', 'best_book_id', 'work_id', 
'books_count', 'isbn', 'isbn13', 'authors', 'original_publication_year',
'original_title', 'title', 'language_code', 'average_rating',
'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
'image_url', 'small_image_url']

print("books.csv loaded.")


## Merge the books and book_ratings datasets
combined_books_data = pd.merge(ratings, books, on='book_id')
print("books and ratings tables merged.")
print("Printing combined books data Preview")
print(combined_books_data.head())

## Create a utility matrix (matrix that omits certain variables for readability purposes) of user ratings
rating_utility_matrix = combined_books_data.pivot_table(values = 'rating', index = 'user_id', columns = 'title', fill_value = 0)

print("Printing utility_matrix Preview")
print(rating_utility_matrix.head())

print("Printing utility_matrix Dimensions")
print(rating_utility_matrix.shape)

## Transpose (swap x and y values) the utility matrix to arrange books in rows and users by column
X = rating_utility_matrix.T

## Use TruncatedSVD to compress the number of users to 30 components (variables) and fit the algorithm to the transposed utility matrix
SVD = TruncatedSVD(n_components=30)
transposed_matrix = SVD.fit_transform(X)
#print("Printing Transposed Matrix Preview")
#print(transposed_matrix.head()) TM isn't a pandas object?

print("Printing transposed matrix Dimensions")
print(transposed_matrix.shape)

## Create a numpy correlation matric, which is a table showing correlation coeffecients between variables
corr_matrix = np.corrcoef(transposed_matrix)

books = rating_utility_matrix.columns
books_list =  list(books)

firstBook = books_list.index('1984')

corr_1984 = corr_matrix[firstBook]

## Print a list of book names that have a correlation score of between 0.8 and 1.0. 
## The higher the correlation coefficient, the stronger the relationship a book has
## to the target book (1984).
print("Printing list of books that match the book 1984 with a correlation of 0.8 - 1.0")
print(list(books[(corr_1984 < 1.0) & (corr_1984 > 0.8)]))

print("End of program.")


# In[ ]:




