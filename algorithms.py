"""
Group 2 
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd
print('test')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english') #transforms text to feature vectors that can be used as input to estimator.
exercises  = pd.read_csv('exercises.csv')
print(exercises)
exercises = exercises.dropna()
print(exercises)
print('ttttttttttt')
print(exercises.columns)
tfidf_matrix = tf.fit_transform(exercises['General Target Area']) #gets if-idf values 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) #calculate a numeric quantity that denotes the similarity between two movies. Higher the cosine value, the more similar the terms are

#print(cosine_sim)
#print(movies['genres'])

titles = exercises['Name']
indices = pd.Series(exercises.index, index = exercises['Name'])
print(indices)

# Function that get movie recommendations based on the cosine similarity score of movie genres
def target_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


print(target_recommendations('Deadlift'))
