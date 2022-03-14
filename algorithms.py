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

#exercises = exercises.dropna()

'''
#gets value from general target area similarity
tfidf_matrix = tf.fit_transform(exercises['General Target Area']) #gets if-idf values 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) #calculate a numeric quantity that denotes the similarity between two movies. Higher the cosine value, the more similar the terms are
#print(type (cosine_sim))


#get value for  exercise category
tfidf_matrix2 = tf.fit_transform(exercises['Exercise Category'])
cosine_sim2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)

'''
tfidf_matrix3 = tf.fit_transform(['Beginner', 'Intermediate', 'Advanced'])
cosine_sim3 = linear_kernel(tfidf_matrix3, tfidf_matrix3)

#print(cosine_sim3)
'''


#get values for average of general target area and exercise category similarity
cosine_sim3 = (cosine_sim + cosine_sim2) / 2
#print(cosine_sim[0],cosine_sim2[1],avg[0])
#print(cosine_sim)
#print(movies['genres'])

'''

#print(indices)
arr = ['Cable Pull Through','Kettle Bell Swings','Pull-ups']

exercises =  pd.read_csv('exercises2.csv')
exercises = exercises.drop(columns=['Location', 'Push or Pull', 'Equipment Type( bar ,bodyweight, barbell , machine , kettlebell, dumbell , specific )'])
print(exercises)
titles = exercises['Name']
indices = pd.Series(exercises.index, index = exercises['Name'])


def create_cosine_similarities(categories,weight = None):
    if weight == None:
        weight = np.full(len(categories),1/len(categories))
    
    if len(categories) > 1:
        print('multiple weights are involved')
        sims = []
        for x in categories:
            tfidf_matrix = tf.fit_transform(exercises[x])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            sims.append(cosine_sim)
        weighted_sim = sims[0] * weight[0]
        theCount = 0
        for x,y in zip(sims,weight):
            if theCount > 0:
                #print(x,y)
                weighted_sim = weighted_sim +  (x * y)
            theCount = theCount + 1
        #weighted_sim = weighted_sim / len(sims)
        return weighted_sim
    elif categories[0] == 'Difficulty':
        theList = exercises['Difficulty'].tolist()
        #print(theList)
        list2 = []
        for x in theList:
            #print(x[:-1])
            if(x[-1].isspace()):
                list2.append(x[:-1])
            else:
                list2.append(x)
        #print(list2)
        cosine_sim = convert_difficulty(list2)
        return cosine_sim
    else:
        tfidf_matrix = tf.fit_transform(exercises[categories[0]]) #gets if-idf values
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) #calculate a numeric quantity that denotes the similarity between two movies. Higher the cosine value, the more similar the terms are
        return cosine_sim

def convert_difficulty(aList):
    return_list = []
    #print(aList)
    for x in aList:
        element_list = []
       # print(element_list)
        for y in aList:
            #print(x,y)
            if x == 'Beginner' and y == 'Beginner':
                #print('same value, beginner')
                element_list.append(1.0)
            elif x == 'Intermediate' and y == 'Intermediate':
                #print('same value, inter')
                element_list.append(1.0)
            elif x == 'Advanced' and y == 'Advanced':
                #print('same value, advanced')
                element_list.append(1.0)
            elif x == 'Beginner' and y == 'Intermediate':
                #print('different value')
                element_list.append(0.5)
            elif x == 'Beginner' and y == 'Advanced':
                #print('different value')
                element_list.append(0.25)
            elif x == 'Intermediate' and y == 'Beginner':
                #print('different value')
                element_list.append(0.5)
            elif x == 'Intermediate' and y == 'Advanced':#pontentially a problem
                #print('different value')
                element_list.append(0.5)
            elif x == 'Advanced' and y == 'Intermediate':
                #print('different value')
                element_list.append(0.5)
            elif x == 'Advanced' and y == 'Beginner':
                #print('different value')
                element_list.append(0.25)
        #print(element_list)
        return_list.append(element_list)
    return return_list
'''
cosine_sim_1 = create_cosine_similarities(['Target Area'])
print(cosine_sim_1)
print('-------------------------------------')
tfidf_matrix = tf.fit_transform(exercises['Target Area']) #gets if-idf values 
cosine_sim1 = linear_kernel(tfidf_matrix, tfidf_matrix) #calculate a numeric quantity that denotes the similarity between two movies. Higher the cosine value, the more similar the terms are
print(cosine_sim1)


print('-------------------------------------')
cosine_sim_2 = create_cosine_similarities(['Exercise Category'])
print(cosine_sim_2 [0])
print('-------------------------------------')
tfidf_matrix = tf.fit_transform(exercises['Exercise Category']) #gets if-idf values 
cosine_sim2 = linear_kernel(tfidf_matrix, tfidf_matrix) #calculate a numeric quantity that denotes the similarity between two movies. Higher the cosine value, the more similar the terms are
print(cosine_sim2[0])        


print('-------------------------------------')
cosine_sim_3 = create_cosine_similarities(['Difficulty'])
print(cosine_sim_3[0])
print('-------------------------------------')
theList = exercises['Difficulty'].tolist()
#print(theList)
list2 = []
for x in theList:
    #print(x[:-1])
    if(x[-1].isspace()):
        list2.append(x[:-1])
    else:
        list2.append(x)
        #print(list2)
cosine_sim3 = convert_difficulty(list2)
print(cosine_sim3[0])
    
print('-------------------------------------')
cosine_sim_4 = create_cosine_similarities(['Target Area','Exercise Category'],[0.5,0.5])
print(cosine_sim_4)
print('-------------------------------------')

tfidf_matrix = tf.fit_transform(exercises['Target Area']) #gets if-idf values 
cosine_sim_sum1 = linear_kernel(tfidf_matrix, tfidf_matrix)
tfidf_matrix = tf.fit_transform(exercises['Exercise Category']) #gets if-idf values 
cosine_sim_sum2 = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim4 = (cosine_sim_sum1 + cosine_sim_sum2) / 2
print(cosine_sim4)
print('-------------------------------------')

print(create_cosine_similarities(['Target Area','Exercise Category']))
arr = [93,367,8]

'''
#print(exercises['Difficulty'])
#print(type(exercises['Difficulty'].tolist()))
#print(type(['Difficulty']))

#print(convert_difficulty(['Beginner','Intermediate','Advanced','Intermediate']))
theList = exercises['Difficulty'].tolist()
#print(theList)
list2 = []
for x in theList:
    #print(x[:-1])
    if(x[-1].isspace()):
        list2.append(x[:-1])
    else:
        list2.append(x)
#print(list2)
    
#print(convert_difficulty(list2))
# Function that get movie recommendations based on the cosine similarity score of movie genres
def target_recommendations(title,sim):
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def give_set_recommendations(title,sim):
    arr2 = []
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    names2 =  titles.iloc[movie_indices]
    '''
    print(type(names2))
    print('names',names2)
    names2 = names2.reset_index()
    print('names',names2)
    '''
    count = 0
    for x in names2:
        print(x)
        if x in arr or x in arr2:
            print('exericse already in template')
        else:
            print('exericse pushed into array')
            arr2.append(x)
        
    return arr2[:3]+ arr

'''
print(target_recommendations('Deadlift',cosine_sim))
print(target_recommendations('Deadlift',cosine_sim2))
'''

print(target_recommendations('Deadlift',create_cosine_similarities(['Target Area'])))
print(target_recommendations('Deadlift',create_cosine_similarities(['Exercise Category'])))
print(target_recommendations('Deadlift',create_cosine_similarities(['Difficulty'])))
print(target_recommendations('Deadlift',create_cosine_similarities(['Target Area','Exercise Category'],[0.5,0.5])))
print(target_recommendations('Deadlift',create_cosine_similarities(['Target Area','Exercise Category'])))
'''
print(target_recommendations('Deadlift',cosine_sim1))
print(target_recommendations('Deadlift',cosine_sim_1))

print(target_recommendations('Deadlift',cosine_sim2))
print(target_recommendations('Deadlift',cosine_sim_2))

print(target_recommendations('Deadlift',cosine_sim3))
print(target_recommendations('Deadlift',cosine_sim_3))

print(target_recommendations('Deadlift',cosine_sim4))
print(target_recommendations('Deadlift',cosine_sim_4))
'''
#print(give_set_recommendations('Deadlift'))
