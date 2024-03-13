import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def find_k_nearest_movies(movieId, k, userId):
    movie_idx = user_movie_matrix.columns.get_loc(movieId)
    similarities = cosine_sim_matrix[movie_idx]
    nearest_indices = np.argsort(similarities)[::-1]
    nearest_movies = nearest_indices[1:]  
    
    nearest_movies_filtered = []
    for nearest_movie_idx in nearest_movies:
        nearest_movie_id = user_movie_matrix.columns[nearest_movie_idx]
        if not np.isnan(actual_movie_matrix.loc[userId, nearest_movie_id]):
            nearest_movies_filtered.append(nearest_movie_id)
            if len(nearest_movies_filtered) == k:
                break  

    return nearest_movies_filtered 

def count_common_users(actual_movie_matrix, movie_id1, movie_id2):

    ratings_movie1 = actual_movie_matrix[movie_id1]
    ratings_movie2 = actual_movie_matrix[movie_id2]


    common_users_mask = ~np.isnan(ratings_movie1) & ~np.isnan(ratings_movie2)
    common_users_count = np.sum(common_users_mask)

    return common_users_count

def weighted_average_pred(test_nearest_movies, actual_movie_matrix):
    weighted_average_true_values = []
    weighted_average_pred_values = []

    for (userId, movieId), nearest_movies_data in test_nearest_movies.items():
        true_rating = test_data[(test_data['userId'] == userId) & (test_data['movieId'] == movieId)]['rating'].iloc[0] 

    
        weighted_sum = 0
        sum_of_weights = 0
        for nearest_movie_id, similarity in nearest_movies_data:
   
            user_rating = actual_movie_matrix.loc[userId, nearest_movie_id]

            weighted_sum += similarity * user_rating
            sum_of_weights += similarity


        if sum_of_weights != 0:
            predicted_rating = weighted_sum / sum_of_weights
        else:
            predicted_rating = 0

        if predicted_rating > 5:
            predicted_rating = 5
        elif predicted_rating <= 1:
            predicted_rating = 1
       
        weighted_average_true_values.append(true_rating)
        weighted_average_pred_values.append(predicted_rating)
    
    return weighted_average_true_values, weighted_average_pred_values

def weighted_average_common_users_pred(test_nearest_movies, actual_movie_matrix):
    weighted_average_common_users_true_values = []
    weighted_average_common_users_pred_values = []

    for (userId, movieId), nearest_movies_data in test_nearest_movies.items():
        true_rating = test_data[(test_data['userId'] == userId) & (test_data['movieId'] == movieId)]['rating'].iloc[0]

        
       
        common_users_weighted_sum = 0
        common_users_sum_of_weights = 0
        for nearest_movie_id, similarity in nearest_movies_data:
            com_count = count_common_users(actual_movie_matrix, movieId, nearest_movie_id)

            user_rating = actual_movie_matrix.loc[userId, nearest_movie_id]

            common_users_weighted_sum += com_count * user_rating
            common_users_sum_of_weights += com_count

     
        if common_users_sum_of_weights != 0:
            predicted_rating = common_users_weighted_sum / common_users_sum_of_weights
        else:
            predicted_rating = 0  

        if predicted_rating > 5:
            predicted_rating = 5
        elif predicted_rating <= 1:
            predicted_rating = 1
       
        weighted_average_common_users_true_values.append(true_rating)
        weighted_average_common_users_pred_values.append(predicted_rating)
    
    return weighted_average_common_users_true_values, weighted_average_common_users_pred_values

def weighted_average_adj_pred(test_nearest_movies, actual_movie_matrix):
    weighted_average_adj_true_values = []
    weighted_average_adj_pred_values = []

    for (userId, movieId), nearest_movies_data in test_nearest_movies.items():
        true_rating = test_data[(test_data['userId'] == userId) & (test_data['movieId'] == movieId)]['rating'].iloc[0] 

        user_avg_rating = actual_movie_matrix.loc[userId].mean()  

        adj_common_users_weighted_sum = 0
        adj_common_users_sum_of_weights = 0

        for nearest_movie_id, similarity in nearest_movies_data:
            neighbor_rating = actual_movie_matrix.loc[userId, nearest_movie_id]
            neighbor_bias = neighbor_rating - user_avg_rating  
            adj_common_users_weighted_sum += similarity * (neighbor_rating - neighbor_bias)  
            adj_common_users_sum_of_weights += similarity  

        if adj_common_users_sum_of_weights != 0:
            predicted_rating = adj_common_users_weighted_sum / adj_common_users_sum_of_weights 
        else:
            predicted_rating = 0  

        if predicted_rating > 5:
            predicted_rating = 5
        elif predicted_rating <= 1:
            predicted_rating = 1
       
        weighted_average_adj_true_values.append(true_rating)
        weighted_average_adj_pred_values.append(predicted_rating)
    
    return weighted_average_adj_true_values, weighted_average_adj_pred_values


def weighted_average_var_pred(test_nearest_movies, actual_movie_matrix):
    weighted_average_var_true_values = []
    weighted_average_var_pred_values = []

    for (userId, movieId), nearest_movies_data in test_nearest_movies.items():
        true_rating = test_data[(test_data['userId'] == userId) & (test_data['movieId'] == movieId)]['rating'].iloc[0] 

        var_common_users_weighted_sum = 0
        var_common_users_sum_of_weights = 0
        for nearest_movie_id, similarity in nearest_movies_data:
            movie_ratings = actual_movie_matrix[nearest_movie_id] 
            movie_ratings = movie_ratings[~np.isnan(movie_ratings)] 
            variance = np.var(movie_ratings)
            user_rating = actual_movie_matrix.loc[userId, nearest_movie_id]

            var_common_users_weighted_sum += variance * user_rating
            var_common_users_sum_of_weights += variance

     
        if var_common_users_sum_of_weights != 0:
            predicted_rating = var_common_users_weighted_sum / var_common_users_sum_of_weights
        else:
            predicted_rating = 0  

        if predicted_rating > 5:
            predicted_rating = 5
        elif predicted_rating <= 1:
            predicted_rating = 1
       
        weighted_average_var_true_values.append(true_rating)
        weighted_average_var_pred_values.append(predicted_rating)
    
    return weighted_average_var_true_values, weighted_average_var_pred_values

def Vals(true_values, pred_values):
    mae = np.mean(np.abs(true_values - pred_values))

    true_labels = []
    pred_labels = []

    for true_val, pred_val in zip(true_values, pred_values):
            if true_val >= 3:
                true_labels.append("positive")
            else:
                true_labels.append("negative")

            if pred_val >= 3:
                pred_labels.append("positive")
            else:
                pred_labels.append("negative")

    

    mar = recall_score(true_labels, pred_labels, average='macro')
    map = precision_score(true_labels, pred_labels, average='macro')
    conf_m = confusion_matrix(true_labels, pred_labels, labels=["positive", "negative"])

    new_row_data = {'k': k, 'MAE': mae, 'PasP': conf_m[0][0], 'PasN': conf_m[0][1], 'NasN': conf_m[1][1], 'NasP': conf_m[1][0], 'MAP': map, 'MAR': mar}
    
    return new_row_data

    


data = pd.read_csv('ratings.csv')
data.drop(columns=['timestamp'], inplace=True)
user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
actual_movie_matrix = user_movie_matrix.copy() 
train_data, test_data = train_test_split(data, test_size=0.2, random_state=11)

test_nearest_movies = {} 

for _, row in test_data.iterrows():
    userId = row['userId']
    movieId = row['movieId']
    rating = row['rating']
    

    if userId in user_movie_matrix.index and movieId in user_movie_matrix.columns:
        user_movie_matrix.loc[userId, movieId] = float('nan')
        actual_movie_matrix.loc[userId, movieId] = float('nan') 

for index, row in user_movie_matrix.iterrows():
    row_mean = row[~np.isnan(row)].mean()
    user_movie_matrix.loc[index] -= row_mean

cosine_sim_matrix = cosine_similarity(user_movie_matrix.T.fillna(0))

k = 6
weighted_average_true_values = []
weighted_average_pred_values = []

weighted_average_adj_true_values = []
weighted_average_adj_pred_values = []

weighted_average_common_users_true_values = []
weighted_average_common_users_true_pred_values = []

weighted_average_var_true_values = []
weighted_average_var_pred_values = []

for _, row in test_data.iterrows():
    userId = row['userId']
    movieId = row['movieId']
    
    nearest_movies = find_k_nearest_movies(movieId, k, userId=userId)
    
    test_nearest_movies[(userId, movieId)] = [(nearest_movie_id, cosine_sim_matrix[user_movie_matrix.columns.get_loc(movieId), user_movie_matrix.columns.get_loc(nearest_movie_id)]) for nearest_movie_id in nearest_movies]


weighted_average_true_values, weighted_average_pred_values = weighted_average_pred(test_nearest_movies, actual_movie_matrix)
weighted_average_common_users_true_values, weighted_average_common_users_pred_values = weighted_average_common_users_pred(test_nearest_movies, actual_movie_matrix)
weighted_average_adj_true_values, weighted_average_adj_pred_values = weighted_average_adj_pred(test_nearest_movies, actual_movie_matrix)
weighted_average_var_true_values, weighted_average_var_pred_values = weighted_average_var_pred(test_nearest_movies, actual_movie_matrix)

df_wa = pd.read_csv('weighted_average.csv')
df_cu = pd.read_csv('common_users.csv')
df_adj = pd.read_csv('adj.csv')
df_var = pd.read_csv('varians.csv')

wa = Vals(np.array(weighted_average_true_values), np.array(weighted_average_pred_values))
cu = Vals(np.array(weighted_average_common_users_true_values), np.array(weighted_average_common_users_pred_values))
adj = Vals(np.array(weighted_average_adj_true_values), np.array(weighted_average_adj_pred_values))
var = Vals(np.array(weighted_average_var_true_values), np.array(weighted_average_var_pred_values))

df_wa.loc[len(df_wa)] = wa
df_cu.loc[len(df_cu)] = cu
df_adj.loc[len(df_adj)] = adj
df_var.loc[len(df_var)] = var

df_wa.to_csv("weighted_average.csv", index=False)
df_cu.to_csv("common_users.csv", index=False)
df_adj.to_csv("adj.csv", index=False)
df_var.to_csv("varians.csv", index=False)