import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
from load_data import load_data, getA

nr_users = 2000
nr_movies = 1500

def train_baseline(training_data):
   """
   Uses the provided dataset to train the baseline predictor. 
   Should return three things:
   r_bar: the average rating over all users and movies
   bu: vector where the i:th entry represents the bias of the i:th user compared to r_bar
   bm: vector where the i:th entry represents the bias of the i:th movie compared to r_bar
   """

   r_bar = np.mean(training_data[:, 2])
   c = training_data[:, 2] - r_bar
   A = getA(training_data)
   b = lsqr(A, c)[0]

   bu = b[:nr_users]
   bm = b[nr_users:]
   
   return r_bar,bu,bm

def baseline_prediction(training_data, datasets_to_predict):
   """
   Uses the training_data to train the baseline predictor, 
   then evaluates its performance on all the datasets in the test_datas list 
   """
   r_bar,bu,bm = train_baseline(training_data)

   
   # Create a list with one element for each dataset in datasets_to_predict.
   # Each entry (r_hat) should be an array with the predicted ratings for all pairs of users and movies in that dataset
   r_hats = []
   for data in datasets_to_predict:
       # Initialize the r_hat vector with zeros
       r_hat = np.zeros(data.shape[0])

       for i, (user, movie, _) in enumerate(data):
           # Calculate the baseline prediction using r_bar, bu, and bm
           prediction = r_bar + bu[user] + bm[movie]
           #print(prediction)
           #print(bu[user], bm[movie])
           r_hat[i] = np.clip(prediction, 1, 5)

       r_hats.append(r_hat)

   return r_hats

def cosine_similarity_matrix(R_tilde, u_min):
   # Calculate the cosine similarity matrix for movies
   sim_matrix = np.zeros((nr_movies, nr_movies))
   for i in range(nr_movies):
      for j in range(i, nr_movies):  # Avoid redundant computations
         # Get vectors of ratings for movies i and j
         #ratings_i = R[:, i]
         if i == j:
            sim_matrix[i, i] = 1
         ratings_i = R_tilde[:, i]
         ratings_j = R_tilde[:, j]

         common_ratings = (ratings_i != 10) & (ratings_j != 10)
         num_common_ratings = np.sum(common_ratings)

         if num_common_ratings >= u_min:
            # Compute cosine similarity between movie i and movie j
            ratings_i = ratings_i[common_ratings]
            ratings_j = ratings_j[common_ratings]
            numerator = np.dot(ratings_i, ratings_j) # ratings_i @ ratings_j
            denominator = np.linalg.norm(ratings_i) * np.linalg.norm(ratings_j)
               
            if denominator != 0:
               similarity = numerator / denominator
               sim_matrix[i, j] = similarity
               sim_matrix[j, i] = similarity  # Symmetric matrix

   return sim_matrix

def neighborhood_prediction(training_data, datasets_to_predict, u_min, L=nr_movies):
   """
   Uses the training_data to train the improved predictor, 
   then evaluates its performance on all the datasets in the test_datas list 
   """

   # ========================== Create the cosine similarity matrix D ==========================

   r_hat_baseline = baseline_prediction(training_data, datasets_to_predict)
   r_hat_baseline_training = r_hat_baseline[0]
   R_tilde = np.zeros((nr_users, nr_movies)) + 10

   for i, (user, movie, actual_rating) in enumerate(training_data):
      prediction = r_hat_baseline_training[i]
      R_tilde[user, movie] = actual_rating - prediction

   D = cosine_similarity_matrix(R_tilde, u_min)

   # Uncomment the following lines to check the correctness of the D matrix for u_min = 20 in the verification dataset
   # You should get an error that is less than 1e-5
   #
   if filename == "verification" and u_min == 20:
       error_in_D = np.linalg.norm(np.load('verification_D_mat.npy') - D)
       print("Error in D matrix: {0:.5f}\n".format(error_in_D))

   # =================== Evaluate the performance of the improved predictor ====================
   
   r_hats = []
   for i, data in enumerate(datasets_to_predict):
      r_hat = np.zeros(len(data))
      
      for idx, (user, movie, _) in enumerate(data):
         # Get the baseline prediction
         baseline = r_hat_baseline[i][idx]

         # Find top-L similar movies
         similarities = D[movie]
         most_similar = np.argsort(similarities)[::-1][0:L]

         # Calculate the weights for the most similar movies and compute the denominator
         weights = similarities[most_similar]
         denominator = np.sum(np.abs(weights))

         # Compute the prediction for the neighborhood prediction model
         if denominator == 0:
            prediction = baseline
         else:
            r_similar = R_tilde[user, most_similar]
            valid_ratings = (r_similar != 10)
            nominator = np.sum(weights[valid_ratings] * r_similar[valid_ratings])
            prediction = baseline + nominator / denominator
         
         r_hat[idx] = np.clip(prediction, 1, 5)

         # Calculate the numerator and denominator for the neighborhood prediction
         """numerator, denominator = 0, 0
         for j in most_similar:
            if R_tilde[user, j] != 10:  # User must have rated movie j
               numerator += similarities[j] * R_tilde[user, j]
               denominator += abs(similarities[j])

         if denominator > 0:
            neighborhood_correction = numerator / denominator
         else:
            neighborhood_correction = 0

         # Final prediction
         prediction = baseline + neighborhood_correction
         r_hat[idx] = np.clip(prediction, 1, 5)  # Clip to valid range"""

      r_hats.append(r_hat)

   return r_hats


def RMSE(r_hat,r):
   # Compute the RMSE between the true ratings r and the predicted ratings r_hat
   rmse = np.sqrt(np.mean((r_hat - r)**2))
   return rmse

def draw_histogram(r_hat,r,name=""):
   # Create the described histogram

   # Step 1: Round each predicted rating to the nearest integer in {1, 2, 3, 4, 5}
   r_hat_rounded = np.clip(np.round(r_hat), 1, 5)
   
   # Step 2: Calculate absolute errors (deviations) between rounded predictions and actual ratings
   errors = np.abs(r_hat_rounded - r)
   
   # Step 3: Plot the histogram of absolute errors
   plt.figure(figsize=(10, 6))
   plt.hist(errors, bins=np.arange(-0.5, 5.5, 1), edgecolor='black', align='mid')
   plt.xlabel('Absolute Error (Deviation)')
   plt.ylabel('Frequency')
   plt.title(f"Absolute Error Histogram {name}")
   plt.xticks(range(6))  # to show each error value from 0 to 5 clearly
   
   # Display the histogram
   plt.show()
   return 0
filename = "verification"

training_data = load_data(filename+'.training')
test_data = load_data(filename+'.test')


# ====================================== TASK 1 ======================================
print("---- baseline predictor ----")

[r_hat_baseline_training,r_hat_baseline_test] = \
   baseline_prediction(training_data,[training_data,test_data])

rmse_baseline_training = RMSE(r_hat_baseline_training,training_data[:,2])
rmse_baseline_test = RMSE(r_hat_baseline_test,test_data[:,2])

print("Training RMSE: {0:.3f}".format(rmse_baseline_training))
print("Test RMSE: {0:.3f}".format(rmse_baseline_test))

draw_histogram(r_hat_baseline_test, test_data[:,2],"Baseline Test")


# ====================================== TASK 2 ======================================
u_min = 20
L = 100
print("\n---- movie neighborhood predictor with u_min = {} and L = {} ----".format(u_min,L))

[r_hat_neighborhood_training,r_hat_neighborhood_test] = \
   neighborhood_prediction(training_data,[training_data,test_data],u_min,L)

rmse_neighborhood_training = RMSE(r_hat_neighborhood_training,training_data[:,2])
rmse_neighborhood_test = RMSE(r_hat_neighborhood_test,test_data[:,2])

print("Training RMSE: {0:.3f}".format(rmse_neighborhood_training))
print("Test RMSE: {0:.3f}".format(rmse_neighborhood_test))

print("\nTraining Improvement: {0:.3f}%".format(
   (rmse_baseline_training-rmse_neighborhood_training)/rmse_baseline_training*100))
print("Test Improvement: {0:.3f}%".format(
   (rmse_baseline_test-rmse_neighborhood_test)/rmse_baseline_test*100))