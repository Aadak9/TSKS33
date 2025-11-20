from load_data import load_data, getA, nr_users, nr_movies
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import math




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
    A = getA(training_data)
    
    r_bar = np.mean(training_data[:,2])
    c = training_data[:,2] - r_bar
    b = lsqr(A,c)[0] #b will be a vector with user and movie biases. First 2000 elements is userbias, rest is moviebiases.
    

    bu = b[:nr_users]
    bm = b[nr_users:]
    return r_bar,bu,bm

def baseline_prediction(training_data,datasets_to_predict):
    """
    Uses the training_data to train the baseline predictor, 
    then evaluates its performance on all the datasets in the test_datas list 
    """
    r_bar,bu,bm = train_baseline(training_data)

    
    # Create a list with one element for each dataset in datasets_to_predict.
    # Each entry (r_hat) should be an array with the predicted ratings for all pairs of users and movies in that dataset
    r_hats = []
    for data in datasets_to_predict:
        users = np.array(data[:,0])
        movies = np.array(data[:,1])
        r_hat = r_bar + bu[users] + bm[movies]
        r_hat = np.clip(r_hat, 1, 5)
        r_hats.append(r_hat)

    return r_hats

def neighborhood_prediction(training_data, datasets_to_predict, u_min=1, L=nr_movies):

    # ------------------------ Train baseline ------------------------
    #r_bar, bu, bm = train_baseline(training_data)
    baseline = baseline_prediction(training_data, datasets_to_predict)

    # ------------------------ Build R_tilde matrix ------------------------
    R_tilde = np.zeros((nr_users, nr_movies)) + 10

    for i,(user, movie, rating) in enumerate(training_data):
        prediction = baseline[0][i]  
        R_tilde[user, movie] = rating - prediction


    # ------------------------ Create cosine similarity matrix D ------------------------
    D = np.zeros((nr_movies, nr_movies))

    for i in range(nr_movies):

        for j in range(i, nr_movies):
            if i==j:
                D[i, j] = 1
            else:

            # users who rated both movies
                common = np.nonzero((R_tilde[:, i] != 10) & (R_tilde[:, j] != 10))[0]

            # apply u_min rule
                if len(common) >= u_min:
                    r_i = R_tilde[common, i]
                    r_j = R_tilde[common, j]

                    num = np.dot(r_i, r_j)
                    den = np.linalg.norm(r_i) * np.linalg.norm(r_j)

                    D[i, j] = num / den if den != 0 else 0
                    D[j, i] = D[i, j]

            D[j, i] = D[i, j]


    # ------------------------ Verification for D matrix ------------------------
    if filename == "verification" and u_min == 20:
        err = np.linalg.norm(np.load("verification_D_mat.npy") - D)
        print("Error in D matrix: {:.5f}".format(err))


    # ------------------------ Apply neighborhood predictor ------------------------
    r_hats = []

    for j, data in enumerate(datasets_to_predict):
        preds = np.zeros(len(data))

        for i in range(len(data)):

            user = data[i, 0]
            movie = data[i, 1]
            
            sims = D[movie]

            sim_movies = np.argsort(np.abs(sims))[::-1][0:L]
            
            weights = sims[sim_movies]

            denominator = np.sum(np.abs(weights))

            r_sim = R_tilde[user, sim_movies]

            ratings = (r_sim != 10)

            numerator = np.sum(weights[ratings] * r_sim[ratings])

            pred = baseline[j][i] + numerator/denominator

            preds[i] = np.clip(pred, 1, 5)

            


        r_hats.append(preds)

    return r_hats




def RMSE(r_hat,r):
    # Compute the RMSE between the true ratings r and the predicted ratings r_hat
    rmse = math.sqrt(np.mean((r - r_hat) ** 2))
    return rmse

def draw_histogram(r_hat,r,name=""):
    # Create the described histogram
    #r=true ratings from file
    #r_hat=predicted ratings from baseline_prediction
    r_hat_rounded = np.round(r_hat)

    abs_errors = np.abs(r_hat_rounded - r)

    plt.hist(abs_errors, bins=range(int(abs_errors.max()) + 2), edgecolor='black')
    plt.xlabel('Abs error')
    plt.ylabel('Frequency')
    plt.title('Absolute error of:' + name)
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

print("Training RMSE baseline: {0:.3f}".format(rmse_baseline_training))
print("Test RMSE baseline: {0:.3f}".format(rmse_baseline_test))

draw_histogram(r_hat_baseline_test, test_data[:,2],"Baseline Test")


# ====================================== TASK 2 ======================================
u_min = 20
L = 100

print("\n---- movie neighborhood predictor with u_min = {} and L = {} ----".format(u_min,L))

[r_hat_neighborhood_training,r_hat_neighborhood_test] = \
    neighborhood_prediction(training_data,[training_data,test_data],u_min,L)

rmse_neighborhood_training = RMSE(r_hat_neighborhood_training,training_data[:,2])
rmse_neighborhood_test = RMSE(r_hat_neighborhood_test,test_data[:,2])

print("Training RMSE neighboorhood: {0:.3f}".format(rmse_neighborhood_training))
print("Test RMSE neighboorhood: {0:.3f}".format(rmse_neighborhood_test))

print("\nTraining Improvement: {0:.3f}%".format(
    (rmse_baseline_training-rmse_neighborhood_training)/rmse_baseline_training*100))
print("Test Improvement: {0:.3f}%".format(
    (rmse_baseline_test-rmse_neighborhood_test)/rmse_baseline_test*100))

