
import numpy as np
import matplotlib.pyplot as plt


def compute_covariance_matrix(Z):

    #subtract mean from the samples
    feature_means = np.mean(Z, axis = 0)
    print(feature_means)
    Z = Z - feature_means
    Z_tp = np.transpose(Z)
    cov = np.dot(Z_tp, Z)
    return cov

def find_pcs(cov):

    L, pcs = np.linalg.eig(cov)

    #sort the eigenvalues in descending order
    sort_idx = np.argsort(L)[::-1]
 
    L = L[sort_idx]

    #sort the eigen vectors similar to the eigen values
    pcs = pcs[:,sort_idx]
    return pcs, L

def project_data(Z, PCS, L):

    #select the 1st pricipal component
    pcs_no = 1
    pcs_subset = PCS[:, 0:pcs_no]
    Z_star = np.dot(Z, pcs_subset)
    return Z_star

def show_plot(Z, Z_star):
    
    plt.figure(figsize=(8,8))  
    plt.scatter(Z[:,0], Z[:,1], label= 'Data')
    plt.scatter(Z_star, np.zeros(Z_star.shape),color= 'red', label= 'Projected Data')
    plt.legend(loc ="lower left")
    plt.grid()
    plt.show()

