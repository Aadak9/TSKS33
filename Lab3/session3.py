import numpy as np

titles = open('titles/2.txt','r', encoding = 'utf-8').read().strip().splitlines()
links = np.genfromtxt('links/2.txt', delimiter=' ', dtype = int)

N = len(titles)

links -= 1

A = np.zeros((N, N))
for (i, j) in links:
    A[j, i] = 1

u = np.ones(N)

#===================Task 1================#
k_in = A @ u
k_out = A.T @ u

k_in = k_in/np.sum(k_in)
k_out = k_out/np.sum(k_out)

s = np.argsort(k_in)[-5:].tolist()[::-1]


top5titles = [titles[i] for i in s]
print("Task 1")
print('Top in-degree \t in-degree \t out_degree')
L = max(map(len, [titles[i] for i in s]))
for i in s:
    print('{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L = L, centrality=k_in[i], centrality2 = k_out[i]))

#===================Task 2================#

#C_nhub
#C_nauth
eigen_values_a, eigen_vectors_a = np.linalg.eig(A.T @ A)
a = eigen_vectors_a[:, np.argmax(eigen_values_a)]

eigen_values_h, eigen_vectors_h = np.linalg.eig(A @ A.T)
h = eigen_vectors_h[:, np.argmax(eigen_values_h)]

a = a/np.sum(a)
h = h/np.sum(h)

s2 = np.argsort(a)[-5:].tolist()[::-1]

top5titles2 = [titles[i] for i in s2]
print("Task 2")
print('Top in-degree \t in-degree \t out_degree')
L2 = max(map(len, [titles[i] for i in s2]))
for i in s2:
    print('{title:<{L2}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L2 = L2, centrality=a[i], centrality2 = h[i]))