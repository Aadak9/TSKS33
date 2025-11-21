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

"""
top_out_idx = np.argsort(k_out)[-10:][::-1]

print("Top 10 out-degree articles:")
for i in top_out_idx:
    print(f"{titles[i]:<30} {k_out[i]:.6f} {k_in[i]:.6f}")
"""

s = np.argsort(k_in)[-5:].tolist()[::-1]


top5titles = [titles[i] for i in s]
print("Task 1")
print('Top in-degree \t in-degree \t out_degree')
L = max(map(len, [titles[i] for i in s]))
for i in s:
    print('{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L = L, centrality=k_in[i], centrality2 = k_out[i]))

# sort first by k_out descending, then by k_in descending
s = np.lexsort((k_in, k_out))[::-1][:5] #Note that Erich_Schmidt_(archaeologist) 0.002638 0.000012 Ahmad_Zarruq 0.002638 0.001404
#so you cant sort on just k_out since they have the same out_degree, a so called slamkrypare

print("\n")
print("Task 1")


L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}       {:>12}   {:>12}'.format('Top out-degree', 'out-degree', 'in-degree', w=L))
for i in s:
    print('{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L = L, centrality=k_out[i], centrality2 = k_in[i]))

#===================Task 2================#
print("\n")
#C_nhub
#C_nauth
eigen_values_a, eigen_vectors_a = np.linalg.eig(A.T @ A)
a = eigen_vectors_a[:, np.argmax(eigen_values_a)]

eigen_values_h, eigen_vectors_h = np.linalg.eig(A @ A.T)
h = eigen_vectors_h[:, np.argmax(eigen_values_h)]

a = a/np.sum(a)
h = h/np.sum(h)

a = a.real
h = h.real

s = np.argsort(a)[-5:].tolist()[::-1]


print("Task 2")
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}   {:>12}       {:>12}'.format('Top hubs', 'hub', 'authority', w=L))
for i in s:
    print('{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L = L, centrality=a[i], centrality2 = h[i]))

s = np.argsort(h)[-5:].tolist()[::-1]
print("\n")
print("Task 2")
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12} {:>12}'.format('Top authorities', 'authority', 'hub', w=L))
for i in s:
    print('{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}' .format(title=titles[i], L = L, centrality=h[i], centrality2 = a[i]))


#===================Task 3================#
print("\n")

eigen_values_a, eigen_vectors_a = np.linalg.eig(A)
a = eigen_vectors_a[:, np.argmax(eigen_values_a)].real
a = a/np.sum(a)

s = np.argsort(a)[-5:].tolist()[::-1]
print("\n")
print("Task 3")
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12}'.format('Top eigenvector centrality', 'Eigenvector centrality', w=L))
for i in s:
    print('{:<{w}}                  {:>12.6f}'.format(titles[i], a[i], w=L))



#===================Task 4================#
print("\n")

eigen_values, _ = np.linalg.eig(A)
alpha = 0.85 / np.max(np.abs(eigen_values))


I = np.eye(A.shape[0])


C = np.linalg.inv(I - alpha * A) @ u
C = C.flatten()
C = C/np.sum(C)
s = np.argsort(C)[-5:].tolist()[::-1]
print("\n")
print("Task 4")
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12}'.format('Top katz', 'Katz centrality', w=L))
for i in s:
    print('{:<{w}}     {:>12.6f}'.format(titles[i], C[i], w=L))

#===================Task 5================#
print("\n")

k_out = A.T @ u
H = np.zeros((N,N))
for i in range(N): 
    for j in range(N): 
        if k_out[j] == 0:
            element = 1 / N
        else:
            element = A[i][j] / k_out[j]
        H[i][j] = element



alpha = 0.3
C = (1-alpha) / N * np.linalg.inv(I-alpha*H) @ u
C = C/np.sum(C)

print("\n")
print("Task 5")
s = np.argsort(C)[-5:].tolist()[::-1]
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12}'.format('Top PageRank, alpha = 0.3', 'PageRank', w=L))
for i in s:
    print('{:<{w}}              {:>12.6f}'.format(titles[i], C[i], w=L))

alpha = 0.99
C = (1-alpha) / N * np.linalg.inv(I-alpha*H) @ u
C = C/np.sum(C)

print("\n")
print("Task 5")
s = np.argsort(C)[-5:].tolist()[::-1]
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12}'.format('Top PageRank, alpha = 0.99', 'PageRank', w=L))
for i in s:
    print('{:<{w}}            {:>12.6f}'.format(titles[i], C[i], w=L))

alpha = 0.85
C = (1-alpha) / N * np.linalg.inv(I-alpha*H) @ u
C = C/np.sum(C)

print("\n")
print("Task 5")
s = np.argsort(C)[-5:].tolist()[::-1]
L = max(map(len, [titles[i] for i in s]))
print('{:<{w}}      {:>12}'.format('Top PageRank, alpha = 0.85', 'PageRank', w=L))
for i in s:
    print('{:<{w}}               {:>12.6f}'.format(titles[i], C[i], w=L))

#===================Task 6================#
print("\nTask 6")

G = alpha * H + (1 - alpha) / N * np.ones((N, N))
x = np.ones(N) / N
iterations_to_record = [1, 2, 5, 10, 100]
recorded = {}

for t in range(1, 101 + 1):
    x = G @ x
    x = x / np.sum(x)
    if t in iterations_to_record:
        recorded[t] = x.copy()

# ---- Print top 5 articles at each recorded iteration ----
for t in iterations_to_record:
    C_iter = recorded[t]
    s = np.argsort(C_iter)[-5:][::-1]  # top 5 articles for this iteration
    L = max(map(len, [titles[i] for i in s]))
    print(f"\nTop iterative PageRank, alpha = 0.85, iteration {t}")
    for i in s:
        print(f"{titles[i]:<{L}}                      {C_iter[i]:.6f}")

