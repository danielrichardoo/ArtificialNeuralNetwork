import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3,   1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5,   3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5,   3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],

    [7,3.2,4.7,1.4],
    [6.4,3.2,4.5,1.5],
    [6.9,3.1,4.9,1.5],
    [5.5,2.3,4,1.3],
    [6.5,2.8,4.6,1.5],
    [5.7,2.8,4.5,1.3],
    [6.3,3.3,4.7,1.6],
    [4.9,2.4,3.3,1],
    [6.6,2.9,4.6,1.3],
    [5.2,2.7,3.9,1.4],

    [6.3,3.3,6,2.5],
    [5.8,2.7,5.1,1.9],
    [7.1,3,5.9,2.1],
    [6.3,2.9,5.6,1.8],
    [6.5,3,5.8,2.2],
    [7.6,3,6.6,2.1],
    [4.9,2.5,4.5,1.7],
    [7.3,2.9,6.3,1.8],
    [6.7,2.5,5.8,1.8],
    [7.2,3.6,6.1,2.5],
]

dataset = np.array(dataset)
mean = np.mean(dataset, axis = 0)

normalized_dataset = dataset - mean

transposed_dataset = np.transpose(normalized_dataset)
covariance_matrix = np.matmul(normalized_dataset,transposed_dataset)
# 2.a Covariance Matrix, Eigen Value, Eigen Vector:
print('Covariance matrix:')
print(covariance_matrix)

eigen_value ,eigen_vector = LA.eig(covariance_matrix)
print('Eigen value:')
print(eigen_value)
print('Eigen vector:')
print(eigen_vector)
# 2.b
reduced_dataset = PCA(n_components = 2).fit_transform(normalized_dataset)
print('Reduced dataset:')
print(reduced_dataset)

# 2.c
'''
Explanation:

Berikut hasil yang ditampilkan setelah data direduced.
Dari gambar tersebut menunjukan persebaran data menjadi kurang lebih 3 cluster 
dimana terdapat warna ungu(kiri atas), biru(kanan atas, tengah, kanan bawah), dan juga kuning(kiri bawah)

PCA sendiri melakukan reduce dimension pada dataset tersebut tanpa menghilangkan fitur utama dari data tersebut.
Jika kita mencoba untuk memvisualisasikan dataset sebelum direduced, akan menghasilkan warna yang kurang lebih terdiri dari 3 warna(cluster) 
dimana hasil tersebut mirip dengan data setelah direduced menjadi 2 dimension

'''
plt.title('Result')
# plt.imshow(dataset)
plt.imshow(reduced_dataset)
plt.xticks([])
plt.yticks([])
plt.show()