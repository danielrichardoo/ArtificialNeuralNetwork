import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def som(data,settings,map_weight):

	# Converting data to numpy array for better accuration and easy compute
    data = np.array(data,dtype=np.float32)
    map_weight = np.array(map_weight,dtype=np.float32)
    
    for i in range(settings['iteration']):
        bmu = {
        	'idx':-1,
        	'distance':-1
        }
        min_dist = 999999999
        # Getting best matching unit using euclidean distance
        for idx, row in enumerate(map_weight):
        	distance_between = distance.euclidean(row,data)
        	# If a new distance is smaller than the latest min_distance, then replace it with the new one
        	if min_dist > distance_between:
        		bmu['idx'] = idx
        		min_dist = distance_between
        		bmu['distance'] = min_dist
		
        # Updating value based on radius
        for idx, w_old in enumerate(map_weight):
        	idx_dist = np.abs(idx - bmu['idx'])
        	if idx_dist <= settings['radius']:
        		'''
				These formula are obtained from ANN - Binus Maya Slides

        		Sigma(n)

        		Formula:
        		sigma(n) = sigma(0) * exp(-n/T)
				
				NS -> Neighbour Strength
				Formula:
				NS = exp(-(d)^2 / 2 * sigma^2)

				'''
        		sigma = settings['sigma'] * np.exp(-(i+1)/settings['iteration'])
        		NS = np.exp(-np.power(idx_dist,2)/(2*np.power(sigma,2)))

        		w_new = w_old + NS * settings['learning_rate'] * (data - w_old)
        		map_weight[idx] = w_new        

    return map_weight

inputs = [
    [1,2,-1],
    [-1,3,2],
]

map_weight = [
    [1,1,-1],
    [2,1,1],
    [-1,2,-3],
    [1,2,3],
    [1,1,3],
]

settings = {
    'learning_rate':.5,
    'radius':1,
    'sigma':1,
    'iteration':1000
}

fig, axs = plt.subplots(nrows=2, ncols=1)

for i,data in enumerate(inputs):
	res = som(data, settings,map_weight)
	plt.subplot(2,1,i+1)
	plt.imshow(res)
	plt.title('Input - '+str(i+1)+str(data))
	plt.xticks([])
	plt.yticks([])

plt.show()