import numpy
from PIL import Image

#bikin initial matrix 10x10 sesuai soal
img = numpy.array([
    [1,0,1,1,1,1,1,1,0,1],
    [1,0,1,1,1,1,1,1,1,1],
    [1,0,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,1,1,1],
    [1,0,1,0,0,0,0,1,1,1],
    [1,0,0,0,0,0,0,1,1,1],
    [1,0,1,0,0,0,0,1,1,1],
    [1,1,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1],
    [1,0,1,1,1,1,1,1,1,1]])


#bikin 2 kernel kosong dengan size 3x3
kernels = np.zeros([2,3,3])

kernels[0,:,:] = numpy.array([[
    [0, 1/4, 0],
    [1/4, 1/4, 1/4],
    [0, 1/4, 0]]])

kernels[1,:,:] = numpy.array([[
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]]])

# referensi -> https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = numpy.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in numpy.uint16(numpy.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in numpy.uint16(numpy.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            curr_region = img[r-numpy.uint16(numpy.floor(filter_size/2.0)):r+numpy.uint16(numpy.ceil(filter_size/2.0)), 
                              c-numpy.uint16(numpy.floor(filter_size/2.0)):c+numpy.uint16(numpy.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = numpy.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
            
    #Clipping the outliers of the result matrix.
    final_result = result[numpy.uint16(filter_size/2.0):result.shape[0]-numpy.uint16(filter_size/2.0), 
                          numpy.uint16(filter_size/2.0):result.shape[1]-numpy.uint16(filter_size/2.0)]
    return final_result

def convolution(img, kernels):
    if len(img.shape) > 2 or len(kernels.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != kernels.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
        if kernels.shape[1] != kernels.shape[2]: # Check if filter dimensions are equal.  
            print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')  
            sys.exit()
        if kernels.shape[1]%2==0: # Check if filter diemnsions are odd.
            print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
            sys.exit()  
        # An empty feature map to hold the output of convolving the filter(s) with the image.  
            feature_maps = numpy.zeros((img.shape[0]-kernels.shape[1]+1,img.shape[1]-kernels.shape[1]+1,kernels.shape[0]))
        for filter_num in range(kernels.shape[0]):
            print("Filter ", filter_num + 1)
            curr_filter = kernels[filter_num, :] # getting a filter from the bank.  
            """
            Checking if there are mutliple channels for the single filter.
            If so, then each channel will convolve the image.
            The result of all convolutions are summed to return a single feature map.
            """
            if len(curr_filter.shape) > 2:
                conv_map = conv_(img[:, :, 0], kernels[:, :, 0]) # Array holding the sum of all feature maps.
                for ch_num in range(1, kernels.shape[-1]): # Convolving each channel with the image and summing the results.
                    conv_map = conv_map + conv_(img[:, :, ch_num],kernels[:, :, ch_num])
            else: # There is just a single channel in the filter.
                conv_map = conv_(img, curr_filter)
            feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
        return feature_maps # Returning all feature maps.

def activation_relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = numpy.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in numpy.arange(0,feature_map.shape[0]):
            for c in numpy.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = numpy.max([feature_map[r, c, map_num], 0])
    return relu_out

def max_pooling(feature_map, size=2, stride=2):
#Preparing the output of the pooling operation.
pool_out = numpy.zeros((numpy.uint16((feature_map.shape[0]-size+1)/stride+1),
                        numpy.uint16((feature_map.shape[1]-size+1)/stride+1),
                        feature_map.shape[-1]))
for map_num in range(feature_map.shape[-1]):
    r2 = 0
    for r in numpy.arange(0,feature_map.shape[0]-size+1, stride):
        c2 = 0
        for c in numpy.arange(0, feature_map.shape[1]-size+1, stride):
            pool_out[r2, c2, map_num] = numpy.max([feature_map[r:r+size,  c:c+size, map_num]])
            c2 = c2 + 1
        r2 = r2 +1
return pool_out

#implementasi hitungan
initial_array = numpycnn.conv(img, kernels)
activation_array = activation_relu(initial_array)
pooling_array = max_pooling(activation_array, 2, 2)


#convert dari ndarray ke grey image
final_image = Image.fromarray(np.uint8(pooling_array * 255) , 'L')
final_image.show()