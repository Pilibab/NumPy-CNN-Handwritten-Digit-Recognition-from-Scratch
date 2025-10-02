import numpy as np
from math import sqrt

class CNN:
    def __init__(self, fs=5, ps=2):
        self.filter_size = fs
        self.pool_size = ps

    def dense_layer(self, input_vector, output_size, weights=None, bias=None):
        """
        Dense (fully-connected) layer — forward pass.ma

        Args:
            input_vector:  1D numpy array (shape (input_size,)) OR
                        2D numpy array for a batch (shape (batch_size, input_size))
                        (If you pass a 1D vector, treat it as a single-example batch.)
            output_size:   int, number of neurons in this dense layer (e.g., number of classes)
            weights:       optional pre-initialized weight matrix (see shape note below)
            bias:          optional pre-initialized bias vector

        Returns:
            logits: raw scores (shape (output_size,) for single input or
                                (batch_size, output_size) for batch input)
        """

        # If user passed a single 1D vector, make it a batch of 1 so dot products work uniformly.
        batch_sample = input_vector
        single_example = False
        if batch_sample.ndim == 1:
            batch_sample = batch_sample.reshape((1, batch_sample.shape[0]))
            single_example = True

        batch_sample = batch_sample.astype(np.float32)

        input_size = batch_sample.shape[1]

        if weights is None:
            scale = sqrt(2 / input_size)
            weights = np.random.randn(input_size, output_size) * scale
        else:
            # If user passed weights, ensure the shape matches what we expect.
            # If mismatch, raise a clear error (or reshape if you intend to).
            # e.g. if weights.shape != (input_size, output_size): raise ValueError(...)
            if weights.shape != (input_size, output_size):
                raise ValueError("weights shape mismatch")


        if bias is None:
            bias =  np.zeros((output_size,), dtype=np.float32)
        else:
            if bias.shape != (output_size,):
                raise ValueError("bias shape mismatch")


        if isinstance(weights, np.ndarray) and isinstance(bias, np.ndarray):
            logits = batch_sample.dot(weights) + bias   # bias broadcasts across batch dim

        # If original input was 1D, return a 1D logits vector (squeeze the batch dim).
        if single_example: 
            return logits.squeeze(0)
        # Otherwise return the batched logits.
        else: 
            return logits
        


    def convolution(self, image, kernel, stride=1):
        """
        Args:
            image (sample):     2d array / bit image shape (28,28)
            kernel (filter):    2d numpy array
            stride:             no of steps a filter take

        return:
            feature map:        
        """
        H,W = image.shape
        kH, kW = kernel.shape

        feature_y = ((H - kH) // stride) + 1
        feature_x = ((W - kW) // stride) + 1

        # stores the convultion 
        feature_map = np.zeros((feature_y, feature_x))


        for h in range(feature_y):
            for w in range(feature_x):
                # a snippet of an image 
                patch = image[h * stride : h * stride + kH, w * stride : w * stride + kW]
                feature_map[h,w] = np.sum(kernel * patch)

        return feature_map

    def reLu(self, feature_map):
        """
        returns activated feature map

        args: 
            feature_map:    2d array recieved from convulution 
        """
        # # creates a true or false mask 
        # mask = feature_map <= 0
        # # do reLu based of mask value
        # feature_map[mask] = 0
        # #return the activated feature map
        # return feature_map
    
        # more effecient way  
        return np.maximum(0, feature_map)   # -> compares the feature map element with zero (since 0 > neg num it replaces neg val with 0)

    def max_pool(self, feature_map, stride = 2, pool_size=2):
        x, y = feature_map.shape

        pool_x = ((x - pool_size) // stride) + 1
        pool_y = ((y - pool_size) // stride) + 1


        pooled_feature = np.zeros((pool_x, pool_y))

        for h in range(pool_y):
            for w in range(pool_x):
                # snippet of the pool 
                pool = feature_map[h * stride : h * stride + pool_size, w * stride : w * stride + pool_size]
                # chose the highest value within a pool 
                pooled_feature[h,w] = np.max(pool)

        return pooled_feature
    def soft_max(Self, logits):

        if logits.ndim == 1:    # single case
            exp_logits = np.exp(logits - np.max(logits)) 
            return exp_logits / np.sum(exp_logits)
        elif logits.ndim == 2:  # batch case
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability improvement
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else: 
            raise ValueError("logits must be either in 1d or 2d")
    
    def loss_function(self, true_y, prediction):
        eps = 1e-12
        prediction = np.clip(prediction, eps, 1. - eps)

        if true_y.ndim == 1:    # single case 
            return -np.sum(true_y * np.log(prediction + 1e-9))
        if true_y.ndim == 2:    # batch case 
            return -np.mean(np.sum(true_y * np.log(prediction + 1e-9), axis=1))    # sum the row 

        return 
    
    def flatten(self, pool): return pool.flatten()