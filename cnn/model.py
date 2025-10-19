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
            input_vector:   1D numpy array (shape (input_size,)) OR
                            2D numpy array for a batch (shape (batch_size, input_size))
                            (If you pass a 1D vector, treat it as a single-example batch.)
            output_size:    int, number of neurons in this dense layer (e.g., number of classes)
            weights:        optional pre-initialized weight matrix
            bias:           optional pre-initi alized bias vector

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
        H, W, _= image.shape
        kH, kW, _= kernel.shape

        feature_y = ((H - kH) // stride) + 1
        feature_x = ((W - kW) // stride) + 1

        # stores the convultion 
        feature_map = np.zeros((feature_y, feature_x))


        for h in range(feature_y):
            for w in range(feature_x):
                # a snippet of an image 
                patch = image[h * stride : h * stride + kH, w * stride : w * stride + kW, : ]
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
        # if logits.ndim == 1:    # single case
        #     exp_logits = np.exp(logits - np.max(logits)) 
        #     return exp_logits / np.sum(exp_logits)
        # elif logits.ndim == 2:  # batch case
        #     exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability improvement
        #     return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # else: 
        #     raise ValueError("logits must be either in 1d or 2d")
        
        # doesnt care abt dimension
        if logits.ndim == 1:
            logits = logits.reshape((1,-1))

        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        probs = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)

        return probs
    
    def loss_function(self, true_y, prediction):
        # eps = 1e-12
        # batch_size = len(true_y)
        # prediction = np.clip(prediction, eps, 1. - eps)

        # if true_y.ndim == 0:    # single case 
        #     return -np.log(prediction[true_y] + 1e-9)
        # if true_y.ndim == 1:    # batch case 
        #     probs = prediction[np.arange(batch_size), true_y]
        #     return -np.mean(np.log(probs + 1e-9), axis=1)    # sum the row 
        # else:
        #     raise ValueError("true y has not been 1 hot coded / is not 1d / 2d")

        # this works regardless if its batch / single case
        eps = 1e-12
        if prediction.ndim == 1:
            prediction = prediction.reshape(1,-1)
        prediction = np.clip(prediction, eps, 1. - eps)

        # ensure true_y is always an array (batch of at least size 1)
        true_y = np.atleast_1d(true_y)  
        batch_size = true_y.shape[0]

        # pick probabilities at the correct class indices
        probs = prediction[np.arange(batch_size), true_y]

        # mean over batch (even if it's just 1 element)
        return -np.mean(np.log(probs))
    
    def flatten(self, pool):
        # If no batch dimension, add one temporarily
        single_sample = False
        if pool.ndim == 3:  # (num_filters, H, W)
            pool = pool[np.newaxis, ...]  # -> (1, num_filters, H, W)
            single_sample = True

        batch_size = pool.shape[0]
        flat = pool.reshape(batch_size, -1)
        return (flat.squeeze(0) if single_sample else flat), pool.shape


    #* BACKPROP: Loss → dSoftmax → dDense → dFlatten → dPool → dReLU → dConv

    def softmax_crossentropy_backward(self, logits, labels):   
        """
        Combined backward for softmax + cross-entropy loss.

        Args:
            logits: raw scores (batch_size, num_classes) or (num_classes,) for single
            labels: array of label OR integer class indices
            from_logits: if True assume logits are raw and compute softmax inside
            average: if True return gradient normalized by batch_size (for mean loss)

        Returns:
            loss: scalar loss (optional to compute here)
            dZ: gradient of loss w.r.t. logits (same shape as logits), ready to feed into previous layer
        """

        labels = np.atleast_1d(labels)

        class_size = logits.shape[1] if logits.ndim > 1 else logits.shape[0]
        batch_size = (len(labels), class_size)

        coded_labels = np.zeros(batch_size)
        coded_labels[np.arange(batch_size[0]), labels] = 1

        prob = self.soft_max(logits)

        loss = self.loss_function(labels, prob)


        dZ = (prob - coded_labels) / batch_size[0]

        return loss, dZ

    def dense_backward(self, dZ, cache, average=True):
        """
        Backprop for a dense (fully-connected) forward: Z = X @ W + b

        Args:
            dZ:    gradient of loss w.r.t. logits Z
                shape: (batch_size, output_size)
            cache: dictionary or tuple containing values saved during forward:
                - X : input batch (batch_size, input_size)
                - W : weights (input_size, output_size)
                - b : bias (output_size,)
            average: if True divide dW and db by batch_size (consistent with mean-loss)

        Returns:
            dX: gradient w.r.t. inputs X, shape (batch_size, input_size)
            grads: dict with keys 'dW' (input_size, output_size) and 'db' (output_size,)
        """

        X, W, b = cache

        batch_size, _ =  X.shape 
        _, output_size = W.shape 

        if dZ.shape != (batch_size, output_size):
            raise ValueError("dZ shape missmatch")

        dW = np.dot(np.transpose(X), dZ)
        db = np.sum(dZ, axis=0)

        if average:
            db = db / batch_size
            dW = dW / batch_size

        dX = np.dot(dZ, np.transpose(W))

        grads = {"dW": dW, "db": db}
        return dX, grads

    def relu_backward(self, dA, cache):
        """
        Backprop for ReLU

        Args:
            dA:   gradient flowing from upper layer (same shape as activation output)
            cache: the pre-activation input Z or the activation output A from forward (needed to make mask)

        Returns:
            dZ: gradient w.r.t. pre-activation input (same shape)
        """

        Z = cache

        mask = (Z > 0).astype(float)

        dz = dA * mask

        return dz

    def maxpool_backward(self, dP, cache):
        """
        Backprop for 2D max pooling.

        Args:
            dP: gradient wrt pooled output (batch_size, out_h, out_w, depth) or (out_h,out_w) for single map
            cache: saved values from forward:
                - input feature map (or batch) 
                - stride
                - optionally argmax indices (recommended for speed)

        Returns:
            dX: gradient wrt input feature map (same shape as the cached input)
        """

        X, stride, pool_size = cache
        H, W = X.shape
        out_H, out_W = dP.shape

        dX = np.zeros_like(X)

        for i in range(out_H):
            for j in range(out_W):
                # Identify the window region used in forward
                h_start, h_end = i * stride, i * stride + pool_size
                w_start, w_end = j * stride, j * stride + pool_size

                # Extract that region
                window = X[h_start:h_end, w_start:w_end]

                # Find where the max was
                mask = (window == np.max(window))

                # Distribute the incoming gradient only to the max
                dX[h_start:h_end, w_start:w_end] += mask * dP[i, j]

        return dX
    
    def flatten_backward(self, dFlat, original_shape):
        return dFlat.reshape(original_shape)


    def conv_backward(self, dOut, cache, stride=1, padding=0):
        """
        Backprop for convolutional forward:
        out[h,w] = sum(kernel * patch)  (possibly across channels)

        Args:
            dOut: gradient wrt output feature map
                shape: (batch_size, out_h, out_w, out_channels) or (out_h,out_w) for single map
            cache: saved values from forward:
                - input X (possibly padded)
                - kernel K
                - stride
                - padding
            stride, padding: same values used in forward

        Returns:
            dX: gradient wrt input (unpadded original shape)
            dK: gradient wrt kernel (same shape as kernel)
            db: gradient wrt bias (one per out_channel)
        """

        X, K = cache
        if X.ndim == 3:  # single image
            X = X[np.newaxis, ...]
            dOut = dOut[np.newaxis, ...]
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w, _, out_c = K.shape
        _, out_h, out_w, _ = dOut.shape

        # Initialize gradients
        dX = np.zeros_like(X)
        dK = np.zeros_like(K)
        db = np.zeros((out_c,), dtype=np.float32)

        # Pad input and its gradient
        if padding > 0:
            X_pad = np.pad(X, ((0,0),(padding,padding),(padding,padding),(0,0)), mode='constant')
            dX_pad = np.pad(dX, ((0,0),(padding,padding),(padding,padding),(0,0)), mode='constant')
        else:
            X_pad = X
            dX_pad = dX

        # Bias gradient
        db = np.sum(dOut, axis=(0,1,2))  # sum over all spatial + batch dims

        # Main backward loops
        for b in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    patch = X_pad[b, h_start:h_start+k_h, w_start:w_start+k_w, :]
                    for c in range(out_c):
                        dK[:, :, :, c] += patch * dOut[b, h, w, c]
                        dX_pad[b, h_start:h_start+k_h, w_start:w_start+k_w, :] += K[:, :, :, c] * dOut[b, h, w, c]

        # Remove padding from dX if applied
        if padding > 0:
            dX = dX_pad[:, padding:-padding, padding:-padding, :]
        else:
            dX = dX_pad

        # Average over batch for stability (optional)
        dK /= batch_size
        db /= batch_size
        dX /= batch_size

        if dX.shape[0] == 1:
            dX = dX.squeeze(0)
            dOut = dOut.squeeze(0)

        return dX, dK, db


