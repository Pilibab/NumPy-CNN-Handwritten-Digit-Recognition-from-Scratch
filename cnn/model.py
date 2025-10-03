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
            weights:        optional pre-initialized weight matrix (see shape note below)
            bias:           optional pre-initialized bias vector

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
    
    def flatten(self, pool): return pool.flatten()


    def softmax_crossentropy_backward(self, logits, labels, from_logits=True, average=True):
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

        # 1) If labels are integer class indices, convert to one-hot of shape (batch_size, num_classes).
        # turns an int into list with 1 element [n]

        labels = np.atleast_1d(labels)

        class_size = logits.shape[1] if logits.ndim > 1 else logits.shape[0]
        batch_size = (len(labels), class_size)

        coded_labels = np.zeros(batch_size)
        coded_labels[np.arange(batch_size[0]), labels] = 1

        # 2) Compute stable softmax:
        #    - shift = logits - max(logits, axis=1, keepdims=True)
        #    - exp_shift = exp(shift)
        #    - probs = exp_shift / sum(exp_shift, axis=1, keepdims=True)
        prob = self.soft_max(logits)


        # 3) Compute loss (optional):
        #    - per-sample loss = -sum(label * log(probs + eps), axis=1)
        #    - loss = mean(per-sample loss) if average else sum(per-sample loss)
        loss = self.loss_function(labels, prob)


        # 4) Compute gradient dZ:
        #    - If using mean-loss: dZ = (probs - labels) / batch_size
        #    - If using sum-loss: dZ = probs - labels
        #    - This is the simplified result of backprop through softmax+cross-entropy.

        dZ = (prob - coded_labels) / batch_size[0]

        # 5) Return loss and dZ (or just dZ if you prefer)
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

        # 1) Unpack cached values: X, W, b
        #    - ensure X.shape = (batch_size, input_size)
        #    - ensure W.shape = (input_size, output_size)

        # 2) Optionally check shapes: dZ.shape == (batch_size, output_size)

        # 3) Compute gradient wrt weights:
        #    - dW = X.T @ dZ
        #    - If average: dW = dW / batch_size
        #    - If using L2 reg: dW += lambda * W  (document and implement separately)

        # 4) Compute gradient wrt bias:
        #    - db = sum(dZ, axis=0)
        #    - If average: db = db / batch_size

        # 5) Compute gradient wrt input:
        #    - dX = dZ @ W.T

        # 6) Return dX and grads dict (dW, db)
        pass

    def relu_backward(self, dA, cache):
        """
        Backprop for ReLU

        Args:
            dA:   gradient flowing from upper layer (same shape as activation output)
            cache: the pre-activation input Z or the activation output A from forward (needed to make mask)

        Returns:
            dZ: gradient w.r.t. pre-activation input (same shape)
        """

        # 1) Unpack cache: either Z (pre-activation) or A (post-activation).
        #    - If you saved Z: mask = (Z > 0)
        #    - If you saved A: mask = (A > 0)  (both work if ReLU is used)

        # 2) Elementwise multiply dA by mask:
        #    - dZ = dA * mask
        #    - This zeros-out gradients where the neuron was inactive during forward.

        # 3) Return dZ
        pass

    def maxpool_backward(self, dP, cache):
        """
        Backprop for 2D max pooling.

        Args:
            dP: gradient wrt pooled output (batch_size, out_h, out_w, depth) or (out_h,out_w) for single map
            cache: saved values from forward:
                - input feature map (or batch)
                - pool_size
                - stride
                - optionally argmax indices (recommended for speed)

        Returns:
            dX: gradient wrt input feature map (same shape as the cached input)
        """

        # 1) Unpack cached values: X (input), pool_size, stride, maybe argmax indices.
        #    - If you didn't save argmax during forward, you'll need to recompute which element was max for each window.

        # 2) Initialize dX as zeros with same shape as X (and same dtype as X).

        # 3) Loop over batch (if present), channels, and spatial positions:
        #    for each pooling window:
        #        - identify the location(s) of the max value (if ties, distribute? Usually first max)
        #        - route the scalar gradient dP[h,w,channel] to dX at that max location:
        #            dX[window_max_index] += dP[h,w,channel]
        #    - If you used "average" loss scaling, divide by batch_size appropriately (consistent with loss)

        # 4) Return dX
        pass

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

        # 1) Unpack cached values: X (original or padded), K (kernel), maybe original_input_shape

        # 2) Initialize gradients:
        #    - dK zeros same shape as K
        #    - dX zeros same shape as padded input (or original input + padding)
        #    - db zeros shape (out_channels,)

        # 3) Compute db:
        #    - db = sum(dOut over batch and spatial dims)  (sum across batch/out_h/out_w for each out_channel)

        # 4) Loop over batch images:
        #    For each image:
        #      For each out_channel:
        #        For each spatial position (h,w) in dOut:
        #          - compute corresponding input patch in X (taking stride and padding into account)
        #          - accumulate dK[:, :, in_channels, out_channel] += patch * dOut[image,h,w,out_channel]
        #          - accumulate dX_patch += K[:, :, :, out_channel] * dOut[image,h,w,out_channel]
        #    - This is the double-convolution-like accumulation; be mindful of broadcasting and channel ordering.

        # 5) After loops, if you padded X, remove padding from dX to match original input shape.

        # 6) If using mean-loss scaling, divide dK and db (and possibly dX) by batch_size to be consistent.

        # 7) Return dX (unpadded), dK, db
        pass


