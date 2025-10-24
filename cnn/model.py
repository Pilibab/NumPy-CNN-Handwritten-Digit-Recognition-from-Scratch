import numpy as np
import pickle
from math import sqrt
from cnn.data import load_binary

class CNN:
    def __init__(self, fs=5, ps=2):
        self.filter_size = fs
        self.pool_size = ps
        
        # Initialize parameters as None
        self.K = None  # Convolutional kernels
        self.W = None  # Dense layer weights
        self.b = None  # Dense layer bias
        self.num_filters = None
        self.num_classes = None

    def dense_layer(self, input_vector, output_size, weights=None, bias=None):
        """Dense (fully-connected) layer — forward pass."""
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
            if weights.shape != (input_size, output_size):
                raise ValueError("weights shape mismatch")

        if bias is None:
            bias = np.zeros((output_size,), dtype=np.float32)
        else:
            if bias.shape != (output_size,):
                raise ValueError("bias shape mismatch")

        logits = batch_sample.dot(weights) + bias

        if single_example: 
            return logits.squeeze(0)
        else: 
            return logits
        
    def convolution(self, image, kernels, stride=1, padding=0):
        """Convolutional layer forward pass."""
        H, W, c = image.shape
        kH, kW, cK, num_filters = kernels.shape

        feature_y = ((H - kH + 2 * padding) // stride) + 1
        feature_x = ((W - kW + 2 * padding) // stride) + 1

        feature_map = np.zeros((feature_y, feature_x, num_filters))

        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

        for f in range(num_filters):
            kernel = kernels[:, :, :, f]
            for h in range(feature_y):
                for w in range(feature_x):
                    patch = image[h * stride : h * stride + kH, w * stride : w * stride + kW, :]
                    feature_map[h, w, f] = np.sum(kernel * patch)

        return feature_map

    def reLu(self, feature_map):
        """ReLU activation function."""
        return np.maximum(0, feature_map)

    def max_pool(self, feature_map, stride=2, pool_size=2):
        """Max pooling operation - handles 3D feature maps."""
        if feature_map.ndim == 3:
            H, W, num_filters = feature_map.shape
            pool_H = ((H - pool_size) // stride) + 1
            pool_W = ((W - pool_size) // stride) + 1
            
            pooled = np.zeros((pool_H, pool_W, num_filters))
            
            for f in range(num_filters):
                for h in range(pool_H):
                    for w in range(pool_W):
                        pool = feature_map[h * stride : h * stride + pool_size, 
                                         w * stride : w * stride + pool_size, f]
                        pooled[h, w, f] = np.max(pool)
            return pooled
        else:
            # 2D case
            x, y = feature_map.shape
            pool_x = ((x - pool_size) // stride) + 1
            pool_y = ((y - pool_size) // stride) + 1
            
            pooled_feature = np.zeros((pool_x, pool_y))
            
            for h in range(pool_y):
                for w in range(pool_x):
                    pool = feature_map[h * stride : h * stride + pool_size, 
                                     w * stride : w * stride + pool_size]
                    pooled_feature[h, w] = np.max(pool)
            
            return pooled_feature
    
    def soft_max(self, logits):
        """Softmax activation function."""
        if logits.ndim == 1:
            logits = logits.reshape((1, -1))

        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        probs = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)

        return probs
    
    def loss_function(self, true_y, prediction):
        """Cross-entropy loss function."""
        eps = 1e-12
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)
        prediction = np.clip(prediction, eps, 1. - eps)

        true_y = np.atleast_1d(true_y)  
        batch_size = true_y.shape[0]

        probs = prediction[np.arange(batch_size), true_y]

        return -np.mean(np.log(probs))
    
    def flatten(self, pool):
        """Flatten pooled feature maps."""
        single_sample = False
        if pool.ndim == 3:
            pool = pool[np.newaxis, ...]
            single_sample = True

        batch_size = pool.shape[0]
        flat = pool.reshape(batch_size, -1)
        return (flat.squeeze(0) if single_sample else flat), pool.shape

    def softmax_crossentropy_backward(self, logits, labels):   
        """Combined backward for softmax + cross-entropy loss."""
        labels = np.atleast_1d(labels)

        class_size = logits.shape[1] if logits.ndim > 1 else logits.shape[0]
        batch_size_num = len(labels)

        coded_labels = np.zeros((batch_size_num, class_size))
        coded_labels[np.arange(batch_size_num), labels] = 1

        prob = self.soft_max(logits)
        loss = self.loss_function(labels, prob)

        dZ = (prob - coded_labels) / batch_size_num

        return loss, dZ

    def dense_backward(self, dZ, cache, average=True):
        """Backprop for dense layer."""
        X, W, b = cache

        batch_size, _ = X.shape 
        _, output_size = W.shape 

        if dZ.shape != (batch_size, output_size):
            raise ValueError("dZ shape mismatch")

        dW = np.dot(np.transpose(X), dZ)
        db = np.sum(dZ, axis=0)

        if average:
            db = db / batch_size
            dW = dW / batch_size

        dX = np.dot(dZ, np.transpose(W))

        grads = {"dW": dW, "db": db}
        return dX, grads

    def relu_backward(self, dA, cache):
        """Backprop for ReLU."""
        Z = cache
        mask = (Z > 0).astype(float)
        dz = dA * mask
        return dz

    def maxpool_backward(self, dP, cache):
        """Backprop for 2D max pooling."""
        X, stride, pool_size = cache
        
        if X.ndim == 3:
            H, W, num_filters = X.shape
            out_H, out_W, _ = dP.shape
            dX = np.zeros_like(X)
            
            for f in range(num_filters):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start, h_end = i * stride, i * stride + pool_size
                        w_start, w_end = j * stride, j * stride + pool_size
                        
                        window = X[h_start:h_end, w_start:w_end, f]
                        mask = (window == np.max(window))
                        dX[h_start:h_end, w_start:w_end, f] += mask * dP[i, j, f]
        else:
            H, W = X.shape
            out_H, out_W = dP.shape
            dX = np.zeros_like(X)
            
            for i in range(out_H):
                for j in range(out_W):
                    h_start, h_end = i * stride, i * stride + pool_size
                    w_start, w_end = j * stride, j * stride + pool_size
                    
                    window = X[h_start:h_end, w_start:w_end]
                    mask = (window == np.max(window))
                    dX[h_start:h_end, w_start:w_end] += mask * dP[i, j]

        return dX
    
    def flatten_backward(self, dFlat, original_shape):
        """Backprop for flatten operation."""
        return dFlat.reshape(original_shape)

    def conv_backward(self, dOut, cache, stride=1, padding=0):
        """Backprop for convolutional layer."""
        X, K = cache
        if X.ndim == 3:
            X = X[np.newaxis, ...]
            dOut = dOut[np.newaxis, ...]
            
        batch_size, in_h, in_w, in_c = X.shape
        k_h, k_w, _, out_c = K.shape
        _, out_h, out_w, _ = dOut.shape

        dX = np.zeros_like(X)
        dK = np.zeros_like(K)
        db = np.zeros((out_c,), dtype=np.float32)

        if padding > 0:
            X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
            dX_pad = np.pad(dX, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        else:
            X_pad = X
            dX_pad = dX

        db = np.sum(dOut, axis=(0, 1, 2))

        for b in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    w_start = w * stride
                    patch = X_pad[b, h_start:h_start+k_h, w_start:w_start+k_w, :]
                    for c in range(out_c):
                        dK[:, :, :, c] += patch * dOut[b, h, w, c]
                        dX_pad[b, h_start:h_start+k_h, w_start:w_start+k_w, :] += K[:, :, :, c] * dOut[b, h, w, c]

        if padding > 0:
            dX = dX_pad[:, padding:-padding, padding:-padding, :]
        else:
            dX = dX_pad

        dK /= batch_size
        db /= batch_size
        dX /= batch_size

        if dX.shape[0] == 1:
            dX = dX.squeeze(0)
            dOut = dOut.squeeze(0)

        return dX, dK, db

    def train(self, X, y, num_classes, epochs=10, lr=0.01, num_filters=8, 
            stride=1, padding=0, save_path='cnn_weights.pkl'):
        """
        Complete training loop with weight updates and saving.
        
        Args:
            X: input images, shape (batch_size, H, W, C)
            y: integer labels, shape (batch_size,)
            num_classes: number of output classes
            epochs: number of training iterations
            lr: learning rate
            num_filters: number of convolutional filters
            stride: convolution stride
            padding: convolution padding
            save_path: path to save trained weights
        """
        batch_size, H, W, C = X.shape
        self.num_filters = num_filters
        self.num_classes = num_classes

        # Initialize convolutional kernels
        self.K = np.random.randn(self.filter_size, self.filter_size, C, num_filters) * \
                 np.sqrt(2.0 / (self.filter_size * self.filter_size * C))

        print(f"Starting training with {batch_size} samples for {epochs} epochs...")
        print(f"Architecture: Conv({num_filters} filters) -> ReLU -> MaxPool -> Dense({num_classes} classes)")
        print("=" * 60)

        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for i in range(batch_size):
                xi = X[i]
                yi = y[i]

                # ===== FORWARD PASS =====
                conv_out = self.convolution(xi, self.K, stride, padding)
                relu_out = self.reLu(conv_out)
                pool_out = self.max_pool(relu_out, stride=self.pool_size, pool_size=self.pool_size)
                flat, flat_shape = self.flatten(pool_out)
                
                # Initialize dense weights on first iteration
                if self.W is None:
                    input_size = flat.shape[0]
                    self.W = np.random.randn(input_size, num_classes) * np.sqrt(2.0 / input_size)
                    self.b = np.zeros((num_classes,), dtype=np.float32)
                
                logits = self.dense_layer(flat, num_classes, self.W, self.b)
                probs = self.soft_max(logits)
                loss = self.loss_function(yi, probs)

                total_loss += loss
                pred_label = np.argmax(probs)
                if pred_label == yi:
                    correct += 1

                # ===== BACKWARD PASS =====
                # Softmax + Cross Entropy gradient
                _, dZ = self.softmax_crossentropy_backward(logits, yi)

                # Dense layer backward
                flat_reshaped = flat.reshape(1, -1)
                dFlat, dense_grads = self.dense_backward(dZ, (flat_reshaped, self.W, self.b))

                # Unflatten
                dPool = self.flatten_backward(dFlat, flat_shape)
                if dPool.ndim == 4:
                    dPool = dPool.squeeze(0)

                # Maxpool backward
                dRelu = self.maxpool_backward(dPool, (relu_out, self.pool_size, self.pool_size))

                # ReLU backward
                dConv = self.relu_backward(dRelu, relu_out)

                # Convolution backward
                dX, dK, db_conv = self.conv_backward(dConv, (xi, self.K), stride, padding)

                # ===== WEIGHT UPDATES =====
                # Update dense layer weights
                self.W -= lr * dense_grads["dW"]
                self.b -= lr * dense_grads["db"]

                # Update convolutional filters
                self.K -= lr * dK

            # ===== END OF EPOCH STATISTICS =====
            avg_loss = total_loss / batch_size
            accuracy = (correct / batch_size) * 100

            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:7.4f} | Accuracy: {accuracy:6.2f}%")

        print("=" * 60)
        print("Training complete!")
        
        # Save trained weights
        self.save_weights(save_path)
        print(f"Weights saved to: {save_path}")

    def save_weights(self, filepath='cnn_weights.pkl'):
        """
        Save trained weights and parameters to a file.
        
        Args:
            filepath: path where weights will be saved
        """
        weights_dict = {
            'K': self.K,
            'W': self.W,
            'b': self.b,
            'filter_size': self.filter_size,
            'pool_size': self.pool_size,
            'num_filters': self.num_filters,
            'num_classes': self.num_classes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(weights_dict, f)
        
        print(f"Model weights successfully saved to {filepath}")

    def load_weights(self, filepath='cnn_weights.pkl'):
        """
        Load trained weights and parameters from a file.
        
        Args:
            filepath: path to the saved weights file
        """
        with open(filepath, 'rb') as f:
            weights_dict = pickle.load(f)
        
        self.K = weights_dict['K']
        self.W = weights_dict['W']
        self.b = weights_dict['b']
        self.filter_size = weights_dict['filter_size']
        self.pool_size = weights_dict['pool_size']
        self.num_filters = weights_dict['num_filters']
        self.num_classes = weights_dict['num_classes']
        
        print(f"Model weights successfully loaded from {filepath}")

    def predict(self, X, stride=1, padding=0):
        """
        Make predictions on new data.
        
        Args:
            X: input images, shape (batch_size, H, W, C) or (H, W, C) for single image
            stride: convolution stride
            padding: convolution padding
            
        Returns:
            predictions: predicted class labels
            probabilities: class probabilities
        """
        if self.K is None or self.W is None:
            raise ValueError("Model not trained! Please train the model or load weights first.")
        
        single_image = False
        if X.ndim == 3:
            X = X[np.newaxis, ...]
            single_image = True
        
        batch_size = X.shape[0]
        predictions = []
        all_probs = []
        
        for i in range(batch_size):
            xi = X[i]
            
            # Forward pass
            conv_out = self.convolution(xi, self.K, stride, padding)
            relu_out = self.reLu(conv_out)
            pool_out = self.max_pool(relu_out, stride=self.pool_size, pool_size=self.pool_size)
            flat, _ = self.flatten(pool_out)
            logits = self.dense_layer(flat, self.num_classes, self.W, self.b)
            probs = self.soft_max(logits)
            
            pred = np.argmax(probs)
            predictions.append(pred)
            all_probs.append(probs.squeeze())
        
        predictions = np.array(predictions)
        all_probs = np.array(all_probs)
        
        if single_image:
            return predictions[0], all_probs[0]
        
        return predictions, all_probs


if __name__ == "__main__":
    # Example with dummy data (replace with real data like MNIST)
    xtrain, ytrain, xtest, ytest = load_binary()
    
    # Create and train CNN
    cnn = CNN(fs=5, ps=2)
    cnn.train(xtrain, ytrain, num_classes=62, epochs=5, lr=0.01, 
            num_filters=8, save_path='my_cnn_model.pkl')
    
    # Make predictions
    # X_test = np.random.randn(10, 28, 28, 1).astype(np.float32)
    # predictions, probabilities = cnn.predict(X_test)
    # print(f"\nPredictions: {predictions}")
    
    # Load weights later
    # cnn_new = CNN(fs=5, ps=2)
    # cnn_new.load_weights('my_cnn_model.pkl')
    # predictions = cnn_new.predict(X_test)