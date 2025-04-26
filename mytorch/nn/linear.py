import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        # store original shape so we can un-flatten later
        self.input_shape = A.shape

        # Flatten all dims except last
        A_flattened = A.reshape(-1, A.shape[-1])  # (batch_size, in_features)
        self.A = A_flattened
        self.N = A_flattened.shape[0]

        # ones vector for bias broadcast
        self.ones = np.ones((self.N, 1))

        # compute the affine transform: Z = A·W^T + b
        Z_flat = np.dot(A_flattened, self.W.T) + np.dot(self.ones, self.b.T)

        # reshape
        out_shape = (*self.input_shape[:-1], self.W.shape[0])
        Z = Z_flat.reshape(out_shape)

        # Store input for backward pass
        self.A = A

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        # Flatten gradients as in forward
        dLdZ_flat = dLdZ.reshape(-1, dLdZ.shape[-1])
        dLdA_flat = np.dot(dLdZ_flat, self.W)


        # Compute gradients (refer to the equations in the writeup)
        self.dLdW = np.dot(dLdZ_flat.T, self.A)
        self.dLdb = np.sum(dLdZ_flat, axis=0, keepdims=True).T

        dLdA = dLdA_flat.reshape(self.input_shape)

        if self.debug:
            self.dLdA = dLdA
        
        # Return gradient of loss wrt input
        return dLdA
