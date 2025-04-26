import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        sum_exp = np.sum(exp_Z, axis=self.dim, keepdims=True)

        self.A = exp_Z / sum_exp
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            grad_moved = np.moveaxis(dLdA, self.dim, -1)
            flat_A = A_moved.reshape(-1, C)
            flat_grad = grad_moved.reshape(-1, C)
        else:
            flat_A = self.A.reshape(-1, C)
            flat_grad = dLdA.reshape(-1, C)

        inner = np.sum(flat_A * flat_grad, axis=1, keepdims=True)
        flat_dLdZ = flat_A * (flat_grad - inner)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            dLdZ_moved = flat_dLdZ.reshape(A_moved.shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            dLdZ = flat_dLdZ.reshape(shape)

        return dLdZ
 

    