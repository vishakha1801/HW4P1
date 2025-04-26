from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project the query, key, and value inputs into query, key, and value
        # (N, L, E) -> (N, L, embed_dim)
        q = self.q_proj(query)
        # (N, S, E) -> (N, S, embed_dim)
        k = self.k_proj(key)
        # (N, S, E) -> (N, S, embed_dim)
        v = self.v_proj(value)

        def split(x, seq_len):
            d_k = self.embed_dim // self.num_heads
            x = x.reshape(self.N, seq_len, self.num_heads, d_k)
            return x.transpose(0, 2, 1, 3)

        # Split the query, key, and value into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        q = split(q, self.L)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        k = split(k, self.S)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        v = split(v, self.S)

        # Merge the masks
        # (N, S) + (L, S) -> (N, H, L, S)
        mask = None
        if key_padding_mask is not None or attn_mask is not None:
            mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads)
        attn_outputs = self.attention.forward(q, k, v, mask)

        # Merge the attention outputs   
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)
        attn_output = attn_outputs.transpose(0, 2, 1, 3).reshape(self.N, self.L, self.embed_dim)

        # Project the attention outputs
        # (N, L, embed_dim) -> (N, L, embed_dim)
        output = self.out_proj(attn_output)

        # Return output
        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # TODO: Implement backward pass 

        # Backpropagate through the output projection   
        # (N, L, embed_dim) -> (N, L, embed_dim) 
        d_attn_output = self.out_proj.backward(d_output)

        # Split the gradients into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        d_attn_outputs = d_attn_output.reshape(self.N, self.L, self.num_heads, self.embed_dim // self.num_heads).transpose(0, 2, 1, 3)

        # Backpropagate through the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Merge the gradients
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)    
        d_q = d_q.transpose(0, 2, 1, 3).reshape(self.N, self.L, self.embed_dim)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(self.N, self.S, self.embed_dim)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(self.N, self.S, self.embed_dim)

        # Backpropagate through the input projections   
        # (N, L, embed_dim) -> (N, L, embed_dim)
        d_q = self.q_proj.backward(d_q)
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_k = self.k_proj.backward(d_k)
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_v = self.v_proj.backward(d_v)

        # Return gradients d_q, d_k, d_v
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        # TODO: Implement merge masks

        # Expand key_padding_mask to (N, 1, 1, S) and broadcast to (N, H, L, S)
        if key_padding_mask is not None:
            key_mask = key_padding_mask[:, None, None, :]
            key_mask = np.broadcast_to(key_mask, (self.N, self.num_heads, self.L, self.S))
        else:
            key_mask = None
        
        # Expand attn_mask to (1, 1, L, S) and broadcast to (N, H, L, S)
        if attn_mask is not None:
            attention_mask = attn_mask[None, None, :, :]
            attention_mask = np.broadcast_to(attention_mask, (self.N, self.num_heads, self.L, self.S))
        else:
            attention_mask = None
        
        # Combine masks using logical_or - if either mask is True, we want to mask that position
        if key_mask is None:
            combined_mask = attention_mask
        elif attention_mask is None:
            combined_mask = key_mask
        else:
            combined_mask = key_mask | attention_mask

        # Return combined mask
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        # TODO: Implement split heads

        # Reshape: (N, L, embed_dim) -> (N, L, num_heads, embed_dim // num_heads)
        x = x.reshape(self.N, x.shape[1], self.num_heads, self.embed_dim // self.num_heads)
        
        # Transpose: (N, L, num_heads, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        x = x.transpose(0, 2, 1, 3)
        
        # Return x
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # TODO: Implement concat heads
        # Transpose: (N, num_heads, L, embed_dim // num_heads) -> (N, L, num_heads, embed_dim // num_heads)
        x = x.transpose(0, 2, 1, 3)
        
        # Reshape: (N, L, num_heads, embed_dim // num_heads) -> (N, L, embed_dim)
        x = x.reshape(self.N, self.L, self.embed_dim)
        
        # Return x
        return x
