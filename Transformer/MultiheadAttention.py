import numpy as np

class TransformerAttention:
    def __init__(self, seq_len, d, num_heads, n=1000):
        self.seq_len = seq_len
        self.d = d
        self.num_heads = num_heads
        self.d_k = d // num_heads  # Dimension of each head
        self.n = n
        self.P = self._positional_encoding()
        
        # Initialize weight matrices for multi-head attention
        self.W_Q = [np.random.randn(d, self.d_k) for _ in range(num_heads)]
        self.W_K = [np.random.randn(d, self.d_k) for _ in range(num_heads)]
        self.W_V = [np.random.randn(d, self.d_k) for _ in range(num_heads)]
        self.W_O = np.random.randn(num_heads * self.d_k, d)
    
    def _positional_encoding(self):
        P = np.zeros((self.seq_len, self.d))
        for k in range(self.seq_len):
            for i in np.arange(int(self.d / 2)):
                denominator = np.power(self.n, 2 * i / self.d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _attention(self, Q, K, V):
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        attention_weights = self._softmax(scores)
        return attention_weights @ V
    
    def multi_head_attention(self):
        Q = K = V = self.P
        heads = []
        
        # Compute attention for each head
        for i in range(self.num_heads):
            Q_i = Q @ self.W_Q[i]
            K_i = K @ self.W_K[i]
            V_i = V @ self.W_V[i]
            head = self._attention(Q_i, K_i, V_i)
            heads.append(head)
        
        # Concatenate all heads and apply the final linear transformation
        multi_head_output = np.concatenate(heads, axis=-1)
        return multi_head_output @ self.W_O

seq_len = 6  
d = 512       
num_heads = 8 

transformer_attention = TransformerAttention(seq_len, d, num_heads)
attention_output = transformer_attention.multi_head_attention()

print("Positional Encoding:\n", transformer_attention.P.shape)
print("Attention Output:\n", attention_output.shape)
