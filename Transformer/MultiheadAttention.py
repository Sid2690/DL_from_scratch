import numpy as np

class MultiheadAttention:
    def __init__(self, seq_len, num_heads, d, n=1000):
        self.seq_len = seq_len
        self.d = d
        self.n = n
        self.num_heads = num_heads
        self.P = self._positionalEncoding()
        self.d_k = d // num_heads
        
        # Weight initialization
        self.W_Q = [np.random.randn(self.d, self.d_k) for _ in range(self.num_heads)]
        self.W_K = [np.random.randn(self.d, self.d_k) for _ in range(self.num_heads)]
        self.W_V = [np.random.randn(self.d, self.d_k) for _ in range(self.num_heads)]
        self.W_0 = np.random.randn(self.num_heads * self.d_k, self.d)

    def _positionalEncoding(self):
        P = np.zeros((self.seq_len, self.d))
        for k in range(self.seq_len):
            for i in range(self.d // 2):
                denominator = np.power(self.n, 2 * i / self.d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _attention(self, Q, K, V):
        score = (Q @ K.T) / np.sqrt(self.d_k)
        attention_weights = self._softmax(score)
        return attention_weights @ V
    
    def _attention_output(self):
        Q = K = V = self.P
        heads = []
        
        for i in range(self.num_heads):
            Q_i = Q @ self.W_Q[i]
            K_i = K @ self.W_K[i]
            V_i = V @ self.W_V[i]
            
            head = self._attention(Q_i, K_i, V_i)
            heads.append(head)
        
        multi_head_output = np.concatenate(heads, axis=-1)
        return multi_head_output @ self.W_0
    
# Example usage
seq_len = 6   # Sequence length
d = 512       # Dimension
num_heads = 8 # Number of attention heads

transformer_attention = MultiheadAttention(seq_len, d, num_heads)
attention_output = transformer_attention._attention_output()
print("Positional Encoding:\n", transformer_attention.P.shape)
print("Attention Output:\n", attention_output.shape)
