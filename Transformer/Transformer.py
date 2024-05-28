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
        
        
        # Initialize weight matrices for feed-forward network
        self.W_1 = np.random.randn(d, d * 4)  # First layer: d -> d * 4
        self.W_2 = np.random.randn(d * 4, d)  # Second layer: d * 4 -> d
    
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
    
    def _attention(self, Q, K, V, mask=None):
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores += mask
        attention_weights = self._softmax(scores)
        return attention_weights @ V
    
    def multi_head_attention(self, Q, K, V, mask=None):
        heads = []
        
        # Compute attention for each head
        for i in range(self.num_heads):
            Q_i = Q @ self.W_Q[i]
            K_i = K @ self.W_K[i]
            V_i = V @ self.W_V[i]
            head = self._attention(Q_i, K_i, V_i, mask)
            heads.append(head)
        
        # Concatenate all heads and apply the final linear transformation
        multi_head_output = np.concatenate(heads, axis=-1)
        return multi_head_output @ self.W_O

    def multi_head_attention_decoder(self, Q, K, V, mask=None):
        heads = []
        
        # Compute attention for each head
        for i in range(self.num_heads):
            Q_i = Q @ self.W_Q[i]
            K_i = K @ self.W_K[i]
            V_i = V @ self.W_V[i]
            head = self._attention(Q_i, K_i, V_i, mask)
            heads.append(head)
        
        # Concatenate all heads and apply the final linear transformation
        multi_head_output = np.concatenate(heads, axis=-1)
        return multi_head_output @ self.W_O

    def layer_norm(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        std = np.std(X, axis=-1, keepdims=True)
        return (X - mean) / (std + 1e-6)
    
    def feed_forward(self, X):
        return np.maximum(0, X @ self.W_1) @ self.W_2

    def encoder_layer(self, X):
        # Multi-head attention sublayer with Add & Norm
        Q = K = V = X + self.P
        attn_output = self.multi_head_attention(Q, K, V)
        attn_output = self.layer_norm(X + attn_output)
        
        # Feed-forward sublayer with Add & Norm
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm(attn_output + ff_output)
        
        return ff_output
    
    def _generate_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -np.inf
        mask[np.isnan(mask)] = 0
        return mask

    def decoder_layer(self, X, encoder_output):
        # Masked multi-head attention sublayer with Add & Norm
        mask = self._generate_mask(self.seq_len)
        Q = K = V = X + self.P
        masked_attn_output = self.multi_head_attention(Q, K, V, mask)
        masked_attn_output = self.layer_norm(X + masked_attn_output)
        
        # Multi-head attention with encoder output
        Q_dec = masked_attn_output
        K_enc = V_enc = encoder_output 
        attn_output = self.multi_head_attention_decoder(Q_dec, K_enc, V_enc)
        attn_output = self.layer_norm(masked_attn_output + attn_output)
        
        # Feed-forward sublayer with Add & Norm
        ff_output = self.feed_forward(attn_output)
        ff_output = self.layer_norm(attn_output + ff_output)
        
        return ff_output

seq_len = 10
d = 512
num_heads = 8

transformer = TransformerAttention(seq_len, d, num_heads)
X = np.random.randn(seq_len, d)
encoder_output = transformer.encoder_layer(X)
decoder_output = transformer.decoder_layer(X, encoder_output)
print(encoder_output.shape)
print(decoder_output.shape)
