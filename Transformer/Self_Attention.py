import numpy as np

class Self_Attention:
    def __init__(self,seq_len,d,n=1000):
        self.seq_len=seq_len
        self.d=d
        self.n=n
        self.P=self._PositionalEncoding()
    def _PositionalEncoding(self):
        P=np.zeros((self.seq_len,self.d))
        for k in range(self.seq_len):
            for i in np.arange(int(self.d/2)):
                denominator=np.power(self.n,2*i/self.d)
                P[k,2*i]=np.sin(k/denominator)
                P[k,2*i+1]=np.cos(k/denominator)
        return P
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    def _attention(self):
        Q=K=V=self.P
        return self._softmax((Q@K.T)/np.sqrt(self.d))@V
    
Attention=Self_Attention(6,512)
