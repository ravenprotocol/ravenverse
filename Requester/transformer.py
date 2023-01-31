import numpy as np
import ravop as R
from ravdl.v2 import Functional
from ravdl.v2.layers import *
import math


class NewGeLU(CustomLayer):
    def __init__(self):
        super().__init__()
        self.half = R.t(0.5)
        self.tanh = Activation('tanh')
        self.one = R.t(1)
        self.c = R.t(math.sqrt(2.0/math.pi))
        self.norm_const = R.t(0.044715)
        self.pow = Power()
        self.mul1 = Multiply()
        self.mul2 = Multiply()
        self.mul3 = Multiply()
        self.mul4 = Multiply()
        self.add1 = Add()
        self.add2 = Add()

    def _forward_pass_call(self, x, training=True):
        # 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        pow_op = self.pow._forward_pass(x, power=3)
        mul1_op = self.mul1._forward_pass(self.norm_const, pow_op)
        add1_op = self.add1._forward_pass(x,mul1_op)
        mul2_op = self.mul2._forward_pass(self.c, add1_op)
        tanh_op = self.tanh._forward_pass(mul2_op)
        add2_op = self.add2._forward_pass(self.one, tanh_op)
        mul3_op = self.mul3._forward_pass(x,add2_op)
        mul4_op = self.mul4._forward_pass(self.half, mul3_op)
        return mul4_op


class ScaledDotProductAttention(CustomLayer):
    def __init__(self, depth) -> None:
        super().__init__()
        self.transpose = Transpose()
        self.dot1 = Dot()
        self.dot2 = Dot()
        self.depth = np.cast['float32'](depth)
        self.depth_sqrt = R.t(np.sqrt(self.depth))
        self.add1 = Add()
        self.div1 = Division()
        self.softmax = Activation('softmax')

    def _forward_pass_call(self, q, k, v, training=True, mask=None):
        k_transpose = self.transpose._forward_pass(k, axes=(-2,-1))#axes=(0,1,3,2))
        matmul_qk = self.dot1._forward_pass(q, k_transpose)  # (..., seq_len_q, seq_len_k)
        scaled_attention_logits = self.div1._forward_pass(matmul_qk, self.depth_sqrt)

        if mask is not None:
            scaled_attention_logits = self.add1._forward_pass(scaled_attention_logits, R.t(mask * -1e9))

        attention_weights = self.softmax._forward_pass(scaled_attention_logits) 
        output = self.dot2._forward_pass(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output
        
        


class MultiHeadAttention(CustomLayer):
    def __init__(self, n_embed, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed
        assert n_embed % self.n_heads == 0
        self.depth = n_embed // self.n_heads

        self.wq = Dense(n_embed, input_shape=(n_embed,))
        self.wk = Dense(n_embed, input_shape=(n_embed,))
        self.wv = Dense(n_embed, input_shape=(n_embed,))
        self.c_proj = Dense(n_embed)

        self.reshape_split_heads1 = Reshape()
        self.transpose_split_heads1 = Transpose()
        self.reshape_split_heads2 = Reshape()
        self.transpose_split_heads2 = Transpose()
        self.reshape_split_heads3 = Reshape()
        self.transpose_split_heads3 = Transpose()

        self.transpose_scaled_attn = Transpose()
        self.scaled_dot_attn = ScaledDotProductAttention(self.depth)
        self.concat_attention_reshape = Reshape(contiguous="True")

    # def split_heads(self, x, batch_size):
    #     x = self.reshape_split_heads._forward_pass(x, shape=(batch_size, -1, self.n_heads, self.depth))
    #     x = self.transpose_split_heads._forward_pass(x, axes=[0, 2, 1, 3])
    #     return x

    def _forward_pass_call(self, input, training=True, batch_size=None, mask=None):
        q = self.wq._forward_pass(input)
        k = self.wk._forward_pass(input)
        v = self.wv._forward_pass(input)

        q = self.reshape_split_heads1._forward_pass(q, shape=(batch_size, -1, self.n_heads, self.depth))
        q = self.transpose_split_heads1._forward_pass(q, axes=(1,2))#[0, 2, 1, 3])
        k = self.reshape_split_heads2._forward_pass(k, shape=(batch_size, -1, self.n_heads, self.depth))
        k = self.transpose_split_heads2._forward_pass(k, axes=(1,2))#axes=[0, 2, 1, 3])
        v = self.reshape_split_heads3._forward_pass(v, shape=(batch_size, -1, self.n_heads, self.depth))
        v = self.transpose_split_heads3._forward_pass(v, axes=(1,2))#axes=[0, 2, 1, 3])

        # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention = self.scaled_dot_attn._forward_pass(q, k, v, training=training, mask=mask)
        scaled_attention = self.transpose_scaled_attn._forward_pass(scaled_attention, axes=(1,2))#[0, 2, 1, 3])  # (batch_size, seq_len_q,      num_heads, depth)

        concat_attention = self.concat_attention_reshape._forward_pass(scaled_attention, shape=(batch_size, -1, self.n_embed)) # (batch_size, seq_len_q, d_model)

        output = self.c_proj._forward_pass(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output

class Block(CustomLayer):
    def __init__(self, n_embed, n_heads) -> None:
        super().__init__()
        self.ln_1 = LayerNormalization(normalized_shape=n_embed)
        self.attn = MultiHeadAttention(n_embed=n_embed, n_heads=n_heads)
        self.ln_2 = LayerNormalization(normalized_shape=n_embed)
        self.c_fc = Dense(4*n_embed)
        self.c_proj = Dense(n_embed)
        self.gelu = NewGeLU()
        self.drpt = Dropout()

        self.add1 = Add()
        self.add2 = Add()

    def _forward_pass_call(self, x, training=True, batch_size=None, mask=None):
        input = x
        x = self.ln_1._forward_pass(x, training=training)
        x = self.attn._forward_pass(x, training=training, batch_size=batch_size, mask=mask)
        x = self.add1._forward_pass(input,x)
        x_ = x
        x = self.ln_2._forward_pass(x, training=training)
        x = self.c_fc._forward_pass(x)
        x = self.gelu._forward_pass(x)
        x = self.c_proj._forward_pass(x)
        x = self.drpt._forward_pass(x, training=training)
        x = self.add2._forward_pass(x_,x)

        return x

class GPT(Functional):
    def __init__(self, vocab_size, embed_dim, block_size, n_heads, n_layer, seq_length, optimizer, embedding_weights=None):
        super().__init__()
        self.wte = Embedding(vocab_size=vocab_size, embed_dim=embed_dim, initial_W=embedding_weights)
        self.wpe = Embedding(block_size, embed_dim)
        self.drop = Dropout(0.1)
        self.h = [Block(n_embed=embed_dim, n_heads=n_heads) for _ in range(n_layer)]
        self.ln_f = LayerNormalization(normalized_shape=embed_dim)
        self.lm_head = Dense(vocab_size, use_bias='False')
        self.add = Add()
        # self.sft = Activation('softmax')
        self.seq_length = seq_length
        self.pos = R.t(np.expand_dims(np.arange(0,self.seq_length),axis=0)) # shape (1, seq_length)
        self.initialize_params(optimizer)

    def _forward_pass_call(self, inputs, training=True, batch_size=None, mask=None):
        if not training:
            self.pos = R.t(np.expand_dims(np.arange(0,self.seq_length),axis=0)) # shape (1, seq_length)
        tok_emb = self.wte._forward_pass(inputs) # token embeddings of shape (b, seq_length, n_embd)
        pos_emb = self.wpe._forward_pass(self.pos) # position embeddings of shape (1, seq_length, n_embd)

        total_emb = self.add._forward_pass(tok_emb, pos_emb)

        x = self.drop._forward_pass(total_emb, training=training)

        for block in self.h:
            x = block._forward_pass(x, training=training, batch_size=batch_size, mask=mask)

        x = self.ln_f._forward_pass(x)
        logits = self.lm_head._forward_pass(x)
        # out = self.sft._forward_pass(logits)
        out = logits
        return out
