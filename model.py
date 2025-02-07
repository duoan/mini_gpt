import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class CausalSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        self.qkv_proj = nn.Dense(3 * self.embed_dim, dtype=self.dtype)
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, mask, deterministic: bool = True):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads
        )
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(
            embed_dim // self.num_heads
        )
        attn_weights = jnp.where(mask, attn_weights, -1e9)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout_layer(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        self.attn = CausalSelfAttention(
            self.embed_dim, self.num_heads, self.dropout, self.dtype
        )
        self.fc1 = nn.Dense(self.mlp_dim, dtype=self.dtype)
        self.activation = nn.gelu
        self.dropout1 = nn.Dropout(rate=self.dropout)
        self.fc2 = nn.Dense(self.embed_dim, dtype=self.dtype)
        self.dropout2 = nn.Dropout(rate=self.dropout)
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, mask, deterministic: bool = True):
        attn_out = self.attn(self.norm1(x), mask, deterministic=deterministic)
        x = x + self.dropout_layer(attn_out, deterministic=deterministic)
        hidden = self.fc1(self.norm2(x))
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden, deterministic=deterministic)
        hidden = self.fc2(hidden)
        mlp_out = self.dropout2(hidden, deterministic=deterministic)
        x = x + self.dropout_layer(mlp_out, deterministic=deterministic)
        return x


class CausalGPT(nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_seq_len: int = 1024
    dropout: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        self.token_embed = nn.Embed(self.vocab_size, self.embed_dim, dtype=self.dtype)
        # 使用可学习的位置编码
        self.pos_embed = nn.Embed(self.max_seq_len, self.embed_dim, dtype=self.dtype)
        self.blocks = [
            TransformerBlock(
                self.embed_dim, self.num_heads, self.mlp_dim, self.dropout, self.dtype
            )
            for _ in range(self.num_layers)
        ]
        self.ln_f = nn.LayerNorm(dtype=self.dtype)
        self.out_proj = nn.Dense(self.vocab_size, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, input_ids, deterministic: bool = True):
        batch_size, seq_len = input_ids.shape
        # 创建位置索引
        positions = jnp.arange(seq_len)[None, :]
        # 获取位置编码
        pos_embeddings = self.pos_embed(positions)

        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        x = self.token_embed(input_ids) + pos_embeddings
        x = self.dropout_layer(x, deterministic=deterministic)

        for block in self.blocks:
            x = block(x, mask, deterministic=deterministic)

        x = self.ln_f(x)
        return self.out_proj(x)
