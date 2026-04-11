浮点运算（FLOPs）（约等于）= 6✖️参数量✖️Token数量

前向传播一次：2✖️参数量（乘法+加法）

反向传播：计算梯度约为前向传播的 2 倍工作量（还要计算梯度），即 4× 参数量。

训练 Token 数 ≈ 20 × 参数量

![截屏2026-04-10 21.01.13](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-10 21.01.13.png)

```python
激活显存 ∝ batch_size × max_seq_len × hidden_size × layers
```

