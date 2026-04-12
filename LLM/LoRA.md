# LoRA(Low-Rank Adaptation，低秩适应)

LoRA 是一个通用方法，但它通常应用于 Attention 层

矩阵的秩是矩阵的行“所在”的维数。

LoRA 是一种常见的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法。相比全参数微调，它只更新少量新增参数，而保留原始模型主体权重不变，因此训练成本更低，也更适合做垂直场景适配。
它的核心思想是在原有权重矩阵旁引入低秩增量分支，仅训练这部分低秩参数，从而用较小代价完成能力迁移。相关实现可见 `model_lora.py` 和 `train_lora.py`，整个流程均为纯手写实现，不依赖第三方封装。