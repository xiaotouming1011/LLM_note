# Embedding模型选择

**V1：text-embedding-3-large（通用顶尖模型）**

MTEB 榜单排名靠前，OpenAI 出品，理论上应该不差。实际上线后：保险专业术语查询的 Recall@5 只有 **0.54**，比整体召回率低 18 个百分点。

根因是通用模型对领域专有名词有系统性偏差。"保单现金价值"和"退保金"在保险业务里是两个不同概念，但 text-embedding-3-large 把两者的向量距离拉得非常近（余弦相似度 0.91），召回时严重混淆。分析 50 个 badcase，发现三类系统性错误：

1. **专业术语泛化**

   ：把"保单现金价值"理解成"资产价值"，召回了大量投资理财文档

2. **同义词无法区分**

   ："退保"（动作）和"退保金"（结果）在向量空间里几乎重叠

3. **上下文依赖不足**

   ："重疾险"在"投保"和"理赔"两个场景的相关文档完全不同，但模型把两者混在一起召回

**V2：BGE-large-zh（中文开源模型）**

换成智源开源的中文 Embedding 模型，专业术语召回率从 0.54 提升到 **0.68**，好了一些，但还有 32% 的专业术语仍然被错误关联到通用金融概念上。

BGE 虽然是中文模型，但保险领域在预训练语料里占比太低，对保险术语的理解还停留在"字面意思"层面。

**V3：BGE 领域微调**

用 800 对人工标注正负例对 BGE 做对比学习微调，专业术语召回率从 0.68 提升到 **0.89**，整体 Recall@5 从 0.76 提升到 **0.83**。

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
 from torch.utils.data import DataLoader

 model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

 # 关键：负例要选语义相近但业务含义不同的样本
 train_examples = [
     # 正例：同一概念不同表述，label = 1.0
     InputExample(texts=["保单现金价值怎么算", "保险合同的退保可得金额"], label=1.0),
     # 困难负例：词汇近，含义不同,label = 0.0
     InputExample(texts=["保单现金价值怎么算", "退保金计算公式"],       label=0.0),
     InputExample(texts=["保单现金价值怎么算", "公司净资产估值方法"],    label=0.0),
 ]

 train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
 model.fit(
     train_objectives=[(train_dataloader, losses.CosineSimilarityLoss(model))],
     epochs=3,
     warmup_steps=100,
     optimizer_params={"lr": 2e-5}
 )
 model.save("bge-large-zh-insurance-finetuned")

 # ⚠️ 微调后必须重新 Embedding 全部文档、重建向量索引
 # 微调改变了向量空间分布，旧向量和新 query 向量不在同一空间
```

![截屏2026-04-10 17.52.05](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-10 17.52.05.png)