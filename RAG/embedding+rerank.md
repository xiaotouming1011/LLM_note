### RAG的embedding模型怎么选（MTEB: **Massive Text Embedding Benchmark**,中文看C-MTEB）



**BGE-M3（BAAI 智源）**：目前中文场景的首选。支持中英多语言，最大 8192 token 的上下文窗口，同时支持稠密向量、稀疏向量和 ColBERT 式多向量检索三种模式。在 MTEB 中文榜单上长期稳居前列。如果你不知道选什么，无脑选 BGE-M3 不会错。

**BGE-large-zh（BAAI 智源）**：专注中文的大尺寸版本，在纯中文场景下精度略高于 M3，但不支持多语言，上下文窗口也只有 512 token。适合纯中文且文档较短的场景。

**GTE-multilingual-base（阿里达摩院）**：阿里出品的多语言 Embedding 模型，在 MTEB 多语言榜单上表现很强。跟 BGE-M3 是直接竞品关系，两者在多语言场景下各有胜负。如果你面的是阿里，了解 GTE 是基本功。

**E5-small/base/large（微软）**：微软出品，特点是有从 small 到 large 的完整尺寸梯度，small 版本只有 33M 参数，特别适合资源紧张或需要部署到边缘设备的场景。精度比 BGE 略低，但推理速度快很多。

**Jina Embeddings v2（Jina AI）**：最大亮点是支持 8K token 的超长上下文。如果你的文档 chunk 特别长（比如整段法律条文或完整的技术文档章节），其他模型可能截断，Jina v2 能全部吃进去。

**MiniLM（微软）**：极致轻量级，速度最快，适合对延迟要求极高或大批量处理的场景。精度是这几个里最低的，但胜在快。

### **怎么选？一句话决策**

- **一般情况不知道选什么 → BGE-M3**，不会错
- **纯中文或中英混合 → BGE-M3 或 BGE-large-zh**
- **多语言场景 → GTE-multilingual-base 或 BGE-M3**，看 MTEB 榜单最新排名
- **资源紧张 / 边缘设备 → E5-small 或 MiniLM**
- **长文档 ≥ 8K token → Jina Embeddings v2**

我们的场景是电商平台场景下的退款政策文档，chunk 长度控制在 500 token 以内，对多语言没有需求，所以选了 BGE-M3。也评估过 GTE-multilingual-base，在我们的测试集上 BGE-M3 的 MRR 高了 3 个百分点，所以最终选了 BGE。（MRR:Mean Reciprocal Rank,模型 / 检索器会返回一堆候选答案,把正确答案排得越靠前，得分越高）

<img src="/Users/anji/Library/Application Support/typora-user-images/截屏2026-03-23 17.44.29.png" alt="截屏2026-03-23 17.44.29" style="zoom: 33%;" />

看MTEB,**不要只看总分。** MTEB 的总分是多个任务的平均值，包括分类、聚类、句对匹配等。RAG 场景最关心的是 **Retrieval** 子任务的得分，要单独筛选看这一项。

从三个维度评估：语种支持（中文/多语言）、上下文长度（chunk 长度匹配）、部署资源（GPU 显存和推理速度）。

### 5.**Rerank 模型怎么选？怎么跟 Embedding 配？**

**Bi-Encoder（Embedding 模型）**：query 和文档各自独立编码成向量，然后算余弦相似度。优点是快——文档向量离线算好，查询时只算一次 query 向量。缺点是 query 和文档之间没有交互，模型看不到它们放在一起时的语义关系。

**Cross-Encoder（Rerank 模型）**：把 query 和文档拼在一起，作为一个整体输入 Transformer，模型能看到两者的完整交互，输出一个相关性分数。精度远高于 Bi-Encoder，但代价是每一对 query-doc 都要做一次完整的模型推理，速度慢很多。

Cross-Encoder 更能理解名称之间的事实关系。

两阶段检索是标配：**Bi-Encoder 负责从百万文档中快速召回 Top 100，Cross-Encoder 负责对这 100 条做精排取 Top 5 给 LLM。**



### **主流 Rerank 模型和搭配方案**

Rerank 模型目前主流的选择：

**BGE-Reranker-base/large（BAAI 智源）**：跟 BGE Embedding 同门，中文效果好，是目前用得最多的开源 Rerank 模型。

**GTE-multilingual-reranker（阿里达摩院）**：阿里出品，多语言场景表现强，适合跟 GTE Embedding 搭配使用。

**MiniLM-L6-cross-encoder（微软）**：轻量级 Cross-Encoder，适合 GPU 资源紧张时做 batch 推理，速度快但精度比 BGE-Reranker 低一些。

**Jina-ColBERT-v2（Jina AI）**：基于 ColBERT 架构的 Late Interaction 模型，介于 Bi-Encoder 和 Cross-Encoder 之间。精度接近 Cross-Encoder 但速度快很多，适合长文档场景。

##### **四种经典搭配方案**（**Embedding 和 Rerank 尽量选同一系列的，因为它们在训练时的数据分布和语义空间更一致，搭配效果最好。**）

- **经典流水线**：BGE-base 检索 Top 100 → BGE-Reranker-base 精排
- **多语言场景**：GTE-multilingual-base + GTE-multilingual-reranker
- **GPU 紧张**：E5-small + MiniLM-L6-cross-encoder（batch 推理）
- **长文档 / 8K**：Jina-embeddings-v2 + Jina-ColBERT-v2，段内匹配更稳

