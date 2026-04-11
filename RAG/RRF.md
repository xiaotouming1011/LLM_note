### 向量检索已经很强了，为什么还需要 BM25？

https://mp.weixin.qq.com/s/wZeLHeOqkHDM8ZRStxzCkg



BM25的优势恰好是向量检索的劣势

### 答案藏在 Embedding 的工作机制里。

把一段文字映射到向量空间，模型做的事情是把语义"压缩"到一个固定维度的点。这个压缩过程对长文本相对友好——长文本包含的语义信息多，各个维度分工相对清晰。但对于==短 query，这个压缩过程会出问题==。

==短 query 的语义密度高，一个词的权重就很重==。但 Embedding 空间里，语义相近的向量会聚集在一起，而"语义相近"这件事是由模型训练数据决定的。如果训练语料里某个领域的词汇稀少，那这些词的向量表示就会被"拉向"语义相近但实际不同的概念。

用具体例子说：在保险领域，"等待期"和"观察期"在业务上意思完全不同，但在通用 Embedding 模型的向量空间里，它们的余弦相似度可能高达 0.91。你搜"28天等待期"，可能把"90天观察期"的文档都召回来。

## **三类向量检索的死穴**

1.专有名词和缩写

​	Embedding 空间里没有语义邻居，或者邻居是错误的。向量检索在这类 query 上经常召回驴唇不对马嘴的文档。

2.精确的数字

​	比如"28天等待期"、"180天保障期限"、"赔付上限300万"。数字本身在 Embedding 里几乎没有区分度，"28天"和"90天"的向量距离可能比"28天"和"等待期"的距离还要近。

3.低频专业术语

​	知识库里只出现 1-2 次的术语，比如某款产品独有的条款名称。这类词在训练语料里几乎不存在，Embedding 模型对它的表示能力极弱。

### **BM25 在这三类场景上天然更强**

BM25 是基于词频的精确匹配，它不做语义理解，只看词有没有出现、出现了多少次。这恰好是向量检索的软肋所在：精确匹配类问题上，BM25 表现稳定可靠。

![截屏2026-04-11 20.21.23](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-11 20.21.23.png)

# **BM25 的核心原理**



## **从 TF-IDF 到 BM25 的演化**

 TF（Term Frequency）词频,  IDF（Inverse Document Frequency）逆文档频率

TF-IDF 的核心公式是：

```
score = TF(t, d) * IDF(t)
```

TF 是词 t 在文档 d 中出现的频率，IDF 是词 t 在整个语料库中的逆文档频率（出现文档越少，IDF 越高）。

==TF-IDF 有两个明显的问题：==

第一，TF 没有上限。如果一个词在文档里出现了 1 次和出现了 100 次，TF-IDF 给的分数相差 100 倍。但实际上，出现 5 次和出现 100 次的文档，相关性差距远没有这么大。这就是 **TF 饱和问题**。

第二，长文档天然吃亏。一篇 10000 字的文档，词出现的绝对次数会比 1000 字的文档高很多，但不代表它更相关。这就是**文档长度偏置问题**。



==BM25 解决了这两个问题。BM25 的完整公式是：==

```python
score(q, d) = Σ IDF(t) * [TF(t,d) * (k1 + 1)] / [TF(t,d) + k1 * (1 - b + b * |d| / avgdl)]
```

两个关键参数：

**k1 参数**：控制词频饱和速度。k1 越小，TF 饱和越快；k1 越大，词频的影响越线性。常用默认值 k1=1.5，意味着==词频对分数的贡献存在上限==，不会无限增长。

==**b 参数**：控制文档长度归一化程度==。b=0 表示完全不考虑文档长度，b=1 表示完全按文档长度归一化。常用默认值 b=0.75，在两个极端之间取平衡。

==**BM25 的局限：语义盲区**==

BM25 的致命弱点是它不理解语义。"车祸"和"交通事故"在 BM25 看来是两个完全不同的词，不存在任何关联。用户用"车祸"搜索，BM25 不会召回只包含"交通事故"的文档。

这就是为什么 BM25 需要和向量检索配合——向量检索负责语义泛化，BM25 负责精确匹配，两者互补。

**BM25 的代码实现**

```python
from rank_bm25 import BM25Okapi
 import jieba

 class BM25Retriever:
     def __init__(self, documents: list[str]):
         # 对文档进行中文分词，BM25 需要以词为单位工作
         self.tokenized_docs = [
             list(jieba.cut(doc)) for doc in documents
         ]
         # 初始化 BM25 模型，k1 和 b 使用默认值
         self.bm25 = BM25Okapi(
             self.tokenized_docs,
             k1=1.5,   # 词频饱和参数
             b=0.75    # 文档长度归一化参数
         )
         self.documents = documents

     def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
         # 对查询进行分词
         tokenized_query = list(jieba.cut(query))

         # 获取 BM25 分数
         scores = self.bm25.get_scores(tokenized_query)

         # 按分数排序，取 top_k
         ranked_indices = sorted(
             range(len(scores)),
             key=lambda i: scores[i],
             reverse=True
         )[:top_k]

         return [
             {
                 "doc_id": idx,
                 "document": self.documents[idx],
                 "score": scores[idx],
                 "rank": rank + 1
             }
             for rank, idx in enumerate(ranked_indices)
         ]
```

## **两种融合策略：加权融合 vs RRF**

==**策略一：加权融合（Weighted Sum）**==

最直觉的方法：给两个检索结果的分数加权求和。

```
final_score = α * bm25_score + (1 - α) * vector_score
```

α 是 BM25 的权重，(1-α) 是向量检索的权重。

这个方法的问题是：**BM25 的分数和向量检索的余弦相似度量纲完全不同**。BM25 分数的范围取决于文档数量和词频，可能是 0-50 甚至更高；余弦相似度的范围是 0-1。直接加权求和，BM25 分数会完全压制向量分数。

解决方法是先归一化。常用的归一化方式是 min-max 归一化：

```python
def normalize_scores(scores: list[float]) -> list[float]:
     """将分数归一化到 [0, 1] 区间"""
     min_score = min(scores)
     max_score = max(scores)
     if max_score == min_score:
         return [1.0] * len(scores)
     return [(s - min_score) / (max_score - min_score) for s in scores]
```

但归一化也带来新问题：归一化后，最低分的文档分数变成 0，最高分的变成 1，**相对差异被压缩了**，不同 query 之间的分数可比性降低。

**策略二：RRF（Reciprocal Rank Fusion）**

RRF 的思路完全不同。它不用分数，只用排名。

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

对于文档 d，遍历所有检索列表，找到它在每个列表中的排名 rank_i，然后把 1/(k + rank_i) 加起来。

这个公式的好处是：**完全不需要归一化**。不管原始分数量纲是什么，只要排名是正整数，RRF 就能正常工作。

==**k 值的作用是什么？**==

k 值控制的是高排名文档的优势大小。

假设 k=0，排名第一的文档得 1/1=1.0，排名第二的得 1/2=0.5，排名第十的得 1/10=0.1。排名第一和排名第二的分数差距是 0.5。

假设 k=60，排名第一的文档得 1/61≈0.0164，排名第二的得 1/62≈0.0161，排名第十的得 1/70≈0.0143。排名第一和排名第二的分数差距只有 0.0003。

k 越大，各个排名之间的分数差距越小，排名靠前的文档优势越弱，融合结果越"平均主义"。k 越小，高排名文档的优势越大。

**k=60 是怎么来的？不是拍脑袋**

k=60 来自 Cormack 等人 2009 年发表的论文《Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods》。他们在 TREC 评测数据集上做了系统的实验，扫描了 k 从 1 到 100 的不同取值，发现 k=60 在多数场景下能达到最优或接近最优的融合效果。

这是有实验依据的，不是经验值。但这个"最优"是在 TREC 通用评测数据集上得到的结论，不一定适用于你的具体业务场景。

**RRF 的完整实现**

```
from collections import defaultdict

 def reciprocal_rank_fusion(
     result_lists: list[list[dict]],
     k: int = 60
 ) -> list[dict]:
     """
     RRF 融合多路检索结果

     参数:
         result_lists: 多路检索结果，每路是一个按相关性排序的文档列表
                      每个文档是包含 doc_id 和其他信息的字典
         k: RRF 平滑参数，默认 60

     返回:
         融合后按 RRF 分数排序的文档列表
     """
     rrf_scores = defaultdict(float)
     doc_store = {}  # 存储文档原始信息

     for result_list in result_lists:
         for rank, doc in enumerate(result_list, start=1):
             doc_id = doc["doc_id"]
             # 核心公式：1 / (k + rank)
             rrf_scores[doc_id] += 1.0 / (k + rank)
             # 保存文档信息，后续返回时用
             doc_store[doc_id] = doc

     # 按 RRF 分数降序排列
     sorted_doc_ids = sorted(
         rrf_scores.keys(),
         key=lambda doc_id: rrf_scores[doc_id],
         reverse=True
     )

     return [
         {
             **doc_store[doc_id],
             "rrf_score": rrf_scores[doc_id]
         }
         for doc_id in sorted_doc_ids
     ]


 def hybrid_retrieve(
     query: str,
     bm25_retriever: BM25Retriever,
     vector_retriever,  # 向量检索器
     top_k: int = 10,
     rrf_k: int = 60
 ) -> list[dict]:
     """混合检索主函数"""
     # 两路检索可以并行执行（见第五节）
     bm25_results = bm25_retriever.retrieve(query, top_k=top_k * 2)
     vector_results = vector_retriever.retrieve(query, top_k=top_k * 2)

     # RRF 融合
     fused_results = reciprocal_rank_fusion(
         [bm25_results, vector_results],
         k=rrf_k
     )

     return fused_results[:top_k]
```

**什么时候选加权融合，什么时候选 RRF？**

选加权融合的场景：两路检索分数都已经归一化到相同量纲（比如都是 0-1 的概率分数），且只有两路检索结果需要融合。

选 RRF 的场景：==分数量纲差异大，或者需要融合三路及以上的检索结果==（比如加了 SPLADE 稀疏向量模型）。在大多数工程场景里，RRF 更推荐，因为它不需要归一化，对超参数不敏感。

## **消融实验的标准做法**

消融实验的基本逻辑：准备一批带有已知正确答案的问答对（评估集），固定其他变量，只改变一个参数，观察指标（Recall@5 或 MRR）的变化。

```python
import numpy as np
 from itertools import product

 def ablation_experiment(
     eval_questions: list[dict],  # 格式：[{"query": "...", "relevant_doc_ids": [...]}]
     bm25_retriever,
     vector_retriever,
     top_k: int = 5
 ) -> dict:
     """
     消融实验：扫描不同参数组合，找最优配置

     eval_questions 中每条数据包含 query 和对应的正确文档 ID 列表
     """
     # 参数搜索空间
     alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # BM25 权重
     rrf_k_values = [10, 20, 30, 40, 60, 80, 100]  # RRF k 值

     results = {}

     # 预先获取两路检索结果，避免重复计算
     print("预计算检索结果...")
     bm25_cache = {}
     vector_cache = {}
     for item in eval_questions:
         query = item["query"]
         bm25_cache[query] = bm25_retriever.retrieve(query, top_k=top_k * 3)
         vector_cache[query] = vector_retriever.retrieve(query, top_k=top_k * 3)

     # 扫描 RRF k 值
     print("扫描 RRF k 值...")
     for k in rrf_k_values:
         recall_list = []
         for item in eval_questions:
             query = item["query"]
             relevant_ids = set(item["relevant_doc_ids"])

             # RRF 融合
             fused = reciprocal_rank_fusion(
                 [bm25_cache[query], vector_cache[query]],
                 k=k
             )
             retrieved_ids = {doc["doc_id"] for doc in fused[:top_k]}

             # 计算 Recall@top_k
             recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
             recall_list.append(recall)

         avg_recall = np.mean(recall_list)
         results[f"RRF_k={k}"] = avg_recall
         print(f"  RRF k={k:3d}: Recall@{top_k} = {avg_recall:.4f}")

     return results
```

![截屏2026-04-11 20.52.59](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-11 20.52.59.png)

结论：

**α=0.3（BM25权重0.3，向量权重0.7）效果最好。** 这和直觉一致——我们的问题以语义理解类为主，精确条款查找类只占约30%，所以向量权重更高是合理的。

**RRF k=60 在我们的场景下确实是最优或接近最优。** 这验证了论文的结论，但不代表所有场景都如此。

知识库里专业术语和精确数字密度高的场景，BM25 权重应该调高；问题比较口语化、需要语义泛化的场景，向量权重应该更高。这是需要根据业务场景实验出来的，没有万能参数。



**工程落地细节**

### 1.BM25 检索和向量检索串行执行

先等 BM25 出结果，再发向量检索请求。这样总延迟是两者之和。

正确做法是异步并行，两路检索同时发出，等两路都完成后再做融合。

### 2.**索引同步更新**

知识库更新（新增或删除文档）时，需要同步更新两个索引：向量索引和 BM25 倒排索引。这两个索引的更新机制完全不同：

向量索引（如 Faiss 或 Milvus）支持增量添加向量，但某些索引类型（如 IVF）在大量增量后需要重建才能保证检索质量。

BM25 索引（rank_bm25 库）不支持增量更新，需要用新文档集合重新构建。在文档量不大（10万以内）的场景下，全量重建的耗时在可接受范围内，可以做成每天凌晨定时重建的任务。



**稀疏+稠密混合的新方向：SPLADE**

值得一提的是，混合检索的前沿方向正在朝着更深层次的融合发展。SPLADE（SParse Lexical AnD Expansion）是一类稀疏向量模型，它把 BM25 的精确匹配能力和向量的语义泛化能力结合到一个模型里。

SPLADE 的输出是一个高维稀疏向量（维度等于词汇表大小），每个维度对应一个词的权重，但这个权重是通过神经网络学到的，不是简单的词频。这样既保留了精确匹配的能力，又通过神经网络实现了同义词扩展。

在我们项目的初步测试中，SPLADE + 向量检索的组合，在不需要独立维护 BM25 索引的情况下，Recall@5 达到 0.87，接近 BM25+向量 RRF 的 0.89，工程复杂度更低。这是未来值得关注的方向，但目前中文 SPLADE 模型的成熟度还不如英文，需要根据场景谨慎评估。





























