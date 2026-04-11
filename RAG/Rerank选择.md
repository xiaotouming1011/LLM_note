# **Rerank **

https://mp.weixin.qq.com/s/1GZtibu07K2rzhGF-PZJ2Q

![截屏2026-04-11 18.40.53](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-11 18.40.53.png)

**GPT-4 为什么效果最差**，

![截屏2026-04-10 17.56.42](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-10 17.56.42.png)

GPT-4 效果最差的三个原因：

**过度推理**：GPT-4 会"脑补"文档里没有的信息，在判断相关性时引入幻觉，导致把无关文档排到高位。

**Prompt 敏感**：同一个 query，换一种 Prompt 写法，Top5 文档会有 40% 不同。Cross-Encoder 的输出是确定性的。

**延迟不可接受**：单次 Rerank 30 个文档需要 3-5 秒，用户等不了，也根本没法上线。

### Cross-Encoder 的核心优势是联合编码：

Cross-Encoder 比 Bi-Encoder（Embedding 模型）准的根本原因：Bi-Encoder 独立编码 query 和 doc，用余弦相似度近似相关性；Cross-Encoder 把两者拼接后联合 Attention，计算真正的语义相关性。代价是不能预计算，只能实时处理，所以只对 Top30-40 做，不对全库做。

**速度快**（毫秒级）

**便宜 / 免费**（本地运行，不花钱）

**简单匹配型**：找事实、找关键词、找段落

**检索回来的文档数量不多**（top 50 ~ 200）

```python
from FlagEmbedding import FlagReranker

 # FP16 推理：延迟从 150ms 降到 80ms，精度损失可忽略（P@5: 0.912→0.910）
 reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)

 query = "保单现金价值怎么计算"
 candidates = [...] # Top30-40 召回候选

 # Cross-Encoder：query 和 doc 拼接后联合编码，Attention 层互相可见
 scores = reranker.compute_score([[query, doc] for doc in candidates])
 top5   = [doc for doc, _ in sorted(zip(candidates, scores),
                                    key=lambda x: x[1], reverse=True)[:5]]
```

**什么时候用 LLM Rerank**而不用Cross-Encoder

**场景1：多跳推理查询**

"如果我 45 岁投保重疾险，60 岁确诊癌症，能拿到多少赔付？"——这种查询需要跨多个文档推理：费率表、赔付标准、疾病分类。Cross-Encoder 只能判断单文档和 query 的相关性，无法做跨文档推理。对这类查询用 GPT-4o-mini 做 Rerank，并在 Prompt 里明确要求模型分析文档间的逻辑关系。

**场景2：低频长尾查询**

每天只出现 1-2 次的复杂查询，单次 LLM 成本可控，且这类查询往往需要更强的语义理解能力。

用LLM Rerank与Cross-Encoder混合的策略来Rerank

```python
def rerank(query: str, docs: list) -> list:
     if is_multi_hop(query) or is_long_tail(query):
         return llm_rerank(query, docs, model="gpt-4o-mini")
     else:
         return cross_encoder_rerank(query, docs)

 def is_multi_hop(query: str) -> bool:
     # 包含假设推理关键词
     return any(kw in query for kw in ["如果", "假设", "会不会", "能不能"])

 def is_long_tail(query: str) -> bool:
     # 查询频率低且问题较长
     return query_frequency(query) < 2 and len(query) > 30
```

### **五个工程大坑**

**坑1：Embedding 和 Rerank 套件不匹配**

用 text-embedding-3-large 召回 + bge-reranker-large 精排，Rerank 后 Top5 有 30% 和召回 Top5 完全不同——Rerank 把召回认为相关的文档打了低分。两个模型对"相关性"的理解空间不一致，导致排序混乱。换成 BGE 全家桶后问题消失。

**坑2：微调后忘记重建索引**

旧向量和新向量**不在一个语义空间**，相似度计算全乱。

**坑3：Rerank 候选集太大**

最开始对 Top100 文档做 Rerank，单次查询延迟 300-500ms，用户明显感觉卡。Top30 Rerank 的 P@5 是 0.91，Top100 只提升到 0.92——多 1 个点，延迟增加 4 倍，不值得。

**坑4：LLM Rerank 输出不稳定**

GPT-4o-mini 做 Rerank 时，同一个 query 连续调用 3 次，Top5 文档有 40% 不同（LLM 随机性）。解决方案：JSON 结构化输出 + temperature=0 + 多次采样取众数，稳定性从 60% 提升到 92%。

**坑5：成本估算不准确**

上线前没仔细估算 LLM Rerank 的调用量，实际上线后每天 5000 次查询用 GPT-4 全量 Rerank，日均成本 ¥900。切换混合策略后降到 ¥25，效果损失不到 1 个召回点。

### 坑6:数据的领域有专有名词偏差

6.1领域微调Cross-Encoder 

拿你自己的**query - 文档相关性标注数据**，对 bge-reranker 做一两轮 LoRA 微调。

6.2 关键词加权 + Cross-Encoder 混合打分

```python
final_score = 0.7 * cross_score + 0.3 *keyword_match_score
```

让模型**必须重视专业词匹配**，不会被通用语义带偏。

6.3 兜底：把专业词问题全部路由给 LLM rerank
