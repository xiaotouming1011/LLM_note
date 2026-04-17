### 第一版本：512 token机械的截断（Recall@5  :   0.67）

​	表头丢失，跨章节内容混合，列表前引导语孤立存在

### 第二版本：句子级别的切分（Recall@5:    0.74）

​	句子边界不截断，表格/列表未专门处理，仍然无法识别文档层级

### 第三版本：语义感知切分（Recall@5:    0.91）

​	文档层级识别，列表前导句保留，表格表头复制，100token智能overlap



## 第一代.**为什么固定 512 token 切分只有 67%？**

```python
# 第一代：固定长度切分
 def chunk_v1(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
     tokens = tokenizer.encode(text)
     chunks = []
     start = 0
     while start < len(tokens):
         end = min(start + chunk_size, len(tokens))
         chunks.append(tokenizer.decode(tokens[start:end]))
         start += chunk_size - overlap
     return chunks
```

失败的根因，不是参数设小了，而是切分逻辑根本不理解文档结构。

这段代码在文档上会产生三类系统性错误：

**错误1：跨章节内容混合。** 上一章节结尾的最后几句和下一章节开头的内容被切进同一个 chunk。这个 chunk 的向量既不像章节 A 也不像章节 B，检索时什么都不匹配。在我们的 2000 个 QA 测试集里，有 18% 的错误召回来自这个原因。

**错误2：表头与数据分离。** 保险费率表常见三五页跨度，表头在前半段，数据行在后半段。固定 512 切出来的表格数据 chunk 里只有数字，没有列名——"45 | 3500 | 100" 这样的内容，LLM 不知道 45 是年龄、3500 是保费还是保额。表格类问题的正确率只有 **43%**。

**错误3：列表项前导句丢失。** 保险条款里大量的免责条款是这种结构：

```
以下情况不在承保范围之内：
 （1）核辐射及核污染
 （2）战争、军事冲突
 （3）被保险人故意行为
```

固定切分会把"（1）核辐射"单独放进一个 chunk。这个 chunk 的向量没有"不在承保范围"这层语义，用户问"核辐射在不在保障范围"，系统找到了这个 chunk，但 LLM 看到核辐射出现了，不确定是正面条款还是免责条款，容易答错。

## 第二代.**句子级切分（0.74）**

改进思路：不在句子中间切，只在自然边界（句号、问号、换行）切，相邻 chunk 保留 2 句重叠。

```python
import re

 def chunk_v2(text: str, max_size: int = 512, overlap_sentences: int = 2) -> list[str]:
     sentences = re.split(r'(?<=[。！？；\n])', text)
     chunks, current, current_len = [], [], 0

     for sent in sentences:
         sent_len = len(tokenizer.encode(sent))
         if current_len + sent_len > max_size and current:
             chunks.append("".join(current))
             current = current[-overlap_sentences:]
             current_len = sum(len(tokenizer.encode(s)) for s in current)
         current.append(sent)
         current_len += sent_len

     if current:
         chunks.append("".join(current))
     return chunks
```



表格类问题和否定/列表查询是 V2 的两个最大短板

## **第三代：语义感知切分（0.91）**

**先识别文档层级结构，按语义边界切，超长章节递归细切，特殊元素单独处理。**

#### **文档结构识别**

保险文档的编号体系非常复杂，混用多种格式：`第一条` / `1.1` / `（一）` / `（1）`。用单一规则无法统一处理：

```python
from enum import Enum
 import re

 class HeaderLevel(Enum):
     H1 = 1   # 第X章/第X条
     H2 = 2   # X.X 小节 或 （一）
     H3 = 3   # （1）子条款

 def detect_header_level(line: str) -> HeaderLevel | None:
     patterns = [
         (HeaderLevel.H1, r'^第[一二三四五六七八九十百\d]+[章条节]'),
         (HeaderLevel.H1, r'^\d+\.\s+[\u4e00-\u9fa5]'),          # 1. 中文标题
         (HeaderLevel.H2, r'^\d+\.\d+\s'),                        # 1.1 小节
         (HeaderLevel.H2, r'^（[一二三四五六七八九十\d]+）'),       # （一）
         (HeaderLevel.H3, r'^（\d+）|^[a-z]\)'),                  # （1）子条款
     ]
     for level, pattern in patterns:
         if re.match(pattern, line.strip()):
             return level
     return None
```

识别出层级之后，同一章节内的内容放在同一 chunk 里，**跨章节绝不合并**。

### **表格专项处理：每个切片复制表头**

表格的每一个切片后的块都带上表头

--即使要多存一份冗余数据，也比让 LLM 面对没有列名的数字堆要好。表格类问题正确率从 43% 提升到 **78%**。



```python
def split_table(table_text: str, table_title: str, max_size: int = 300) -> list[dict]:
     rows = parse_table_rows(table_text)
     header_rows = rows[:2]         # 表头（通常1-2行）
     data_rows   = rows[2:]

     header_text   = "\n".join(header_rows)
     header_tokens = len(tokenizer.encode(header_text))

     chunks, current_rows, current_tokens = [], [], header_tokens

     for row in data_rows:
         row_tokens = len(tokenizer.encode(row))
         if current_tokens + row_tokens > max_size and current_rows:
             chunks.append({
                 "text": header_text + "\n" + "\n".join(current_rows),
                 "metadata": {"type": "table", "title": table_title}
             })
             current_rows, current_tokens = [], header_tokens
         current_rows.append(row)
         current_tokens += row_tokens

     if current_rows:
         chunks.append({"text": header_text + "\n" + "\n".join(current_rows),
                         "metadata": {"type": "table", "title": table_title}})
     return chunks
```

### **列表前导句强制保留**

识别前导句（"以下情况不在..."、"本保险赔付以下范围：..."），把前导句复制到每个列表子项 chunk 的开头。否定性查询召回率从 0.58 提升到 **0.83**。

### **智能 Overlap：量化实验决定参数**

Overlap 参数不是拍脑袋定的。我们做了系统实验：

召回率与存储率的平衡

![截屏2026-04-10 12.34.47](/Users/anji/Documents/LLM note/八股/assets/截屏2026-04-10 12.34.47.png)

## 在此基础上，进一步用句子边界对齐 overlap 截断点

###### （不在句子中间切断重叠区域），额外提升 2 个点——最终 Recall@5 = **0.91**。



## **质量评估：2000 个 QA 测试集**

切分质量必须量化，不能靠"感觉好多了"。我们的评估流程：

从 5000 份文档里抽 200 份，每份人工编写 10 个问题，总计 2000 个 QA 对。每个问题标注"正确答案应来自哪个 chunk"。跑端到端检索，看 Top-5 命中率（Recall@5）。

**关键点：分维度拆解，不只看总体数字。**

V2 的总体 Recall@5 是 0.74，看起来还行，但分维度看：表格类 0.55，否定查询 0.65。如果不分维度，会以为 0.74 已经够好，不知道往哪里优化。分维度之后，短板一目了然，V3 直接针对这两个短板优化。

构建 2000 个 QA 对花了两个人一周时间，这是一次性投入。后续每次改切分方案，重跑评估只需要 10 分钟，立刻知道有没有提升、哪类问题提升最多。没有测试集的优化是盲优化，这在面试里很减分。

![三代切分方案分维度召回率对比图](https://mmbiz.qpic.cn/sz_mmbiz_png/tEqAVhxeCuu7E180xWpOTG2gwZuqDuZuia2M1yBnyCQtyRz09UqulaqLSsvSEPdPx13BP8sfjUa6crDBuqMSV3MpOMWf7iaribRErEdU1qqriaw/640?from=appmsg)三代切分方案分维度召回率对比图

## **面试如何回答这道题**

**第一层：说出核心结论（15秒）**

"切分决定了 LLM 能看到什么。在我们的项目里，只改切分方案，不换 Embedding 模型，Recall@5 从 0.67 提升到 0.91——提升幅度比换最好的 Embedding 模型还大。"

**第二层：讲演进路径（40秒）**

"V1 固定 512 召回率 0.67，核心问题是三类：跨章节内容混合、表头与数据分离、列表前导句孤立。V2 改成句子边界切分，到 0.74，但不识别文档结构。V3 做了四个改进：按文档层级切（不跨章节合并）、表格每切片复制表头、列表前导句强制保留、100 token 智能 overlap——最终 0.91。"

**第三层：讲量化验证（30秒）**

"评估用 2000 个 QA 对，分文本、表格、否定查询、多跳推理四类分别看。V2 总体 0.74 看起来不错，但分维度看表格类只有 0.55，否定查询 0.65——这两个短板直接指导了 V3 的优化方向。没有测试集，改参数靠感觉，说不清楚提升了多少，面试官不认。"

**第四层（加分项）：讲 Overlap 实验（20秒）**

"Overlap 大小做了系统对比，100 token 是性价比最优点：比无 overlap 提升 8 个召回点，存储只增 10%，200 token 只再多 1 个点但存储增 20%，不值得。这个参数不是拍脑袋定的。"

**追问准备：**



- "chunk_size 怎么选？" — 和文档类型挂钩，保险条款句子比通用文本长 1.5 倍，512 太小；普通段落用 1024，关键条款用 1536
- "测试集 2000 个 QA 对，构建成本高怎么办？" — 前期一次性投入，后续每次改方案只需 10 分钟跑评估；没有测试集的优化是盲优化，更贵
- "有没有更好的切分方案？" — 语义切分（Semantic Chunking）理论上更准，但对 5000 份文档全量 Embedding 一次约 30-40 秒/份，规则方案只需 2-3 秒，当前延迟约束下规则方案是唯一可用方案

![面试答题框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/tEqAVhxeCus8riaLFn6bFYy7p7ibKo6to832LoodBgDR4HOEeicW0XjT5iaSN1BkkP2icv0VkpNRZoj0QzRcwcs2rHc42jUrYmhlDy9CTRg5UG8w/640?from=appmsg)面试答题框架图





