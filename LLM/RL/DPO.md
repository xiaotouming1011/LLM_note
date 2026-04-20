## 数据准备

```
train/data/preprocessor.py          # 数据预处理，了解数据格式
```

SFT: messages格式，列表套字典

```python
messages = [    
{"role": "system", "content": "你是一个助手"},    {"role": "user", "content": "你好"},    
{"role": "assistant", "content": "你好！有什么可以帮你？"},    
{"role": "user", "content": "解释一下Python包"} ]
```

input: csv    output: jsonl(一行一行的json)



SFT 数据处理器

​    将原始 CSV/JSON 数据转换为 SFT 训练格式

​    输入格式 (CSV):

​    \- query: 用户问题

​    \- answer: 标准回答

​    \- system_prompt: (可选) 系统提示词

​    输出格式 (JSONL):

```python
{
        "messages": [
{"role": "system", "content": "..."},
{"role": "user", "content": "..."},
{"role": "assistant", "content": "..."}
        ]
    }
```





DPO: chosen/rejected格式

