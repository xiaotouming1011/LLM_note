#### Main.py（print/REPL）：1.加载环境变量/组装prompt/加载hooks/注册所有工具  2.创建queryEngine 3.后处理-每轮对话结束后存session

#### core：query_loop.py状态机,query_engine.py(散件组合：工具,权限,hooks,client)

model：message.py

tools:base.py

prompt

Memory

session

api:claude.py:

- SDK 返回的是 `message_start`、`content_block_delta` 等底层事件
- 内部系统期望的是 `TextDelta`、`ToolUseStart` 等高层事件
- 这个文件把它们对齐，转换格式、提取数据、错误映射

## query_loop.py: 

step1：消息规范化，工具的schema转化为API请求所需要的dict格式列表

step2:   先检查需不需要compact，要的话把需要压缩的文本转换成纯文本

​	1.核心决策及输出

​	2.重要文件路径，函数名，代码变更

​	3.现在任务的状态

​	4.任何没解决的问题及接下来的步骤

压缩失败的话安全降级，返回原始信息列表，不中断用户对话。成功的话记得重新把消息规范化normalize

step3:  流式调用，模型/工具。模型是根据StreamingToolExecutor

工具：首先是不是并发安全工具。"content_block_start"开始，不断追加"text_delta" ，直到"content_block_stop"开始执行

step4:  工具执行+结果拼回

### message

四种消息类型：

| 类型                   | 谁发的 | 用途                   |
| ---------------------- | ------ | ---------------------- |
| UserMessage            | 用户   | 用户的输入             |
| AssistantMessage       | 模型   | 模型的回复             |
| SystemMessage          | 系统   | 系统通知（不发给 API） |
| CompactBoundaryMessage | 系统   | 压缩边界标记           |

具体字段

```python
@dataclass
class UserMessage:
    content: str | list[UserContentBlock]  # 消息内容
    uuid: str                               # 唯一标识
    timestamp: str                          # 时间戳
    type: Literal["user"] = "user"         # 类型标记
    is_meta: bool = False                   # 是否元消息
    is_compact_summary: bool = False        # 是否压缩摘要

@dataclass
class AssistantMessage:
    content: list[AssistantContentBlock]   # 内容块列表（可能是thinking+text+tool_use）
    stop_reason: str | None                # 为什么停止（end_turn/tool_use/max_tokens）
    usage: Usage                            # token 消耗统计
    model: str                              # 用的哪个模型
```

#### 
