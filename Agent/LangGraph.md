# 一.LangGraph
Graph控制执行流程--Node读取并更新State--Edge决定下一步走向，执行更新--State

目录

1. LangGraph 是什么？
2. 核心概念详解
3. 环境搭建
4. 第一个 LangGraph 程序
5. 状态管理深入理解
6. 条件边与流程控制
7. 集成 LLM 构建聊天机器人
8. 工具调用 (Tool Calling)
9. 构建完整的 ReAct Agent
10. 人机协作 (Human-in-the-Loop)
11. 持久化与记忆



**2.1 State状态** 相当于说明书，待办，日志本，node可以读取其中内容

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# 简单状态
class SimpleState(TypedDict):
    user_input: str
    result: str

# 带 Reducer 的状态 (推荐)
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 自动追加
    context: dict
```



**2.2 节点Node**

```python
from langgraph.graph import StateGraph, START, END
# 2. 定义节点函数（实际执行的代码）
def agent_function(state: ChatState) -> dict:
    """这是 agent 节点实际执行的函数"""
    return {"messages": ["Agent: 我在思考..."]}

def tool_function(state: ChatState) -> dict:
    """这是 tool 节点实际执行的函数"""
    return {"messages": ["Tool: 搜索完成"]}

# 3. 创建图
graph = StateGraph(ChatState)

# 4. ⭐ 添加节点：把名称和函数关联起来
graph.add_node("agent", agent_function)  # "agent" 是名称，agent_function 是函数
graph.add_node("tool", tool_function)    # "tool" 是名称，tool_function 是函数
```



**2.3 边Edge**

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ① 普通边
graph.add_edge(START, "agent")        # START → agent
graph.add_edge("agent", "tool")       # agent → tool
graph.add_edge("tool", END)           # tool → END

# ② 条件边
def should_continue(state: ChatState) -> str:
    """路由函数: 返回下一个节点名"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool"
    return "end"

graph.add_conditional_edges(
    "agent",                # 源节点
    should_continue,        # 路由函数
    {
        "tool": "tool",     # 返回 "tool" → 去 tool 节点
        "end": END          # 返回 "end" → 结束
    }
)

# ③ 循环边 (tool 执行后回到 agent)
graph.add_edge("tool", "agent")
```



**2.4 图Graph**

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义状态
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. 定义节点函数（实际执行的代码）
def agent_function(state: ChatState) -> dict:
    """这是 agent 节点实际执行的函数"""
    return {"messages": ["Agent: 我在思考..."]}

def tool_function(state: ChatState) -> dict:
    """这是 tool 节点实际执行的函数"""
    return {"messages": ["Tool: 搜索完成"]}

# 3. 创建图
graph = StateGraph(ChatState)

# 4. ⭐ 添加节点：把名称和函数关联起来
graph.add_node("agent", agent_function)  # "agent" 是名称，agent_function 是函数
graph.add_node("tool", tool_function)    # "tool" 是名称，tool_function 是函数

# 5. 添加边（现在才能用 "agent" 和 "tool"）
graph.add_edge(START, "agent")
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

# 6. 编译运行
app = graph.compile()
result = app.invoke({"messages": ["你好"]})
print(result["messages"])
```



#### **3.环境搭建**

```python
from dotenv import load_dotenv
load_dotenv()

#   # 用 Docker 一键部署
#   git clone https://github.com/langfuse/langfuse.git
#   cd langfuse
#   docker compose up -d

#   然后访问 http://localhost:3000 创建账号并获取 API Key。


```



![截屏2026-04-15 14.17.03](/Users/anji/Library/Application Support/typora-user-images/截屏2026-04-15 14.17.03.png)



**4.一个简单的LangGraph例子**

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class ProcessState(TypedDict):
    numbers: List[int]
    doubled: List[int]
    total: int

def double_numbers(state: ProcessState) -> dict:
    """将所有数字翻倍"""
    doubled = [n * 2 for n in state["numbers"]]
    return {"doubled": doubled}

def calculate_sum(state: ProcessState) -> dict:
    """计算总和"""
    total = sum(state["doubled"])
    return {"total": total}

# 创建并配置图
graph = StateGraph(ProcessState)
graph.add_node("double", double_numbers)
graph.add_node("sum", calculate_sum)

graph.add_edge(START, "double")
graph.add_edge("double", "sum")
graph.add_edge("sum", END)

app = graph.compile()

# 运行
result = app.invoke({
    "numbers": [1, 2, 3, 4, 5],
    "doubled": [],
    "total": 0
})

print(f"原始数字: {result['numbers']}")      # [1, 2, 3, 4, 5]
print(f"翻倍后: {result['doubled']}")        # [2, 4, 6, 8, 10]
print(f"总和: {result['total']}")            # 30
```



**5.用langfuse监控过程**

```python
from langfuse import observe, Langfuse

# 查看 observe
print(type(observe))

import os
from langfuse import observe, Langfuse
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# 7. 确保数据上传
langfuse = Langfuse()
langfuse.flush()
```



**6.状态管理深入理解**

Langgraph的状态更新机制：合并式（其他语言常见的是覆盖式：新值替换，手动合并）

```python
"""
6.1 状态更新机制：合并式 vs 覆盖式
State Update Mechanism: Merge, Not Override

LangGraph 的状态更新是合并式的，不是覆盖式的。
节点只需返回要更新的字段，其他字段会自动保留。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入依赖
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langfuse import observe, Langfuse

# 初始化 Langfuse 客户端
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 定义 State

# %%
class State(TypedDict):
    a: str
    b: str
    c: str

print("State 结构:")
print("  a: str")
print("  b: str")
print("  c: str")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="node_1")
def node_1(state: State) -> dict:
    """
    只更新字段 a，其他字段保持不变
    """
    print(f"node_1 收到: {state}")
    
    # 只返回要更新的字段
    result = {"a": "updated_a"}
    
    print(f"node_1 返回: {result}")
    return result


@observe(name="node_2")
def node_2(state: State) -> dict:
    """
    只更新字段 b，其他字段保持不变
    """
    print(f"node_2 收到: {state}")
    
    # 只返回要更新的字段
    result = {"b": "updated_b"}
    
    print(f"node_2 返回: {result}")
    return result

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 构建图

# %%
graph = StateGraph(State)

# 添加节点
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)

# 添加边
graph.add_edge(START, "node_1")
graph.add_edge("node_1", "node_2")
graph.add_edge("node_2", END)

app = graph.compile()
print("✓ 图构建完成")

# %% [markdown]
# ## 5. 执行演示

# %%
# 初始状态
initial_state = {
    "a": "init_a",
    "b": "init_b",
    "c": "init_c"
}

print("=" * 50)
print("6.1 状态更新机制演示")
print("=" * 50)
print(f"\n初始状态: {initial_state}")
print("\n执行流程:")
print("-" * 30)

# 执行图
result = app.invoke(initial_state)

print("-" * 30)
print(f"\n最终状态: {result}")

# 验证结果
print("\n验证:")
print(f"  a: 'init_a' -> '{result['a']}' (被 node_1 更新)")
print(f"  b: 'init_b' -> '{result['b']}' (被 node_2 更新)")
print(f"  c: 'init_c' -> '{result['c']}' (未被任何节点修改，保持原值)")

# %% [markdown]
# ## 6. 发送 Langfuse 数据

# %%
# 确保 Langfuse 数据被发送
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")

✓ Langfuse 初始化完成
State 结构:
  a: str
  b: str
  c: str
✓ 节点函数定义完成
✓ 图构建完成
==================================================
6.1 状态更新机制演示
==================================================

初始状态: {'a': 'init_a', 'b': 'init_b', 'c': 'init_c'}

执行流程:
------------------------------
node_1 收到: {'a': 'init_a', 'b': 'init_b', 'c': 'init_c'}
node_1 返回: {'a': 'updated_a'}
node_2 收到: {'a': 'updated_a', 'b': 'init_b', 'c': 'init_c'}
node_2 返回: {'b': 'updated_b'}
------------------------------

最终状态: {'a': 'updated_a', 'b': 'updated_b', 'c': 'init_c'}

验证:
  a: 'init_a' -> 'updated_a' (被 node_1 更新)
  b: 'init_b' -> 'updated_b' (被 node_2 更新)
  c: 'init_c' -> 'init_c' (未被任何节点修改，保持原值)

✓ 执行完成，数据已发送到 Langfuse 监控
```



**6.2 使用 Reducer 处理列表累加**

```python
"""
6.2 使用 Reducer 处理列表累加
Using Reducer for List Accumulation

当你需要将新数据追加到列表而不是覆盖时，需要使用 Reducer。
通过 Annotated 和 operator.add 实现列表的累加效果。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入依赖
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langfuse import observe, Langfuse

# 初始化 Langfuse 客户端
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 定义 State（使用 Reducer）

# %%
class State(TypedDict):
    # 使用 Annotated 和 operator.add 作为 reducer
    # 这样每次返回的列表会被追加，而不是覆盖
    messages: Annotated[List[str], operator.add]
    count: int

print("State 结构:")
print("  messages: Annotated[List[str], operator.add]  ← 使用 Reducer!")
print("  count: int")
print("\n关键: operator.add 会将新列表追加到旧列表")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="add_greeting")
def add_greeting(state: State) -> dict:
    """添加问候语到消息列表"""
    print(f"add_greeting 收到: messages={state['messages']}, count={state['count']}")
    
    result = {
        "messages": ["Hello!"],  # 会被追加到现有列表
        "count": state["count"] + 1
    }
    
    print(f"add_greeting 返回: {result}")
    return result


@observe(name="add_question")
def add_question(state: State) -> dict:
    """添加问题到消息列表"""
    print(f"add_question 收到: messages={state['messages']}, count={state['count']}")
    
    result = {
        "messages": ["How are you?"],  # 继续追加
        "count": state["count"] + 1
    }
    
    print(f"add_question 返回: {result}")
    return result


@observe(name="add_farewell")
def add_farewell(state: State) -> dict:
    """添加告别语到消息列表"""
    print(f"add_farewell 收到: messages={state['messages']}, count={state['count']}")
    
    result = {
        "messages": ["Goodbye!"],  # 继续追加
        "count": state["count"] + 1
    }
    
    print(f"add_farewell 返回: {result}")
    return result

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 构建图

# %%
graph = StateGraph(State)

# 添加节点
graph.add_node("greeting", add_greeting)
graph.add_node("question", add_question)
graph.add_node("farewell", add_farewell)

# 添加边
graph.add_edge(START, "greeting")
graph.add_edge("greeting", "question")
graph.add_edge("question", "farewell")
graph.add_edge("farewell", END)

app = graph.compile()
print("✓ 图构建完成")

# %% [markdown]
# ## 5. 执行演示

# %%
# 初始状态
initial_state = {
    "messages": [],  # 空列表
    "count": 0
}

print("=" * 60)
print("6.2 Reducer 列表累加演示")
print("=" * 60)
print(f"\n初始状态: {initial_state}")
print("\n执行流程:")
print("-" * 40)

# 执行图
result = app.invoke(initial_state)

print("-" * 40)
print(f"\n最终状态:")
print(f"  messages: {result['messages']}")
print(f"  count: {result['count']}")

# 验证结果
print("\n验证 Reducer 效果:")
print("  每个节点返回的 messages 都被追加，而不是覆盖:")
for i, msg in enumerate(result['messages'], 1):
    print(f"    {i}. \"{msg}\"")

# 对比：如果没有 Reducer
print("\n对比（如果没有 Reducer）:")
print("  最后一个节点返回 ['Goodbye!']")
print("  结果会是: messages=['Goodbye!'] ← 前面的消息丢失！")

# %% [markdown]
# ## 6. 发送 Langfuse 数据

# %%
# 确保 Langfuse 数据被发送
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```

```python
✓ Langfuse 初始化完成
State 结构:
  messages: Annotated[List[str], operator.add]  ← 使用 Reducer!
  count: int

关键: operator.add 会将新列表追加到旧列表
✓ 节点函数定义完成
✓ 图构建完成
============================================================
6.2 Reducer 列表累加演示
============================================================

初始状态: {'messages': [], 'count': 0}

执行流程:
----------------------------------------
add_greeting 收到: messages=[], count=0
add_greeting 返回: {'messages': ['Hello!'], 'count': 1}
add_question 收到: messages=['Hello!'], count=1
add_question 返回: {'messages': ['How are you?'], 'count': 2}
add_farewell 收到: messages=['Hello!', 'How are you?'], count=2
add_farewell 返回: {'messages': ['Goodbye!'], 'count': 3}
----------------------------------------

最终状态:
...
  最后一个节点返回 ['Goodbye!']
  结果会是: messages=['Goodbye!'] ← 前面的消息丢失！

✓ 执行完成，数据已发送到 Langfuse 监控

```



**6.3MessagesState(一般用于聊天应用)**

```python
"""
6.3 使用 MessagesState（推荐用于聊天应用）
Using MessagesState for Chat Applications

LangGraph 提供了预定义的 MessagesState，专门用于处理消息历史。
内置了 add_messages reducer，自动处理消息追加和去重。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入依赖
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langfuse import observe, Langfuse

# 初始化 Langfuse 客户端
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 了解 MessagesState

# %%
# MessagesState 已经内置了 messages 字段和正确的 reducer
# 等价于：
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

print("MessagesState 特点:")
print("  ✓ 内置 messages 字段")
print("  ✓ 内置 add_messages reducer")
print("  ✓ 支持 HumanMessage, AIMessage, SystemMessage 等")
print("  ✓ 自动处理消息 ID 去重")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="process_input")
def process_input(state: MessagesState) -> dict:
    """处理用户输入，生成系统分析"""
    messages = state["messages"]
    last_message = messages[-1]
    
    print(f"process_input: 收到用户消息 - \"{last_message.content}\"")

    # 添加一条系统处理消息
    system_note = AIMessage(
        content=f"[系统] 正在处理您的请求: {last_message.content}"
    )
    
    return {"messages": [system_note]}  # 会自动追加


@observe(name="generate_response")
def generate_response(state: MessagesState) -> dict:
    """生成 AI 回复"""
    messages = state["messages"]
    
    # 找到用户的原始消息（第一条 HumanMessage）
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    original_query = user_messages[-1].content if user_messages else "未知请求"
    
    print(f"generate_response: 生成回复...")
    
    # 生成回复
    response = AIMessage(
        content=f"您好！您说的是「{original_query}」。我是 AI 助手，很高兴为您服务！"
    )
    
    return {"messages": [response]}  # 会自动追加


@observe(name="add_followup")
def add_followup(state: MessagesState) -> dict:
    """添加后续提示"""
    print(f"add_followup: 添加后续提示...")
    
    followup = AIMessage(
        content="还有什么我可以帮助您的吗？"
    )
    
    return {"messages": [followup]}  # 会自动追加

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 构建图

# %%
graph = StateGraph(MessagesState)

# 添加节点
graph.add_node("process", process_input)
graph.add_node("respond", generate_response)
graph.add_node("followup", add_followup)

# 添加边
graph.add_edge(START, "process")
graph.add_edge("process", "respond")
graph.add_edge("respond", "followup")
graph.add_edge("followup", END)

app = graph.compile()
print("✓ 图构建完成")

# %% [markdown]
# ## 5. 执行演示

# %%
# 初始状态：用户发送一条消息
initial_state = {
    "messages": [
        HumanMessage(content="你好，我想了解 LangGraph！")
    ]
}

print("=" * 60)
print("6.3 MessagesState 演示")
print("=" * 60)

print(f"\n初始消息:")
for msg in initial_state["messages"]:
    print(f"  [{msg.type}] {msg.content}")

print("\n执行流程:")
print("-" * 40)

# 执行图
result = app.invoke(initial_state)

print("-" * 40)
print(f"\n最终消息列表 ({len(result['messages'])} 条):")
for i, msg in enumerate(result["messages"], 1):
    msg_type = msg.type.upper()
    prefix = "👤" if msg_type == "HUMAN" else "🤖"
    print(f"  {i}. {prefix} [{msg_type}] {msg.content}")

# 验证消息累加
print("\n验证:")
print(f"  初始: 1 条消息")
print(f"  最终: {len(result['messages'])} 条消息")
print(f"  所有消息都被保留，新消息自动追加")

# %% [markdown]
# ## 6. 发送 Langfuse 数据

# %%
# 确保 Langfuse 数据被发送
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```



**6.4 扩展MessageState**（MessageState只有message字段，实际需要用户信息，会话ID等）

```python
✓ Langfuse 初始化完成
CustomState 结构:
  继承自 MessagesState:
    ├── messages (内置 add_messages reducer)
  自定义字段:
    ├── user_name: Optional[str]
    ├── session_id: str
    ├── tool_calls_count: int
    ├── current_intent: Optional[str]
    └── conversation_started: str
✓ 节点函数定义完成
✓ 图构建完成
======================================================================
6.4 扩展 MessagesState 演示
======================================================================

初始状态:
  messages: [human] 你好，我是张三，我想查询一下订单状态
  user_name: None
  session_id: f48162a7...
  tool_calls_count: 0
  current_intent: None

执行流程:
--------------------------------------------------
...
  ✓ 用户名从消息中提取: 张三
  ✓ 意图被正确识别: query

✓ 执行完成，数据已发送到 Langfuse 监控
```



**7.条件边与流程控制**

**7.1**基本条件边

**7.2实现循环loop**

​	循环的实现：条件返回的节点指向自己

**7.3多条件分支**

```python
"""
7.3 多条件分支
Multiple Branch Routing

根据输入类型路由到不同处理节点，实现智能分发。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入依赖
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langfuse import observe, Langfuse

# 初始化 Langfuse 客户端
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 定义 State

# %%
class RequestState(TypedDict):
    request_type: str
    response: str

print("RequestState 结构:")
print("  request_type: str  # 用户请求类型")
print("  response: str      # 处理结果")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="handle_order")
def handle_order(state: RequestState) -> dict:
    """处理订单请求"""
    print("handle_order: 正在处理订单请求...")
    return {"response": "📦 处理订单请求：已为您查询订单状态"}


@observe(name="handle_refund")
def handle_refund(state: RequestState) -> dict:
    """处理退款请求"""
    print("handle_refund: 正在处理退款请求...")
    return {"response": "💰 处理退款请求：已提交退款申请"}


@observe(name="handle_inquiry")
def handle_inquiry(state: RequestState) -> dict:
    """处理咨询请求"""
    print("handle_inquiry: 正在处理咨询请求...")
    return {"response": "❓ 处理咨询请求：请问有什么可以帮助您的？"}


@observe(name="handle_unknown")
def handle_unknown(state: RequestState) -> dict:
    """处理无法识别的请求"""
    print("handle_unknown: 无法识别请求类型")
    return {"response": "🤷 无法识别的请求类型，请选择：订单查询、退款申请、问题咨询"}

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 定义路由函数

# %%
def route_request(state: RequestState) -> Literal["order", "refund", "inquiry", "unknown"]:
    """
    根据请求类型路由到不同处理节点
    
    返回值必须是目标节点的名字字符串
    """
    request_type = state["request_type"].lower()
    
    if "订单" in request_type or "购买" in request_type:
        print(f"route_request: 识别为订单请求 → order")
        return "order"
    elif "退款" in request_type or "退货" in request_type:
        print(f"route_request: 识别为退款请求 → refund")
        return "refund"
    elif "咨询" in request_type or "问题" in request_type:
        print(f"route_request: 识别为咨询请求 → inquiry")
        return "inquiry"
    else:
        print(f"route_request: 无法识别 → unknown")
        return "unknown"

print("✓ 路由函数定义完成")

# %% [markdown]
# ## 5. 构建图

# %%
graph = StateGraph(RequestState)

# 添加节点
graph.add_node("order", handle_order)
graph.add_node("refund", handle_refund)
graph.add_node("inquiry", handle_inquiry)
graph.add_node("unknown", handle_unknown)

# 从 START 直接使用条件边路由
# 注意：可以直接从 START 进行条件路由，不需要先经过一个节点！
graph.add_conditional_edges(
    START,
    route_request,
    ["order", "refund", "inquiry", "unknown"]
)

# 所有处理节点都连接到 END
graph.add_edge("order", END)
graph.add_edge("refund", END)
graph.add_edge("inquiry", END)
graph.add_edge("unknown", END)

app = graph.compile()
print("✓ 图构建完成")

# %% [markdown]
# ## 6. 执行演示

# %%
@observe(name="multi_branch_demo")
def run_demo():
    """演示多条件分支路由"""
    
    print("=" * 60)
    print("7.3 多条件分支演示")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        "我要购买商品",
        "申请退款",
        "有个问题想咨询",
        "你好，在吗？"  # 无法识别的请求
    ]
    
    results = []
    for i, request in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: \"{request}\" ---")
        result = app.invoke({"request_type": request, "response": ""})
        print(f"响应: {result['response']}")
        results.append(result)
    
    print("\n" + "=" * 60)
    print("路由流程图:")
    print("""
                                  ┌─────────────┐
                       "订单" ──→ │   order     │ ──┐
                                  └─────────────┘   │
                                  ┌─────────────┐   │
    ┌───────┐    ┌─────────┐      │   refund    │   │   ┌─────┐
    │ START │──→ │  route  │──→   └─────────────┘  ─┼─→ │ END │
    └───────┘    └─────────┘      ┌─────────────┐   │   └─────┘
                       "咨询" ──→ │  inquiry    │ ──┤
                                  └─────────────┘   │
                                  ┌─────────────┐   │
                       其他 ───→  │  unknown    │ ──┘
                                  └─────────────┘
    """)
    
    return results


# 运行演示
results = run_demo()

# %% [markdown]
# ## 7. 发送 Langfuse 数据

# %%
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```



8.1

8.2messageState多轮对话

8.3流式输出

**9.用langgrapg调用工具**

```python
# 定义工具
from langchain_core.tools import tool
from typing import Optional

@tool
def search_weather(city: str) -> str:
    """
    查询指定城市的天气。
    
    Args:
        city: 城市名称
    """
    # 模拟天气 API
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "广州": "雷阵雨，30°C",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")

@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_current_time() -> str:
    """获取当前时间。不需要任何参数。"""
    from datetime import datetime
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# 将工具收集到列表中
tools = [search_weather, calculate, get_current_time]
```



#### 11.记忆能力集成     checkpoint

**11.1使用MemorySaver**（内存存储）

==使用 Checkpointer 让 LangGraph 记住对话历史。==

==相同 thread_id 的对话会自动累加消息。==

==\# 🔑 关键：创建内存检查点存储==

==memory = MemorySaver()==

==\# 🔑 关键：编译时传入 checkpointer==

==app = graph.compile(checkpointer=memory)==

==\# 🔑 关键：使用 thread_id 标识对话==

​    ==config = {"configurable": {"thread_id": "user-abc-123"}}==

==result1 = app.invoke({"messages": [HumanMessage(content=user_msg1)]}, config)==

```python
"""
11.1 使用 MemorySaver（内存存储）
Memory Persistence with MemorySaver

使用 Checkpointer 让 LangGraph 记住对话历史。
相同 thread_id 的对话会自动累加消息。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
from dotenv import load_dotenv
load_dotenv()

import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langfuse import observe, Langfuse

# 初始化 Langfuse
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 初始化 LLM (Qwen)

# %%
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("✓ Qwen LLM 初始化完成")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="chatbot")
def chatbot(state: MessagesState) -> dict:
    """聊天机器人节点"""
    # 添加系统提示
    messages = [
        SystemMessage(content="你是一个友好的助手，用中文简洁回答。记住用户告诉你的信息。")
    ] + state["messages"]
    
    print(f"chatbot: 当前消息数 {len(state['messages'])}")
    
    # 调用 LLM
    response = llm.invoke(messages)
    
    return {"messages": [response]}

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 构建图（带 Checkpointer）

# %%
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 🔑 关键：创建内存检查点存储
memory = MemorySaver()

# 🔑 关键：编译时传入 checkpointer
app = graph.compile(checkpointer=memory)

print("✓ 图构建完成（已启用 MemorySaver）")

# %% [markdown]
# ## 5. 演示多轮对话记忆

# %%
@observe(name="memory_saver_demo")
def run_demo():
    """演示 MemorySaver 的多轮对话记忆"""
    
    print("=" * 60)
    print("11.1 MemorySaver 多轮对话记忆演示")
    print("=" * 60)
    
    # 🔑 关键：使用 thread_id 标识对话
    config = {"configurable": {"thread_id": "user-abc-123"}}
    
    # 第一轮对话
    print("\n--- 第一轮对话 ---")
    user_msg1 = "你好，我叫小明"
    print(f"👤 用户: {user_msg1}")
    
    result1 = app.invoke({"messages": [HumanMessage(content=user_msg1)]}, config)
    print(f"🤖 助手: {result1['messages'][-1].content}")
    print(f"   (当前消息数: {len(result1['messages'])})")
    
    # 第二轮对话 - 会自动加载之前的消息
    print("\n--- 第二轮对话 ---")
    user_msg2 = "我喜欢打篮球和编程"
    print(f"👤 用户: {user_msg2}")
    
    result2 = app.invoke({"messages": [HumanMessage(content=user_msg2)]}, config)
    print(f"🤖 助手: {result2['messages'][-1].content}")
    print(f"   (当前消息数: {len(result2['messages'])})")
    
    # 第三轮对话 - 测试记忆
    print("\n--- 第三轮对话（测试记忆）---")
    user_msg3 = "你还记得我叫什么名字吗？我有什么爱好？"
    print(f"👤 用户: {user_msg3}")
    
    result3 = app.invoke({"messages": [HumanMessage(content=user_msg3)]}, config)
    print(f"🤖 助手: {result3['messages'][-1].content}")
    print(f"   (当前消息数: {len(result3['messages'])})")
    
    # 打印完整对话历史
    print("\n" + "=" * 60)
    print("完整对话历史:")
    print("=" * 60)
    for i, msg in enumerate(result3["messages"], 1):
        role = "👤 用户" if msg.type == "human" else "🤖 助手"
        content = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"{i}. {role}: {content}")
    
    return result3


result = run_demo()

# %% [markdown]
# ## 6. 演示不同 thread_id 的隔离

# %%
@observe(name="thread_isolation_demo")
def run_isolation_demo():
    """演示不同 thread_id 的对话隔离"""
    
    print("\n" + "=" * 60)
    print("thread_id 隔离演示")
    print("=" * 60)
    
    # 用户 A 的对话
    config_a = {"configurable": {"thread_id": "user-A"}}
    print("\n--- 用户 A (thread_id='user-A') ---")
    result_a = app.invoke(
        {"messages": [HumanMessage(content="我是用户A，我喜欢看电影")]}, 
        config_a
    )
    print(f"🤖 助手: {result_a['messages'][-1].content}")
    
    # 用户 B 的对话 - 完全独立
    config_b = {"configurable": {"thread_id": "user-B"}}
    print("\n--- 用户 B (thread_id='user-B') ---")
    result_b = app.invoke(
        {"messages": [HumanMessage(content="我是用户B，请问用户A喜欢什么？")]}, 
        config_b
    )
    print(f"🤖 助手: {result_b['messages'][-1].content}")
    print("\n💡 注意: 用户 B 不知道用户 A 的信息，因为 thread_id 不同")


run_isolation_demo()

# %% [markdown]
# ## 7. 发送 Langfuse 数据

# %%
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```



**11.2使用SQLite持久化**

使用 SQLite 持久化

使用 SqliteSaver 将对话历史持久化到文件。

即使程序重启，对话历史也会保留。

==\# 🔑 关键：使用 with 语句确保正确关闭连接，memory存储路径改为数据库==

​    ==with SqliteSaver.from_conn_string(db_path) as memory:==

```python
"""
11.2 使用 SQLite 持久化
SQLite Persistence for Chat History

使用 SqliteSaver 将对话历史持久化到文件。
即使程序重启，对话历史也会保留。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
from dotenv import load_dotenv
load_dotenv()

import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langfuse import observe, Langfuse

# 初始化 Langfuse
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 初始化 LLM (Qwen)

# %%
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("✓ Qwen LLM 初始化完成")

# %% [markdown]
# ## 3. 定义节点函数

# %%
@observe(name="chatbot")
def chatbot(state: MessagesState) -> dict:
    """聊天机器人节点"""
    messages = [
        SystemMessage(content="你是一个友好的助手，用中文简洁回答。记住用户告诉你的信息。")
    ] + state["messages"]
    
    print(f"chatbot: 当前消息数 {len(state['messages'])}")
    
    response = llm.invoke(messages)
    return {"messages": [response]}

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 4. 构建图（使用 SqliteSaver）

# %%
# 构建图结构
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

print("✓ 图结构定义完成")

# %% [markdown]
# ## 5. 使用 SqliteSaver 进行持久化对话

# %%
@observe(name="sqlite_persistence_demo")
def run_demo():
    """演示 SQLite 持久化"""
    
    print("=" * 60)
    print("11.2 SQLite 持久化演示")
    print("=" * 60)
    
    # 数据库文件路径
    db_path = "chat_history.db"
    
    # 🔑 关键：使用 with 语句确保正确关闭连接
    with SqliteSaver.from_conn_string(db_path) as memory:
        # 编译图时传入 checkpointer
        app = graph.compile(checkpointer=memory)
        
        # 使用固定的 thread_id（模拟持久化场景）
        config = {"configurable": {"thread_id": "persistent-chat-001"}}
        
        # 第一轮对话
        print("\n--- 第一轮对话 ---")
        user_msg1 = "你好，请记住：我的生日是 5 月 20 日"
        print(f"👤 用户: {user_msg1}")
        
        result1 = app.invoke({"messages": [HumanMessage(content=user_msg1)]}, config)
        print(f"🤖 助手: {result1['messages'][-1].content}")
        print(f"   (消息数: {len(result1['messages'])})")
        
        # 第二轮对话
        print("\n--- 第二轮对话 ---")
        user_msg2 = "我的生日是哪天？"
        print(f"👤 用户: {user_msg2}")
        
        result2 = app.invoke({"messages": [HumanMessage(content=user_msg2)]}, config)
        print(f"🤖 助手: {result2['messages'][-1].content}")
        print(f"   (消息数: {len(result2['messages'])})")
    
    print(f"\n💾 对话已保存到: {db_path}")
    print("   即使程序重启，下次使用相同 thread_id 时会恢复历史")
    
    return result2


result = run_demo()

# %% [markdown]
# ## 6. 模拟程序重启后恢复对话

# %%
@observe(name="restore_demo")
def run_restore_demo():
    """模拟程序重启后恢复对话"""
    
    print("\n" + "=" * 60)
    print("模拟程序重启后恢复对话")
    print("=" * 60)
    
    db_path = "chat_history.db"
    
    # 模拟"重启"：重新打开数据库连接
    with SqliteSaver.from_conn_string(db_path) as memory:
        app = graph.compile(checkpointer=memory)
        
        # 使用相同的 thread_id
        config = {"configurable": {"thread_id": "persistent-chat-001"}}
        
        print("\n--- 恢复对话（使用相同 thread_id）---")
        user_msg = "你还记得我之前告诉你的信息吗？，我生日什么时候"
        print(f"👤 用户: {user_msg}")
        
        result = app.invoke({"messages": [HumanMessage(content=user_msg)]}, config)
        print(f"🤖 助手: {result['messages'][-1].content}")
        print(f"   (消息数: {len(result['messages'])})")
        
        # 显示完整历史
        print("\n--- 完整对话历史 ---")
        for i, msg in enumerate(result["messages"], 1):
            role = "👤" if msg.type == "human" else "🤖"
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            print(f"  {i}. {role} {content}")
    
    return result


result2 = run_restore_demo()

# %% [markdown]
# ## 7. 异步版本（可选）

# %%
print("\n" + "=" * 60)
print("异步版本代码示例")
print("=" * 60)
print("""
# 如果需要异步支持，可以使用 AsyncSqliteSaver

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def async_chat():
    async with AsyncSqliteSaver.from_conn_string("chat.db") as memory:
        app = graph.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "async-chat"}}
        
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="你好")]},
            config
        )
        return result
""")

# %% [markdown]
# ## 8. 发送 Langfuse 数据

# %%
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```



**11.3跨会话的长期记忆**

除了消息历史，还可以存储用户偏好、对话计数等自定义状态。

使用 SqliteSaver 持久化，实现程序重启后记忆不丢失。

```python
"""
11.3 跨会话的长期记忆
Long-term Memory Across Sessions

除了消息历史，还可以存储用户偏好、对话计数等自定义状态。
使用 SqliteSaver 持久化，实现程序重启后记忆不丢失。
"""

# %% [markdown]
# ## 1. 环境配置

# %%
from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated, List, Optional
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langfuse import observe, Langfuse

# 初始化 Langfuse
langfuse = Langfuse()
print("✓ Langfuse 初始化完成")

# %% [markdown]
# ## 2. 初始化 LLM (Qwen)

# %%
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
print("✓ Qwen LLM 初始化完成")

# %% [markdown]
# ## 3. 定义扩展的 State（包含长期记忆）

# %%
class LongTermMemoryState(TypedDict):
    """
    扩展的 State，包含:
    - messages: 对话消息（使用 add_messages reducer）
    - user_profile: 用户档案（姓名、偏好等）
    - conversation_count: 对话次数
    - topics_discussed: 讨论过的话题
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: dict
    conversation_count: int
    topics_discussed: Annotated[List[str], operator.add]

print("LongTermMemoryState 结构:")
print("  messages: 对话消息（自动累加）")
print("  user_profile: 用户档案（姓名、偏好）")
print("  conversation_count: 对话计数")
print("  topics_discussed: 讨论过的话题")

# %% [markdown]
# ## 4. 定义节点函数

# %%
@observe(name="extract_user_info")
def extract_user_info(state: LongTermMemoryState) -> dict:
    """从对话中提取用户信息"""
    messages = state["messages"]
    last_msg = messages[-1].content if messages else ""
    profile = state.get("user_profile", {})
    
    print(f"extract_user_info: 分析用户消息...")
    
    # 简单的信息提取（实际项目中可以用 LLM 来提取）
    if "我叫" in last_msg:
        parts = last_msg.split("我叫")
        if len(parts) > 1:
            name = parts[1].split("，")[0].split(",")[0].split("。")[0].strip()[:10]
            profile["name"] = name
            print(f"  提取到姓名: {name}")
    
    if "喜欢" in last_msg:
        # 简单提取喜好关键词
        hobbies = profile.get("hobbies", [])
        for keyword in ["编程", "篮球", "电影", "音乐", "游戏", "读书", "旅游"]:
            if keyword in last_msg and keyword not in hobbies:
                hobbies.append(keyword)
                print(f"  提取到爱好: {keyword}")
        profile["hobbies"] = hobbies
    
    return {"user_profile": profile}


@observe(name="extract_topics")
def extract_topics(state: LongTermMemoryState) -> dict:
    """提取讨论话题"""
    messages = state["messages"]
    last_msg = messages[-1].content.lower() if messages else ""
    
    topics = []
    topic_keywords = {
        "天气": ["天气", "温度", "下雨"],
        "工作": ["工作", "上班", "项目"],
        "学习": ["学习", "编程", "课程"],
        "生活": ["吃饭", "睡觉", "运动"],
    }
    
    for topic, keywords in topic_keywords.items():
        if any(kw in last_msg for kw in keywords):
            topics.append(topic)
            print(f"  识别到话题: {topic}")
    
    # 更新对话计数
    count = state.get("conversation_count", 0) + 1
    print(f"  对话次数: {count}")
    
    return {
        "topics_discussed": topics,
        "conversation_count": count
    }


@observe(name="chat_with_memory")
def chat_with_memory(state: LongTermMemoryState) -> dict:
    """使用长期记忆生成个性化回复"""
    profile = state.get("user_profile", {})
    count = state.get("conversation_count", 0)
    topics = state.get("topics_discussed", [])
    
    # 构建个性化系统提示
    system_content = "你是一个友好的助手，用中文简洁回答。"
    
    if profile.get("name"):
        system_content += f"\n用户名字叫{profile['name']}，可以适当称呼他的名字。"
    
    if profile.get("hobbies"):
        system_content += f"\n用户的爱好包括: {', '.join(profile['hobbies'])}。"
    
    if count > 1:
        system_content += f"\n这是你们的第 {count} 次对话。"
    
    if topics:
        recent_topics = list(set(topics))[-3:]
        system_content += f"\n之前讨论过的话题: {', '.join(recent_topics)}。"
    
    print(f"chat_with_memory: 生成个性化回复...")
    print(f"  用户档案: {profile}")
    print(f"  对话次数: {count}")
    
    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

print("✓ 节点函数定义完成")

# %% [markdown]
# ## 5. 构建图

# %%
graph = StateGraph(LongTermMemoryState)

# 添加节点
graph.add_node("extract_info", extract_user_info)
graph.add_node("extract_topics", extract_topics)
graph.add_node("chat", chat_with_memory)

# 添加边
graph.add_edge(START, "extract_info")
graph.add_edge("extract_info", "extract_topics")
graph.add_edge("extract_topics", "chat")
graph.add_edge("chat", END)

print("✓ 图结构定义完成")

# 数据库路径
DB_PATH = "long_term_memory.db"

# %% [markdown]
# ## 6. 演示长期记忆

# %%
@observe(name="long_term_memory_demo")
def run_demo():
    """演示跨会话的长期记忆"""
    
    print("=" * 70)
    print("11.3 跨会话长期记忆演示（SqliteSaver 持久化）")
    print("=" * 70)
    
    # 🔑 使用 SqliteSaver 持久化
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = graph.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "long-term-user-001"}}
        
        # 初始状态
        initial_state = {
            "messages": [],
            "user_profile": {},
            "conversation_count": 0,
            "topics_discussed": []
        }
        
        # 第一轮对话
        print("\n" + "=" * 50)
        print("第一轮对话")
        print("=" * 50)
        user_msg1 = "你好，我叫小明，我喜欢编程和篮球"
        print(f"\n👤 用户: {user_msg1}")
        
        initial_state["messages"] = [HumanMessage(content=user_msg1)]
        result1 = app.invoke(initial_state, config)
        print(f"\n🤖 助手: {result1['messages'][-1].content}")
        print(f"\n📊 当前状态:")
        print(f"   用户档案: {result1['user_profile']}")
        print(f"   对话次数: {result1['conversation_count']}")
        print(f"   话题: {result1['topics_discussed']}")
        
        # 第二轮对话
        print("\n" + "=" * 50)
        print("第二轮对话")
        print("=" * 50)
        user_msg2 = "今天天气真好，适合打篮球"
        print(f"\n👤 用户: {user_msg2}")
        
        result2 = app.invoke({"messages": [HumanMessage(content=user_msg2)]}, config)
        print(f"\n🤖 助手: {result2['messages'][-1].content}")
        print(f"\n📊 当前状态:")
        print(f"   用户档案: {result2['user_profile']}")
        print(f"   对话次数: {result2['conversation_count']}")
        print(f"   话题: {list(set(result2['topics_discussed']))}")
    
    print(f"\n💾 数据已保存到: {DB_PATH}")
    return result2


result = run_demo()

# %% [markdown]
# ## 7. 模拟程序重启后恢复记忆

# %%
@observe(name="restore_memory_demo")
def run_restore_demo():
    """模拟程序重启后，恢复用户的长期记忆"""
    
    print("\n" + "=" * 70)
    print("模拟程序重启后恢复长期记忆")
    print("=" * 70)
    print("（重新打开数据库，使用相同 thread_id）")
    
    # 重新打开数据库连接（模拟程序重启）
    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        app = graph.compile(checkpointer=memory)
        
        # 使用相同的 thread_id
        config = {"configurable": {"thread_id": "long-term-user-001"}}
        
        print("\n--- 恢复对话（测试长期记忆）---")
        user_msg = "你还记得我的信息吗？我叫什么？喜欢什么？"
        print(f"\n👤 用户: {user_msg}")
        
        result = app.invoke({"messages": [HumanMessage(content=user_msg)]}, config)
        print(f"\n🤖 助手: {result['messages'][-1].content}")
        
        print(f"\n📊 恢复的状态:")
        print(f"   用户档案: {result['user_profile']}")
        print(f"   对话次数: {result['conversation_count']}")
        print(f"   话题: {list(set(result['topics_discussed']))}")
        print(f"   消息数: {len(result['messages'])}")
        
        print("\n✅ 长期记忆恢复成功！用户档案、对话次数、话题都已恢复")
    
    return result


result2 = run_restore_demo()

# %% [markdown]
# ## 8. 长期记忆的应用场景

# %%
print("\n" + "=" * 70)
print("长期记忆的应用场景")
print("=" * 70)
print("""
1. 个性化推荐
   - 记住用户偏好，提供定制化建议
   - 例：记住用户喜欢的编程语言，优先用该语言举例

2. 客服机器人
   - 记住用户的问题历史，避免重复询问
   - 记住用户的账号信息、订单状态

3. 教育助手
   - 跟踪学习进度
   - 记住学生的薄弱点，针对性辅导

4. 健康助手
   - 记录用户的健康数据
   - 跟踪用药、运动等习惯

5. 游戏 NPC
   - 记住玩家的选择和偏好
   - 根据历史互动调整对话风格
""")

# %% [markdown]
# ## 9. MemorySaver vs SqliteSaver 对比

# %%
print("\n" + "=" * 70)
print("MemorySaver vs SqliteSaver 对比")
print("=" * 70)
print("""
┌─────────────────┬─────────────────────┬─────────────────────┐
│                 │ MemorySaver         │ SqliteSaver         │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 数据存储位置    │ 内存                │ SQLite 文件         │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 程序重启后      │ ❌ 数据丢失         │ ✅ 数据保留         │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 适用场景        │ 开发测试            │ 生产环境            │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 性能            │ 最快                │ 稍慢（磁盘 I/O）    │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 存储内容        │ 完整 State          │ 完整 State          │
│                 │ (messages +         │ (messages +         │
│                 │  自定义字段)        │  自定义字段)        │
└─────────────────┴─────────────────────┴─────────────────────┘

💡 本示例使用 SqliteSaver，实现了：
   - 消息历史持久化
   - 用户档案持久化
   - 对话次数持久化  
   - 话题列表持久化
""")

# %% [markdown]
# ## 10. 发送 Langfuse 数据

# %%
langfuse.flush()
print("\n✓ 执行完成，数据已发送到 Langfuse 监控")
```





#### 12.Test2SQL

四个核心流程：1.为LLM提供Schema. 2.LLM生成SQL的策略 ：few-shot,CoT,Self-consistency     3.SQL Validator执行SQL前安全检查     4.Result Formatter结果的后处理：表格格式化，自然语言总结，错误提示



## 二、Text2SQL 如何解决 JOIN 关系？

Text2SQL 系统解决 JOIN 问题主要通过以下几个关键步骤：

### 1. Schema 理解与建模

系统首先需要理解数据库的结构：
- **表结构**：有哪些表，每个表有哪些列
- **主键/外键关系**：表之间通过哪些字段关联
- **语义信息**：列名、表名的含义

通常会构建一个 **Schema Graph（模式图）**，将表作为节点，外键关系作为边。

### 2. 实体识别与表映射

当用户输入自然语言问题时，系统需要：

| 用户表述 | 映射结果 |
|---------|---------|
| "张三" | students.name |
| "班主任" | classes.teacher |

系统识别出问题涉及多个表，就知道需要 JOIN。

### 3. JOIN 路径推理

这是核心难点。常见的解决方案包括：

**a) 基于图的路径搜索**
- 在 Schema Graph 上找两个表之间的最短路径
- 路径上的边就是 JOIN 条件

```
students --[class_id]--> classes
```

**b) 基于神经网络的关系预测**
- 使用 GNN（图神经网络）学习表之间的关联
- 模型预测哪些表需要 JOIN，以及 JOIN 条件

**c) 基于大语言模型的推理**
- 现代 LLM 方法（如基于 GPT/Claude）直接将 schema 信息和问题一起输入
- 模型通过上下文理解推断出正确的 JOIN 关系

### 4. 典型模型的处理方式

| 模型类型 | JOIN 处理方法 |
|---------|--------------|
| **Spider/SParC 基准模型** | 使用 Schema Linking + Graph Encoder |
| **RAT-SQL** | 关系感知的 Transformer，显式建模表-列-问题之间的关系 |
| **BRIDGE** | 结合 Schema 和问题的联合编码 |
| **ChatGPT/Claude 等 LLM** | 通过 Prompt 提供 Schema 信息，利用推理能力生成 JOIN |

### 5. 一个完整的处理流程示例

用户问题："**列出所有销售额超过100万的门店经理姓名**"

```
步骤1: 实体识别
  - "销售额" → sales.amount
  - "门店" → stores
  - "经理姓名" → managers.name

步骤2: 发现涉及3个表，需要JOIN

步骤3: 查找外键关系
  - stores.manager_id → managers.id
  - sales.store_id → stores.id

步骤4: 生成SQL
SELECT m.name 
FROM managers m
JOIN stores s ON m.id = s.manager_id
JOIN sales sa ON s.id = sa.store_id
GROUP BY m.name
HAVING SUM(sa.amount) > 1000000;
```

---

## 三、主要挑战

1. **多跳 JOIN**：问题涉及多个中间表时，路径选择困难
2. **歧义消解**：多条可能的 JOIN 路径，需要选择语义正确的
3. **隐式关系**：用户没有明确提到某些表，但逻辑上必须经过
4. **自连接**：同一张表需要 JOIN 自己的情况
