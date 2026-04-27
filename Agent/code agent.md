1、负责代码智能体（Coding Agent）系统的全流程研发：数据构建（代码语料清洗、代码/对话/工具轨迹标注）、预训练/持续预训练、指令微调（SFT）与后训练（DPO/ORPO/RLHF/RLAIF），在 HumanEval、MBPP、SWE-bench、RepoBench 等基准及业务集上达成目标。 

2、基于主流 Agents 框架（LangChain、AutoGen、Agno 等）搭建具备规划、工具调用、记忆与多步推理能力的多代理系统，落地代码生成、调试与单测生成、仓库级理解、自动修复、PR 评审与CI修复等核心场景。 

3、以 MCP（Model Context Protocol）为核心构建工具生态：设计与实现 MCP servers/tools，标准化接入代码搜索、编译/运行、CI/CD、Issue/PR、知识库与内部服务；支持安全沙箱执行与资源隔离，提升可观测与可维护性。 

4、设计并优化 ReAct/Plan-Act/Tree-of-Thought 等推理策略及函数调用，结合 RAG（向量/符号/结构化检索与代码图）实现大规模代码库检索增强，支持跨文件、跨仓库的复杂问题求解。 

5、推理与部署优化：蒸馏与量化（AWQ/GPTQ/GGUF）、图编译与高效推理（vLLM、TGI、TensorRT-LLM）、并发与缓存优化（PagedAttention、speculative decoding、prompt caching）、Agent 执行器与轨迹缓存加速，满足低延时与高并发。 

6、产品化与集成：将 Agent 接入 IDE（VSCode/JetBrains）、Git 平台（GitHub/GitLab/Gerrit）、CI（Jenkins/GitHub Actions）与内部开发平台；完善API/SDK、鉴权、灰度、监控告警与回滚策略，保障稳定上线。 

7、评测与数据闭环：搭建自动评测体系（******、执行成功率、修复成功率、TTFR/TTFX 等），采集交互日志、失败轨迹与检索质量指标，构建合成/自监督数据与轨迹蒸馏管线，驱动持续迭代。 

8、合规与安全：权限与数据隔离、依赖与供应链安全扫描、提示注入/越权工具调用防护、审计与可追踪性建设（tracing/telemetry）。 



9了解大模型训练与推理：可使用 PyTorch/DeepSpeed/Lightning/Colossal-AI 等完成 SFT/LoRA/QLoRA；理解后训练与对齐方法；熟悉 Transformers/vLLM/TGI 等高效推理与服务化。