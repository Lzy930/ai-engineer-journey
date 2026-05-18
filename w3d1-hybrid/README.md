# W1D1: Claude + GLM CLI with RAG

> 转型 AI 工程师第 1 周第 1 天的项目。从 0 到 1 跑通大模型 API,并集成 LlamaIndex RAG。

## 功能

- ✅ **双 provider 支持**:Claude(Anthropic)+ GLM-4(智谱)
- ✅ **Streaming 输出**:两个 provider 都支持流式响应
- ✅ **命令行交互**:`--provider` / `--key-source` / `--rag` / `--data-dir` 等参数
- ✅ **RAG 检索增强**:基于 LlamaIndex + ZhipuAI Embedding 的本地知识库问答
- ✅ **自定义 PromptTemplate**:强制基于上下文回答,修复主观题"瞎编"
- ✅ **占位符校验**:template render 前校验 `{context_str}` / `{query_str}` 完整性
- ✅ **命中日志**:打印检索到的 chunks,方便调试 RAG 命中质量

## 技术栈

- Python 3.11
- `anthropic`(Claude API)
- `zhipuai`(GLM-4 API)
- `llama-index` + `llama-index-llms-zhipuai` + `llama-index-embeddings-zhipuai`(RAG)
- `python-dotenv`(环境变量)

## 快速开始

```bash
# 1. 进入目录
cd w1d1-hello

# 2. 创建虚拟环境并激活
python -m venv .venv
source .venv/bin/activate  # macOS / Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API key(把 .env.example 复制为 .env 并填入你的 key)
cp .env.example .env
# 编辑 .env,填入 GLM_API_KEY 或 ANTHROPIC_API_KEY

# 5. 准备 RAG 知识库(可选,仅 --rag 模式需要)
mkdir -p data
# 把任意 .md / .txt 文件丢进 data/ 即可(data/ 已在 .gitignore 中,不会上传)

# 6. 运行
python hello-world.py                       # 交互模式
python hello-world.py --provider glm --rag  # GLM + RAG
```

## 学习笔记

### 5.6 — Hello World 跑通(超额)
首次跑通 Claude + GLM-4 API,从单轮调用做到 streaming,踩坑 3 次:
- 环境变量加载路径(`.env` vs 命令行 export)
- Streaming 事件解析(Claude messages.stream vs OpenAI 风格 chunk)
- 中文编码与 max_tokens 的关系

### 5.8 — LlamaIndex RAG 集成(超额)
原计划"扫盲 + quickstart",实际把 RAG 集成进了 W1D1 的 CLI:
- 加 `--rag` 开关 + 数据目录引导输入
- 用 ZhipuAI 系列 embedding(`embedding-3`)+ GLM-4 LLM
- 3 组对照实验:有 RAG / 无 RAG / 不同 top_k

### 5.9 — 修复主观题"瞎编"
发现 RAG 检索成功但模型生成时仍瞎编。一晚解决:
- **自定义 PromptTemplate**:强制 "If the answer is not in the context, say you don't know"
- **占位符校验**:template render 前先验 `{context_str}` 和 `{query_str}` 都在,缺失就抛
- **模板外置**:prompt 写在配置里,不写死在代码里
- **命中日志**:`_log_rag_sources` 打印每次检索到的 chunks,反向验证

## 反思

[W1 复盘后填,见 progress/W1-overview.md]

---

*Part of [ai-engineer-journey](../README.md).*
