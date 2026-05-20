# W3D1: 混合检索（Hybrid Retrieval）

> 转型 AI 工程师 W3D1 项目。在 W1D1 的 LlamaIndex 纯向量 RAG 基础上，做**向量 + BM25 + RRF 排名融合**的混合检索改造。

## 这是什么

W1D1 的 RAG 流程是隐式的：
```
docs → VectorStoreIndex.from_documents → as_query_engine → answer
```
向量检索藏在 `index.as_query_engine()` 里面。中文里关键词命中（人名、术语、id）这种"应该 BM25 一发入魂"的查询，在纯向量下经常被语义距离冲淡。

W3D1 把这层封装拆开：
```
docs
  → SentenceSplitter.get_nodes_from_documents → nodes (显式)
  ├─→ VectorStoreIndex(nodes) → VectorIndexRetriever → 向量 top_k
  └─→ BM25Retriever.from_defaults(nodes, jieba)      → BM25 top_k
     ↓
  reciprocal_rank_fusion(k=60)  → 融合后 top_k
     ↓
  ResponseSynthesizer(qa_template) → answer
```

三种模式都能跑：
- `--retriever vector`：只走向量（兼容原 W1D1 行为）
- `--retriever bm25`：只走关键词（jieba 分词）
- `--retriever hybrid`：两路同时跑 + RRF 融合（**主推**）

## 技术栈

- Python 3.11+（用 `tomllib`，3.10 及以下需自己装 `tomli`）
- `llama-index`（核心 RAG 框架）
- `llama-index-llms-zhipuai` + `llama-index-embeddings-zhipuai`（智谱 LLM + embedding）
- `llama-index-retrievers-bm25`（BM25 检索器）
- `jieba`（中文分词，喂给 BM25）
- `anthropic` / `openai`（Claude / GLM 普通对话保留下来）

## 快速开始

```bash
cd w3d1-hybrid
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env 填 GLM_API_KEY

mkdir -p data
# 把任意 .md / .txt 文件丢进 data/
```

## 三种检索模式对照实验

假设 `data/` 里有一份中文文档 `lzy.md`，问"李哲羽今年多大？"

### 1. 纯向量（W1D1 行为）

```bash
python hello-world.py --rag --retriever vector --question "李哲羽今年多大？" --log-level INFO
```

预期日志：
```
[INFO] 切出 1 个 node（chunk_size=512, overlap=50）
[INFO] 已构建向量 retriever（top_k=3）
[INFO] retriever=vector 命中 1 个 node
[INFO]   [1] vec_score=0.587 text=李哲羽,30 岁,上海人,目前在新能源电机制造业做 IT 负责人...
```

### 2. 纯 BM25（关键词命中）

```bash
python hello-world.py --rag --retriever bm25 --question "李哲羽今年多大？" --log-level INFO
```

预期日志：
```
[INFO] 已构建 BM25 retriever（top_k=3, tokenizer=jieba）
[INFO] retriever=bm25 命中 1 个 node
[INFO]   [1] bm25_score=2.341 text=李哲羽,30 岁...
```

注意 `bm25_score` 是 BM25 算法的相关性分数（绝对值无意义，只能横向比较；不与 vec_score 在同一尺度上），这也是为什么我们用 **RRF 而不是简单加权**。

### 3. 混合检索（RRF 融合）— **主推**

```bash
python hello-world.py --rag --retriever hybrid --question "李哲羽今年多大？" --log-level INFO
```

预期日志（三路全打）：
```
[INFO] 切出 1 个 node
[INFO] 已构建向量 retriever（top_k=3）
[INFO] 已构建 BM25 retriever（top_k=3, tokenizer=jieba）
[INFO] retriever=hybrid 单路向量召回 1 个
[INFO]   vec[1] score=0.587 text=李哲羽,30 岁...
[INFO] retriever=hybrid 单路 BM25 召回 1 个
[INFO]   bm25[1] score=2.341 text=李哲羽,30 岁...
[INFO] retriever=hybrid 融合后取 1 个（RRF）
[INFO]   fused[1] vec=#1 bm25=#1 rrf_score=0.0328 text=李哲羽,30 岁...
```

最后一行 `vec=#1 bm25=#1 rrf_score=0.0328` 就是核心读点：这个 node 在向量路排第 1，在 BM25 路也排第 1，融合分数 = 1/(60+1) + 1/(60+1) = 0.0328。

如果一个 node 只在某一路命中（比如纯靠关键词或纯靠语义），对应位置会显示 `—`，例如 `vec=#2 bm25=—`。

## 大语料压测：4 个真实业务场景

短 demo 题只能验证流程跑通，要看混合检索的实际表现，需要多 chunk 的大语料。`data/hybrid_retrieval_test.md` 是一份 16KB 中文测试语料（135 行 / 15 章节），模拟制造企业内部档案，含设备报警码、事故记录、供应商交期、项目复盘。

**测试条件**：`chunk_size=512` 切出 18 个 node，`top_k=5`，全部 `hybrid` 模式。

### 4 个测试结果

| # | 问题 | 检索命中（top-1） | RRF 分数 | LLM 答案 | 评估 |
|---|---|---|---|---|---|
| 1 | VFD-F17 是什么报警？ | `vec=#1 bm25=#1` | 0.0328 | "当母线电压超过安全阈值时，变频器触发过压保护..." | ✅ |
| 2 | E-7429 是什么事故？发生在什么时候？ | `vec=#1 bm25=#1` | 0.0328 | ❌ "无法从已提供资料中判断" | ⚠️ 拒答 |
| 3 | 哪些供应商可能影响下月交付？ | `vec=#3 bm25=#3` | 0.0317 | ❌ "无法从已提供资料中判断" | ⚠️ 拒答 |
| 4 | RAG 模型为什么会胡编内容？ | `vec=#1 bm25=#2` | 0.0325 | "回答模板太宽松，模型仍倾向于补充常识..." | ✅ |

### RRF 数学验证（实测对上公式）

- Test 1 / 2：`fused[1] = 1/(60+1) + 1/(60+1) = 0.0328` ✓（两路都排第 1）
- Test 4：`fused[1] = 1/61 + 1/62 = 0.0325` ✓（向量第 1 + BM25 第 2）
- Test 1 / 2：`fused[3] = 1/62 = 0.0161` ✓（单路命中）

公式 `1/(k+rank)` 在大语料场景下行为完全符合预期。

### 关键发现：检索 ✅ ≠ 生成 ✅

**检索层 4/4 全部命中正确 chunk。生成层却只有 2/4 给出答案。**

- **Test 2**：原文白纸黑字写着"事故编号 E-7429 发生在 2026 年 5 月 17 日晚间"，问题用"事故"原文用"故障记录 / 故障现象"，字面不完全匹配，LLM 在 `qa_template` "凡是资料中没有直接出现、无法被原文明确支持的内容，一律视为不知道" 的约束下选择拒答。
- **Test 3**：这恰好是向量检索本该发光的场景（语义改写：原文"到货窗口后移" vs 问题"影响下月交付"），向量召回也确实命中了正确片段，但被 prompt 严格度抹杀。

**洞察**：RAG 不是一个 prompt 搞定所有场景的活——**事实题需要宽松让 LLM 敢答，主观题需要严格不让它乱编，这两件事应该分开**。当前 `qa_template` 是 W2 修"主观题瞎编"时引入的强约束，治好了瞎编，但反过来咬了事实题。下一步：分层 `qa_template`（事实型 / 主观型双模板）。

## 关键配置（hello-world.toml `[rag]` 段）

| 字段 | 默认 | 说明 |
|---|---|---|
| `retriever` | `"hybrid"` | `vector` / `bm25` / `hybrid` |
| `top_k` | `3` | 每路取前 K + 融合后取前 K |
| `chunk_size` | `512` | 文档切片大小 |
| `chunk_overlap` | `50` | 切片重叠（避免边界信息断裂） |
| `bm25_tokenizer` | `"jieba"` | `jieba`（中文）或 `char`（按字切） |
| `rrf_k` | `60` | RRF 平滑常数，越大越平滑 |

命令行同名参数（`--retriever` / `--chunk-size` / `--chunk-overlap` / `--bm25-tokenizer` / `--rrf-k`）会覆盖配置文件。

## RRF 是怎么算的

[`reciprocal_rank_fusion`](hello-world.py) 完全手写在代码里，不调任何黑盒。核心就一行：

```python
score_map[nid] = score_map.get(nid, 0.0) + 1.0 / (k + rank)
```

每个 node 在每条检索路里贡献 `1/(k+rank)`，多路相加。`k=60` 是 Cormack 等人 2009 年原始论文给的经验值，目的是让 rank=1 和 rank=2 的差距没那么悬殊（rank=1 贡献 1/61 ≈ 0.0164，rank=2 贡献 1/62 ≈ 0.0161），从而允许"两路都中等"的 node 战胜"一路第一另一路没中"的 node。

之所以选 RRF 而不是 `α*vec + (1-α)*bm25` 加权，是因为：
- **不需要做分数归一化**：向量是余弦相似度 [0,1]，BM25 是无上界的 TF-IDF 变体，两者尺度天差地别。
- **对每路 retriever 内部分数分布不敏感**：只看排名，不看具体分数。
- **业界默认**：Elastic、Weaviate、LangChain、LlamaIndex 内部默认都是 RRF。

## 学习笔记

### 5.18 — 把"node"从 LlamaIndex 隐式封装里抠出来
- 原 `VectorStoreIndex.from_documents` 一步走完，nodes 是黑盒。
- 拆出来 `SentenceSplitter.get_nodes_from_documents` 之后，**同一份 nodes 可以喂给多个 retriever**——这是混合检索能成立的前提。
- 顺手把 `chunk_size` / `chunk_overlap` 暴露出来，发现 512/50 对短文档明显过大，单文档切不出多个 node，hybrid 跟纯向量结果一样（因为候选只有 1 个）。验证混合检索效果需要多 chunk 的语料。

### 5.18 — RRF 手写不调库
- LlamaIndex 自带 `QueryFusionRetriever`，但内部融合后只暴露最终 score，看不到原始两路 rank。
- 手写之后能在日志里精确打出 `vec=#? bm25=#? rrf_score=?`，反向调试每个候选"为什么进 top_k"。
- 走的坑：忘记给"没命中"的 rank 留 0，导致 rank_map 长度不一致。修正：每个 node 首次出现时先初始化 `[0] * n_lists`。

## 反思

[W3 复盘后填，见 `progress/W3-overview.md`]

---

*Part of [ai-engineer-journey](../README.md). 上一篇：[w1d1-hello](../w1d1-hello/README.md)。*
