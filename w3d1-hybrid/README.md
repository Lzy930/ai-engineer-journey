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

**洞察**：RAG 不是一个 prompt 搞定所有场景的活——**事实题需要宽松让 LLM 敢答，主观题需要严格不让它乱编，这两件事应该分开**。当前 `qa_template` 是 W2 修"主观题瞎编"时引入的强约束，治好了瞎编，但反过来咬了事实题。**下一步在 W3D4 落地**：分层 `qa_template` + 题型路由。

## W3D4：题型路由（factual / subjective）

把 5.18 发现的"prompt 太严"问题拆成两个工件：

**1. prompts/ 目录拆出两份模板**

```
prompts/qa_factual.txt    宽松：允许同义/近义/改写匹配，事实题用
prompts/qa_subjective.txt 严格：凡资料未明确出现一律拒答，主观/高风险题用
```

**2. `[rag.templates]` 路由表（hello-world.toml）**

```toml
default_question_type = ""   # 不指定时退化到 toml.qa_template 字段（向后兼容）

[rag.templates]
factual    = "prompts/qa_factual.txt"
subjective = "prompts/qa_subjective.txt"
```

**3. CLI 加 `--question-type {factual|subjective}`**

```bash
python hello-world.py --rag --retriever hybrid --rag-top-k 5 \
  --question-type factual \
  --question "E-7429 是什么事故？发生在什么时候？"
```

**模板优先级**：`--rag-qa-template-file` > `--question-type` > `[rag].qa_template` > LlamaIndex 默认。日志里会打出 `template=question_type=factual -> prompts/qa_factual.txt`，方便后续按题型分组评估。

### 4 题 × 2 题型 回归对比（W3D4）

同样 `top_k=5 / hybrid / chunk_size=512`，跑脚本 `run_question_type_compare.sh`：

| # | 问题 | factual（宽松） | subjective（严格） |
|---|---|---|---|
| 1 | VFD-F17 是什么报警？ | ✅ 答出 | ✅ 答出 |
| 2 | E-7429 是什么事故？ | ✅ **救回**（5.18 拒答 → 通过） | ✅ 通过（不稳定，会受 LLM 随机性影响） |
| 3 | 哪些供应商可能影响下月交付？ | ⚠️ **半救回**：诚实回答"资料中未提及具体供应商名称" | ❌ 拒答（与 5.18 一致） |
| 4 | RAG 模型为什么会胡编内容？ | ✅ 答出（含 5.9 修复方案细节） | ❌ **意外拒答**（5.18 通过 → 今天拒答） |

**修正版洞察**（覆盖 5.18 的"事实宽松、主观严格"二分法）：

1. **factual 模板表现最稳**：4/4 全部给出有效回答，且 Test 3 没瞎编（诚实承认资料里只有间接表述）—— **prompt 设计中"允许语义改写 + 强制诚实兜底"是有效组合**。
2. **subjective 模板"太严"**：不仅误杀事实题（Test 2/3），还可能误杀本应回答的主观题（Test 4）。当前这套是 5.9 为修瞎编引入的，**适用场景应缩到"法律/医疗/合规"等高风险场景**，不是普通主观题。
3. **"主观题"应该再细分**：普通主观题需要"允许基于资料推理 + 禁止外部知识"的中间档 prompt（未来 `comparison` / `causal` 题型留好了路由位）。

### 为什么要做"题型路由"而不只是"两份 prompt 文件"？

把"prompt 选择"从"裸文件路径"升级成"业务概念"，意义有 4 层：

1. **语义化抽象**：使用者只回答"我问的是什么类型的问题"，不关心实现文件路径。
2. **可扩展**：新增题型只在 `[rag.templates]` 加一行，业务侧零改动。
3. **可评估**：题型标签是按题型分组算成功率的前置条件，是 W4 ragas 评估的输入。**整体准确率告诉不了你"该改 prompt 还是该改 retriever"**。
4. **可演进**：今天 `--question-type` 是手动指定（V1）；V2 可以接一个便宜小模型做题型分类 agent，从手动 RAG 走向 Agent-based RAG，路由表不动。

## W3D5：Chroma 持久化（C1 阶段）

W3D1 - W3D4 的所有 RAG 索引都建在内存里（LlamaIndex 默认的 `SimpleVectorStore`）。每次启动 → 切 nodes → 算 embedding → 建索引 → 进程退出，索引全没。16KB 测试语料 18 个 node 跑一次要发 22 次 `/embeddings` 请求，生产场景下每次重算又慢又费 token。

W3D5 把向量存储换成 [Chroma](https://www.trychroma.com/) 的 `PersistentClient`，embedding 落盘到本地目录，**跑一次能看见数据真的在那**。

### C1 / C2 切两段（W3D5 / W3D6）

- **C1（本次）**：只做 ingest + persist。第一次跑 → 算 embedding → 写入 `./chroma_db/`；**第二次跑仍会重新切 + 重算 embedding 再写入同一 collection**（等价"每次重建索引但写到磁盘"）。功能上现在还看不到"省 token"的收益，但**持久化通路通了** —— 下一步只需在 query 时跳过 ingest 直接 load 就能拿到真正的索引复用。
- **C2（W3D6 5.23）**：query-time 直接从 `./chroma_db/` 加载，跳过切片+嵌入，验证 query 结果与 C1 首次一致。这才是真省 token 的形态。

切两段的理由：W3D5 周五降级日预算只有 60 分硬上限，做不完一整个回路。先把"会写"做扎实，周六再做"会读"。

### 启用方式

```bash
python hello-world.py --rag --retriever hybrid --chroma \
  --question "VFD-F17 是什么报警？" --log-level INFO
```

或在 `hello-world.toml` 里：

```toml
[rag.chroma]
enabled = true
persist_dir = "./chroma_db"
collection_name = "rag_default"
```

CLI 覆盖配置文件：`--chroma` / `--no-chroma` / `--chroma-persist-dir` / `--chroma-collection`。日志里会看到一行 `已启用 Chroma 持久化向量库（persist_dir=./chroma_db, collection=rag_default, mode=ingest+persist）`，能直接确认走的是哪条路。

### 跑完磁盘上有什么

```
chroma_db/
├── chroma.sqlite3                              # ~1.2 MB，collection metadata + embedding 引用
└── 52873757-a65b-4648-80f9-6425960d76fe/       # collection UUID 目录（HNSW 向量索引）
```

`chroma.sqlite3` 是 Chroma 用 SQLite 维护的 collection / document / embedding 元数据；UUID 子目录里是底层 HNSW 向量索引文件。两路 retriever 行为对比（同一份 `data/lzy.md`、`chunk_size=200`、`top_k=1`、问"李哲羽多大?"）：

| 模式 | 首次跑用时 | embedding 请求数 | 磁盘落盘 | 答案 |
|---|---|---|---|---|
| `--no-chroma`（W3D4 行为） | ~11s | 1（只算 query） | 无 | "30 岁" |
| `--chroma`（W3D5 C1） | ~74s | 22（22 个 node 全部 ingest）| `chroma_db/` 2.5 MB | "30 岁" |

C1 首次跑慢是预期 —— 把所有 node 都送 embedding 接口才能写盘。**省 token 的形态在 C2 才能看到**：二次启动直接 load 已落盘的 embedding，跳过整个 ingest 阶段。

### 工程小决策

- **`get_or_create_collection`** 而非 `create_collection`：容许同一 collection 多次 ingest（开发期友好）；生产场景需要"幂等 ingest"要换成基于文件 hash 的去重，留给 W4 ragas 评估改造一起做。
- **`PersistentClient` 而非 `Client`**：前者立即把所有写入落盘（WAL），后者只在显式 `persist()` 时落盘。`PersistentClient` 对"跑完关进程不丢"更稳。
- **`chroma_db/` 进 .gitignore**：索引文件 + sqlite3 不进仓（占空间、二进制、依赖本地路径）。要恢复时 `--chroma` 重跑一次即可重建。
- **默认 `[rag.chroma].enabled = false`**：新增字段不破坏现有 hybrid 路径，需要持久化时显式开启。

### 顺手清的两项 W3D1 backlog

- **backlog #10**：`requirements.txt` 锁版本 —— 原来是 `llama-index>=0.11` 这类宽松范围，现在全部锁到 `==<具体版本号>`（基于 `pip freeze` 实测装好的版本），保证 reproducible。
- **backlog #11**：`llama-index-readers-file` 显式声明 —— 原依赖 `llama-index` 主包带入（transitive 安装、版本随主包飘），现显式声明 `==0.6.0`。今天 ingest 走的是 `SimpleDirectoryReader`，未来要读 PDF / docx 时这条声明就生效了。

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

### 5.21 — BM25 真相反转：jieba 从来没生效过

清理 5.18 backlog #4（`tokenizer` 参数 deprecation warning）时翻源码，发现一个 5.18 没看到的真相：

`BM25Retriever.from_defaults(tokenizer=callable)` 这个参数虽然接收，但**只打 warning 后被静默丢弃**，从来没传给底层 `bm25s` 构造器。也就是说 W3D1 以来配置的 jieba 分词从来没生效过。BM25 一直在用 `bm25s` 默认的英文 token_pattern `(?u)\b\w\w+\b`，对中文几乎切不出任何 token。

这就是 5.18 README "BM25 score 全是 0（IDF 退化）" 一节的**真实底层原因 —— 不是 IDF 退化，是根本没切到中文词**。当时之所以 hybrid 还能 4/4 命中正确 chunk，是因为：（1）问题里都含 `VFD-F17` / `E-7429` 这种英文+数字 token，默认 pattern 切得出；（2）BM25 返回 top_k 时即使 score=0 也会 fall back 返回前几条，碰巧排序和 vector 一致。**hybrid 看起来在工作，其实 BM25 那一路是"无信息地猜对了"**。

修法（最小改动，0 额外依赖）：

```python
BM25_TOKEN_PATTERN_CN = r"(?u)\w+|[\u4e00-\u9fff]"  # 英文/数字按 word + 中文按字

BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=top_k,
    token_pattern=BM25_TOKEN_PATTERN_CN,
    skip_stemming=True,  # 中文不需要 stemming
    language="en",       # stopwords 用英文表（中文不会命中）
)
```

修复前后对比（Test 2 `E-7429 是什么事故？` 在纯 BM25 模式下）：

| 时间 | bm25[1] score | 说明 |
|---|---|---|
| 5.18（修复前） | ≈ 0 | jieba 没生效，默认 pattern 切不出中文 |
| 5.21（修复后） | **2.028** | 真正基于 IDF 的命中 |

更进一步的"真正词级中文 BM25"（jieba 包装 retriever，corpus + query 两侧都预切）实现起来更重，留给 W4。当前 char-level 中文 BM25 已经足够支撑 hybrid 检索（实测 4/4 命中保持）。

**给读者的提醒**：用 LlamaIndex / 任何 RAG 框架的高层 API 时，**任何带 deprecation warning 的参数都值得查一下源码确认有没有被静默吞掉**。这次踩坑成本 = 几乎重写一次拆解笔记的检索层故事。

### 5.21 — qa_template 分层 + 题型路由
- 5.18 测试时 4/4 检索命中但 2/4 LLM 拒答，根因是单一 prompt 同时管事实题 + 主观题相互打架。今晚把它落成代码：拆 prompt 文件 + 加 `--question-type` 路由表。
- 跑完发现一个 5.18 没看到的现象：**严格 prompt 不仅误杀事实题，也会误杀主观题**（Test 4 在严格模板下今天反而拒答了）。所以"事实宽松 / 主观严格"二分法不够，至少要预留"中间档 prompt"位置。路由表 `[rag.templates]` 加一行就行，验证了"配置化"的扩展性。
- 把"模板选择"从命令行裸文件路径升级到业务概念（factual / subjective）后，**有了题型标签才能做按题型分组的评估**（W4 ragas 的前置条件）。这是路由表真正的长期价值，不只是写起来短。

## 反思

[W3 复盘后填，见 `progress/W3-overview.md`]

---

*Part of [ai-engineer-journey](../README.md). 上一篇：[w1d1-hello](../w1d1-hello/README.md)。*
