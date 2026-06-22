# AI Engineer Journey

> 新能源制造业 IT 负责人，2026.05 起转型 AI 应用工程 / AI 产品方向。
> 这个仓库记录我的 2-3 年转型实战：代码、踩坑、复盘、成长。

## 我是谁

- 30 岁，坐标上海
- 制造业 IT 负责人（团队 6 人，业务系统 + 数字化）
- 过往项目经验：Unity 独立游戏、语音 APP（全栈）
- 2026.05.06 起，执行 2-3 年 AI 转型计划

## 目标

- **2026.05 - 2027.04**：12 周筑基 + 9 个月作品集 + 求职 → AI 应用工程师 / AI 产品经理
- **2027.05 - 2028.04**：入职 AI native 公司
- **长期**：用 AI 把自己迁移到更有杠杆的位置

## 仓库结构

```
ai-engineer-journey/
├── w1d1-hello/          # W1D1: GLM-4 + Claude CLI + RAG 集成（LlamaIndex）
├── w1d3-rag/            # W1D3: LlamaIndex quickstart 跑通
└── w3d1-hybrid/         # W3D1: vector + BM25 + RRF 混合检索（PR #1）
```

每个 `w{周}d{天}-xxx/` 是当天的实战产出，带独立 README 与运行说明。

## 当前进度

### W1（2026.05.06 - 05.12）— 环境搭建 + 第一次 API 调用

- ✅ Claude + GLM-4 CLI 跑通（streaming + 单轮）
- ✅ LlamaIndex RAG 集成（`--rag` 开关 + 引导输入）
- ✅ 自定义 PromptTemplate + 占位符校验 + 命中日志（修复主观题"瞎编"）
- 🚀 这个 repo 上线

### W2（2026.05.13 - 05.19）— 出差期，零代码进度（计划内 buffer）

重庆出差 + 周末休息消化。计划内允许的 buffer 周。

### W3（2026.05.18 - 05.24，进行中）— RAG 进阶：混合检索 + 题型路由

- ✅ **W3D1（5.18）**：vector + BM25 + RRF 三模式混合检索。手撸 RRF 不调 LlamaIndex 黑盒，16KB 中文语料 4 测试压测，RRF 分数与公式 `1/(k+rank)` 精确对上。详见 [`w3d1-hybrid/README.md`](w3d1-hybrid/README.md)。
- 🔍 **W3D1 意外发现**：大语料压测中 **检索 4/4 全命中，但生成层 2/4 拒答**——`qa_template` 严格度对事实题过严。详细拆解：[拆解笔记 #1（即刻 / V2EX / 小红书）](#)。
- ✅ **W3D4（5.21）题型路由落地**：`prompts/qa_factual.txt`（宽松）+ `qa_subjective.txt`（严格）+ `[rag.templates]` 配置化路由表 + CLI `--question-type`，日志加 `template=` 字段（为按题型分组评估铺路）。
- 🔄 **W3D4 两个修正**（同一晚翻车两次，记下来）：
  1. **"事实宽松 / 主观严格" 二分法被自己跑的 4×2 矩阵打脸**：factual 模板 4/4 全答出（含主观题），subjective 模板 2/4 拒答（含主观题）。真实的轴不是题型，是 prompt 严格度本身。
  2. **W3D1 起 BM25 一直是哑的**：清 `bm25s tokenizer` deprecation warning 时翻源码发现 `BM25Retriever.from_defaults(tokenizer=callable)` 是 **silent-drop**，jieba 从未生效，BM25 一直用默认英文 token_pattern 切中文。改用 `(?u)\w+|[一-鿿]` + `skip_stemming=True` 后 Test 2 BM25 score：0 → 2.028。**5.18 README 中"BM25 score 全是 0 = IDF 退化" 的诊断是错的**，真因在这里。详见 [`w3d1-hybrid/README.md`](w3d1-hybrid/README.md) 学习笔记段。
- 📅 W3D5-D7 计划：B. cross-encoder reranker（5.23）+ C. Chroma 持久化索引（5.22 + 5.23 切两段）+ 拆解笔记 #2 自我修正版（5.24）。

## 为什么公开

对外学习的压力 → 真实表达 → 找到同路人。
失败和踩坑（包括把自己的诊断推翻）也会留在这里，这是过程的一部分。

---

*Last updated: 2026-05-21*
