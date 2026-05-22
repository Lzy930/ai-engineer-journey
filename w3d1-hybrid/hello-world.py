"""W1D1 hello-world — 交互式选择 Claude / GLM 并打印错误。

用法：
- 直接运行：`python hello-world.py`（会交互式选择提供方与 API Key 来源）
- 或用参数：`python hello-world.py --provider claude --key-source env`

需要的环境变量（可选）：
- Claude：`ANTHROPIC_API_KEY`
- GLM：`GLM_API_KEY`（智谱的 API Key）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from dataclasses import dataclass
from getpass import getpass
from typing import Any, Optional

from dotenv import load_dotenv

DEFAULT_PROMPT = "用一句话欢迎我——一位制造业 IT 负责人，今天迈出了转型 AI 应用工程师的第一步。"

logger = logging.getLogger("hello-world")


@dataclass(frozen=True)
class RunResult:
    provider: str
    model: str
    text: str
    usage: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None


def _mask_key(key: str) -> str:
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}...{key[-4:]}"


def _prompt_choice(title: str, options: list[tuple[str, str]], default: str) -> str:
    print(title)
    for i, (value, label) in enumerate(options, start=1):
        d = " (默认)" if value == default else ""
        print(f"  {i}. {label}{d}")
    while True:
        raw = input("请输入序号并回车：").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("输入无效，请重试。")


def _get_api_key(provider: str, key_source: str) -> str:
    env_var = "ANTHROPIC_API_KEY" if provider == "claude" else "GLM_API_KEY"
    if key_source == "env":
        key = os.getenv(env_var, "").strip()
        if not key:
            raise RuntimeError(f"未检测到环境变量 {env_var}，请先设置或改用手动输入。")
        return key
    if key_source == "paste":
        key = getpass(f"请粘贴 {provider.upper()} API Key（输入不可见）：").strip()
        if not key:
            raise RuntimeError("你没有输入 API Key。")
        return key
    raise ValueError(f"未知 key_source: {key_source}")


def _read_config(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"读取配置文件失败：{path}（{type(e).__name__}: {e}）") from e


def _cfg_get(cfg: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _coerce_message_text(content: Any) -> str:
    """
    OpenAI SDK 的 message.content 通常是 str，但有些兼容实现会返回 list[part]。
    这里做最大兼容：尽量从常见结构里提取文本。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if isinstance(p, dict):
                # 常见：{"type":"text","text":"..."}
                t = p.get("text") or p.get("content") or p.get("value")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    if isinstance(content, dict):
        t = content.get("text") or content.get("content") or content.get("value")
        return t if isinstance(t, str) else ""
    return ""


def call_claude(api_key: str, model: str, prompt: str, max_tokens: int) -> RunResult:
    from anthropic import Anthropic
    from anthropic import BadRequestError

    try:
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", None),
            "output_tokens": getattr(resp.usage, "output_tokens", None),
        }
        return RunResult(provider="claude", model=resp.model, text=text, usage=usage)
    except BadRequestError as e:
        request_id = getattr(e, "request_id", None)
        detail = getattr(e, "body", None)
        raise RuntimeError(
            f"Claude 请求失败：{e}（request_id={request_id}，detail={detail}）"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Claude 请求失败：{type(e).__name__}: {e}") from e


def stream_claude(api_key: str, model: str, prompt: str, max_tokens: int) -> RunResult:
    from anthropic import Anthropic
    from anthropic import BadRequestError

    try:
        client = Anthropic(api_key=api_key)
        text_parts: list[str] = []
        usage: dict[str, Any] | None = None

        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                # 只处理文本增量，边来边打印
                if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                    chunk = getattr(event.delta, "text", "") or ""
                    if chunk:
                        print(chunk, end="", flush=True)
                        text_parts.append(chunk)
                # 最终 usage 在 message_stop 后可取到
            final_msg = stream.get_final_message()
            u = getattr(final_msg, "usage", None)
            if u is not None:
                usage = {
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                }

        print()  # 结束补一个换行
        text = "".join(text_parts)
        return RunResult(provider="claude", model=getattr(final_msg, "model", model), text=text, usage=usage)
    except BadRequestError as e:
        request_id = getattr(e, "request_id", None)
        detail = getattr(e, "body", None)
        raise RuntimeError(
            f"Claude 请求失败：{e}（request_id={request_id}，detail={detail}）"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Claude 请求失败：{type(e).__name__}: {e}") from e


def call_glm(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    base_url: str,
    debug: bool = False,
) -> RunResult:
    # GLM 提供 OpenAI 兼容接口（base_url）
    from openai import APIStatusError, OpenAI

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        text = ""
        if resp.choices:
            msg = resp.choices[0].message
            # OpenAI 兼容：优先 content，其次尝试常见的 reasoning 字段（不同厂商命名不同）
            text = _coerce_message_text(getattr(msg, "content", None)).strip()
            if not text:
                text = _coerce_message_text(getattr(msg, "reasoning_content", None)).strip()
            if not text and debug:
                try:
                    dumped = msg.model_dump()
                except Exception:
                    dumped = {"message_repr": repr(msg)}
                logger.debug("GLM message dump: %s", dumped)
        usage = getattr(resp, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
        request_id = getattr(getattr(resp, "_response", None), "headers", {}).get("x-request-id")
        return RunResult(provider="glm", model=model, text=text, usage=usage_dict, request_id=request_id)
    except APIStatusError as e:
        request_id = getattr(getattr(e, "response", None), "headers", {}).get("x-request-id")
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = None
        raise RuntimeError(
            f"GLM 请求失败：HTTP {e.status_code} - {detail or e.message}（request_id={request_id}）"
        ) from e
    except Exception as e:
        raise RuntimeError(f"GLM 请求失败：{type(e).__name__}: {e}") from e


def stream_glm(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    base_url: str,
    debug: bool = False,
) -> RunResult:
    from openai import APIStatusError, OpenAI

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        text_parts: list[str] = []
        usage_dict: dict[str, Any] | None = None
        request_id: str | None = None

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            # 尽量让兼容实现返回 usage（不保证每家都支持）
            stream_options={"include_usage": True},
        )
        dumped_any = False
        for chunk in stream:
            # 兼容：delta.content 可能是 str 或 list
            try:
                delta = chunk.choices[0].delta if chunk.choices else None
                piece = _coerce_message_text(getattr(delta, "content", None))
                if not piece:
                    piece = _coerce_message_text(getattr(delta, "reasoning_content", None))
                if not piece:
                    piece = _coerce_message_text(getattr(delta, "text", None))
            except Exception:
                piece = ""

            if piece:
                print(piece, end="", flush=True)
                text_parts.append(piece)
            elif debug and not dumped_any:
                dumped_any = True
                try:
                    logger.debug("GLM stream chunk dump: %s", chunk.model_dump())
                except Exception:
                    logger.debug("GLM stream chunk repr: %r", chunk)

            # usage 通常只在最后一个 chunk 出现
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage_dict = u.model_dump() if hasattr(u, "model_dump") else None

            # request_id：部分实现会挂在 _response.headers
            if request_id is None:
                request_id = getattr(getattr(chunk, "_response", None), "headers", {}).get("x-request-id")

        print()
        return RunResult(provider="glm", model=model, text="".join(text_parts), usage=usage_dict, request_id=request_id)
    except APIStatusError as e:
        request_id = getattr(getattr(e, "response", None), "headers", {}).get("x-request-id")
        detail = None
        try:
            detail = e.response.json()
        except Exception:
            detail = None
        raise RuntimeError(
            f"GLM 请求失败：HTTP {e.status_code} - {detail or e.message}（request_id={request_id}）"
        ) from e
    except Exception as e:
        raise RuntimeError(f"GLM 请求失败：{type(e).__name__}: {e}") from e


def _resolve_glm_key(cfg: dict[str, Any], key_source: str) -> str:
    """RAG 始终用 GLM（embedding-3 / glm-*）。这里独立解析 GLM key，避免依赖 chat 流程的 provider。"""
    env_key = os.getenv("GLM_API_KEY", "").strip()
    if env_key:
        return env_key
    cfg_key = str(_cfg_get(cfg, ["glm", "api_key"], "")).strip()
    if cfg_key:
        return cfg_key
    if key_source == "paste":
        key = getpass("RAG 需要 GLM API Key，请粘贴（输入不可见）：").strip()
        if not key:
            raise RuntimeError("你没有输入 GLM API Key。")
        return key
    raise RuntimeError("RAG 需要 GLM_API_KEY，但未在环境变量/配置文件中找到。")


@dataclass
class RAGComponents:
    """W3D1 重构：把原 query_engine 一体化拆成可独立操作的组件。"""
    nodes: list[Any]
    vec_retriever: Optional[Any]
    bm25_retriever: Optional[Any]
    response_synthesizer: Any


# W3D4 发现：BM25Retriever.from_defaults 的 tokenizer 参数已 deprecated，
# 而且实际是被静默吞掉的（只打 warning，不会传给底层 bm25s）。原 _make_bm25_tokenizer
# 函数（jieba / char 两个实现）从 W3D1 起就没生效过 —— BM25 一直在用 bm25s 默认的
# 英文 token_pattern (\b\w\w+\b)，几乎切不出任何中文。
#
# 治本方案是写一个 jieba 包装 retriever（corpus + query 两侧都用 jieba 预切再喂 bm25s），
# 复杂度较高，留给 W4。
#
# 当前临时方案：用中文友好的 token_pattern —— 英文/数字按 word 切，中文按字切（char-level
# 中文 BM25）。0 额外依赖、立即激活中文 BM25。配合 skip_stemming=True（中文不需 stemming）。
BM25_TOKEN_PATTERN_CN = r"(?u)\w+|[\u4e00-\u9fff]"


def reciprocal_rank_fusion(
    rank_lists: list[list[Any]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> list[tuple[Any, float, list[int]]]:
    """手写 RRF（Reciprocal Rank Fusion）。

    公式：score(node) = Σ_i  1 / (k + rank_i(node))
    其中 rank_i 是 node 在第 i 个排好序的检索结果中的位置（从 1 起）。
    没出现在某路里就不参与该路的累加。

    Args:
        rank_lists: 多路检索结果，每路是 [NodeWithScore, ...]，已按相关性降序。
        k: RRF 平滑常数，业界默认 60。越大越平滑（高 rank 优势越被抹平）。
        top_n: 只返回前 N 个；None = 全返回。

    Returns:
        [(node_with_score, fused_score, [rank_in_list_0, rank_in_list_1, ...]), ...]
        按 fused_score 降序。某路未命中的 rank 记为 0。
    """
    score_map: dict[str, float] = {}
    rank_map: dict[str, list[int]] = {}
    node_map: dict[str, Any] = {}
    n_lists = len(rank_lists)

    for li, lst in enumerate(rank_lists):
        for rank, ns in enumerate(lst, start=1):
            nid = ns.node.node_id
            score_map[nid] = score_map.get(nid, 0.0) + 1.0 / (k + rank)
            if nid not in rank_map:
                rank_map[nid] = [0] * n_lists
            rank_map[nid][li] = rank
            node_map[nid] = ns

    sorted_ids = sorted(score_map, key=lambda x: -score_map[x])
    if top_n is not None:
        sorted_ids = sorted_ids[:top_n]
    return [(node_map[nid], score_map[nid], rank_map[nid]) for nid in sorted_ids]


def _build_rag_components(
    *,
    api_key: str,
    data_dir: str,
    llm_model: str,
    embed_model: str,
    retriever_mode: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    bm25_tokenizer_name: str,
    streaming: bool,
    qa_template: Optional[str] = None,
    chroma_enabled: bool = False,
    chroma_persist_dir: Optional[str] = None,
    chroma_collection: Optional[str] = None,
) -> RAGComponents:
    """W3D1 重构核心：显式切 nodes，按 retriever_mode 构建对应的检索器与合成器。"""
    try:
        from llama_index.core import (
            PromptTemplate,
            Settings,
            SimpleDirectoryReader,
            VectorStoreIndex,
        )
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.response_synthesizers import get_response_synthesizer
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
        from llama_index.llms.zhipuai import ZhipuAI
    except ImportError as e:
        raise RuntimeError(
            "缺少 RAG 依赖。请先安装：\n"
            "  .venv/bin/pip install llama-index "
            "llama-index-llms-zhipuai llama-index-embeddings-zhipuai\n"
            f"详情：{e}"
        ) from e

    if retriever_mode not in ("vector", "bm25", "hybrid"):
        raise RuntimeError(f"未知 retriever: {retriever_mode}（仅支持 vector / bm25 / hybrid）")

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"未找到数据目录：{data_dir}（用 --data-dir 指定，或在配置文件 [rag] 里设置）")

    Settings.llm = ZhipuAI(model=llm_model, api_key=api_key)
    Settings.embed_model = ZhipuAIEmbedding(model=embed_model, api_key=api_key)

    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        raise RuntimeError(f"目录 {data_dir} 内没有可加载的文档")
    logger.info("已加载 %d 个文档（来自 %s）", len(documents), data_dir)

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    if not nodes:
        raise RuntimeError("切片后未产生任何 node，检查 chunk_size / 文档内容")
    logger.info(
        "切出 %d 个 node（chunk_size=%d, overlap=%d）",
        len(nodes), chunk_size, chunk_overlap,
    )

    vec_retriever = None
    bm25_retriever = None

    if retriever_mode in ("vector", "hybrid"):
        # W3D5 C1：若开启 Chroma 持久化，把向量存储从内存 SimpleVectorStore 换成
        # ChromaVectorStore（PersistentClient 写盘）。本阶段只做 ingest + 落盘；
        # 二次启动仍会重建（重算 embedding 后写入同一 collection）。
        # 二次启动跳过 ingest、直接 load 索引复用，留给 W3D6 (5.23) C2 阶段。
        storage_context = None
        if chroma_enabled:
            try:
                import chromadb
                from llama_index.core import StorageContext
                from llama_index.vector_stores.chroma import ChromaVectorStore
            except ImportError as e:
                raise RuntimeError(
                    "缺少 Chroma 依赖。请先安装：\n"
                    "  .venv/bin/pip install chromadb llama-index-vector-stores-chroma\n"
                    f"详情：{e}"
                ) from e

            persist_dir = chroma_persist_dir or "./chroma_db"
            coll_name = chroma_collection or "rag_default"
            os.makedirs(persist_dir, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection_obj = chroma_client.get_or_create_collection(coll_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection_obj)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logger.info(
                "已启用 Chroma 持久化向量库（persist_dir=%s, collection=%s, mode=ingest+persist）",
                persist_dir, coll_name,
            )

        if storage_context is not None:
            vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            vector_index = VectorStoreIndex(nodes)
        vec_retriever = vector_index.as_retriever(similarity_top_k=top_k)
        logger.info("已构建向量 retriever（top_k=%d）", top_k)

    if retriever_mode in ("bm25", "hybrid"):
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError as e:
            raise RuntimeError(
                "缺少 BM25 依赖。请先安装：\n"
                "  .venv/bin/pip install llama-index-retrievers-bm25\n"
                f"详情：{e}"
            ) from e

        # W3D4：从 deprecated 的 tokenizer=callable 改成 token_pattern + skip_stemming，
        # 用中文友好正则激活中文 BM25。bm25_tokenizer_name 参数现在只用于日志/留作 W4 jieba
        # 包装的开关位，不再实际控制分词逻辑（详见 BM25_TOKEN_PATTERN_CN 注释）。
        if bm25_tokenizer_name and bm25_tokenizer_name.lower() == "jieba":
            logger.warning(
                "bm25_tokenizer=jieba 当前未实现真正的词级 BM25（W4 待办），"
                "正在用中文 char-level token_pattern 兜底。"
            )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
            token_pattern=BM25_TOKEN_PATTERN_CN,
            skip_stemming=True,
            language="en",
        )
        logger.info(
            "已构建 BM25 retriever（top_k=%d, token_pattern=cn_char_word, skip_stemming=True）",
            top_k,
        )

    synth_kwargs: dict[str, Any] = {"streaming": streaming}
    if qa_template:
        tmpl = qa_template.strip()
        missing = [p for p in ("{context_str}", "{query_str}") if p not in tmpl]
        if missing:
            raise RuntimeError(
                f"RAG qa_template 缺少必须占位符：{missing}（必须同时包含 {{context_str}} 和 {{query_str}}）"
            )
        synth_kwargs["text_qa_template"] = PromptTemplate(tmpl)
        logger.info("已应用自定义 RAG qa_template（%d chars）", len(tmpl))

    response_synthesizer = get_response_synthesizer(**synth_kwargs)

    return RAGComponents(
        nodes=nodes,
        vec_retriever=vec_retriever,
        bm25_retriever=bm25_retriever,
        response_synthesizer=response_synthesizer,
    )


def _snippet(node_with_score: Any, max_len: int = 120) -> str:
    try:
        text = node_with_score.node.get_content().strip().replace("\n", " ")
    except Exception:
        text = str(node_with_score)
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _log_retrieval(
    mode: str,
    *,
    vec_list: Optional[list[Any]] = None,
    bm25_list: Optional[list[Any]] = None,
    fused_tuples: Optional[list[tuple[Any, float, list[int]]]] = None,
) -> None:
    """打印检索阶段命中。

    - vector / bm25 单路：直接列 [rank] score text
    - hybrid：列融合后的 top_n，附带 [vec=#? bm25=#? fused=#? score=...]，方便看每个候选在两路里的位置
    """
    if mode == "vector" and vec_list:
        logger.info("retriever=vector 命中 %d 个 node", len(vec_list))
        for i, ns in enumerate(vec_list, 1):
            score = getattr(ns, "score", None)
            score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
            logger.info("  [%d] vec_score=%s text=%s", i, score_s, _snippet(ns))
        return

    if mode == "bm25" and bm25_list:
        logger.info("retriever=bm25 命中 %d 个 node", len(bm25_list))
        for i, ns in enumerate(bm25_list, 1):
            score = getattr(ns, "score", None)
            score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
            logger.info("  [%d] bm25_score=%s text=%s", i, score_s, _snippet(ns))
        return

    if mode == "hybrid":
        vec_list = vec_list or []
        bm25_list = bm25_list or []
        fused_tuples = fused_tuples or []

        # 先把单路 ranking 也打出来，方便人工对照融合前后的差异
        logger.info("retriever=hybrid 单路向量召回 %d 个", len(vec_list))
        for i, ns in enumerate(vec_list, 1):
            score = getattr(ns, "score", None)
            score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
            logger.info("  vec[%d] score=%s text=%s", i, score_s, _snippet(ns))

        logger.info("retriever=hybrid 单路 BM25 召回 %d 个", len(bm25_list))
        for i, ns in enumerate(bm25_list, 1):
            score = getattr(ns, "score", None)
            score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
            logger.info("  bm25[%d] score=%s text=%s", i, score_s, _snippet(ns))

        logger.info("retriever=hybrid 融合后取 %d 个（RRF）", len(fused_tuples))
        for i, (ns, fused, ranks) in enumerate(fused_tuples, 1):
            vec_rank = ranks[0] if len(ranks) >= 1 else 0
            bm25_rank = ranks[1] if len(ranks) >= 2 else 0
            vec_s = f"#{vec_rank}" if vec_rank else "—"
            bm25_s = f"#{bm25_rank}" if bm25_rank else "—"
            logger.info(
                "  fused[%d] vec=%s bm25=%s rrf_score=%.4f text=%s",
                i, vec_s, bm25_s, fused, _snippet(ns),
            )
        return


def answer_with_rag(
    api_key: str,
    question: str,
    *,
    data_dir: str,
    llm_model: str,
    embed_model: str,
    retriever_mode: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    bm25_tokenizer_name: str,
    rrf_k: int,
    streaming: bool,
    qa_template: Optional[str] = None,
    chroma_enabled: bool = False,
    chroma_persist_dir: Optional[str] = None,
    chroma_collection: Optional[str] = None,
) -> RunResult:
    """W3D1 重构：vector / bm25 / hybrid 三模式共用同一份 nodes + LLM。

    流程：
      1. _build_rag_components 拿 nodes / retriever(s) / synthesizer
      2. 按 mode 跑 retrieve（hybrid 跑两路 + 手写 RRF 融合）
      3. synthesizer.synthesize 用融合后的 nodes 出回答
      4. 打日志（hybrid 模式打三路 rank）
    """
    try:
        from llama_index.core.schema import NodeWithScore

        components = _build_rag_components(
            api_key=api_key,
            data_dir=data_dir,
            llm_model=llm_model,
            embed_model=embed_model,
            retriever_mode=retriever_mode,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            bm25_tokenizer_name=bm25_tokenizer_name,
            streaming=streaming,
            qa_template=qa_template,
            chroma_enabled=chroma_enabled,
            chroma_persist_dir=chroma_persist_dir,
            chroma_collection=chroma_collection,
        )

        # ---- Retrieve ----
        vec_list: Optional[list[Any]] = None
        bm25_list: Optional[list[Any]] = None
        fused_tuples: Optional[list[tuple[Any, float, list[int]]]] = None

        if retriever_mode == "vector":
            assert components.vec_retriever is not None
            vec_list = components.vec_retriever.retrieve(question)
            nodes_for_synth = vec_list
        elif retriever_mode == "bm25":
            assert components.bm25_retriever is not None
            bm25_list = components.bm25_retriever.retrieve(question)
            nodes_for_synth = bm25_list
        else:  # hybrid
            assert components.vec_retriever is not None
            assert components.bm25_retriever is not None
            vec_list = components.vec_retriever.retrieve(question)
            bm25_list = components.bm25_retriever.retrieve(question)
            fused_tuples = reciprocal_rank_fusion(
                [vec_list, bm25_list], k=rrf_k, top_n=top_k,
            )
            # 用 fused_score 重新包装 NodeWithScore，这样后续日志/下游能看到 RRF 分数
            nodes_for_synth = [
                NodeWithScore(node=ns.node, score=fused)
                for ns, fused, _ in fused_tuples
            ]

        # ---- Synthesize ----
        response = components.response_synthesizer.synthesize(question, nodes=nodes_for_synth)

        text_parts: list[str] = []
        if streaming:
            gen = getattr(response, "response_gen", None)
            if gen is None:
                full = str(response)
                print(full, end="", flush=True)
                text_parts.append(full)
            else:
                for token in gen:
                    if not token:
                        continue
                    print(token, end="", flush=True)
                    text_parts.append(token)
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            full = str(response)
            print(full)
            sys.stdout.flush()
            text_parts.append(full)

        _log_retrieval(
            retriever_mode,
            vec_list=vec_list,
            bm25_list=bm25_list,
            fused_tuples=fused_tuples,
        )
        return RunResult(
            provider=f"rag-{retriever_mode}(glm)",
            model=llm_model,
            text="".join(text_parts),
        )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"RAG 请求失败：{type(e).__name__}: {e}") from e


def main(argv: list[str]) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.getenv("HELLO_WORLD_CONFIG", "hello-world.toml"))
    parser.add_argument("--provider", choices=["claude", "glm"], default=None)
    parser.add_argument("--key-source", choices=["env", "paste"], default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--question", "-q", default=None, help="提问内容（等价于 --prompt，更直观）。")
    parser.add_argument("--rag", action="store_true", help="启用 RAG（基于本地文档回答）。")
    parser.add_argument("--no-rag", action="store_true", help="禁用 RAG（仅普通对话）。")
    parser.add_argument("--data-dir", default=None, help="RAG 文档目录（默认读 [rag].data_dir）。")
    parser.add_argument("--rag-top-k", type=int, default=None)
    parser.add_argument("--rag-llm-model", default=None)
    parser.add_argument("--rag-embed-model", default=None)
    parser.add_argument(
        "--retriever",
        choices=["vector", "bm25", "hybrid"],
        default=None,
        help="RAG 检索器：vector(纯向量) / bm25(纯关键词) / hybrid(RRF 融合)。覆盖配置文件 [rag].retriever。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="文档切片大小（覆盖 [rag].chunk_size）。",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="文档切片重叠（覆盖 [rag].chunk_overlap）。",
    )
    parser.add_argument(
        "--bm25-tokenizer",
        choices=["jieba", "char"],
        default=None,
        help="BM25 分词器（覆盖 [rag].bm25_tokenizer）。",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="RRF 平滑常数 k（hybrid 模式生效，覆盖 [rag].rrf_k，默认 60）。",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="启用 Chroma 持久化向量库（W3D5 C1，覆盖 [rag.chroma].enabled）。"
        "C1 阶段：embedding 写入 persist_dir 落盘；二次启动仍重建（C2 在 W3D6 实现）。",
    )
    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="禁用 Chroma 持久化（覆盖 [rag.chroma].enabled）。",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        default=None,
        help="Chroma 持久化目录（覆盖 [rag.chroma].persist_dir，默认 ./chroma_db）。",
    )
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Chroma collection 名（覆盖 [rag.chroma].collection_name，默认 rag_default）。",
    )
    parser.add_argument(
        "--rag-qa-template-file",
        default=None,
        help="从文件读取 RAG 回答模板，最高优先级。",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        help=(
            "题型路由：按 [rag.templates] 表把题型名映射到 prompts/xxx.txt。"
            "默认题型走 [rag].default_question_type。"
            "可选值由配置文件决定，未知值会列出所有可选项。"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--claude-model", default=os.getenv("CLAUDE_MODEL", "claude-haiku-4.5"))
    parser.add_argument("--glm-model", default=os.getenv("GLM_MODEL", "glm-5.1"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream", action="store_true", help="启用流式输出（覆盖配置文件）。")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出（覆盖配置文件）。")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="控制日志输出级别（默认 ERROR，只显示错误）。",
    )
    args = parser.parse_args(argv)

    cfg = _read_config(args.config)

    # 优先级：命令行 > 环境变量 > 配置文件 > 代码默认值
    if args.log_level is not None:
        log_level = args.log_level
    elif os.getenv("LOG_LEVEL"):
        log_level = os.environ["LOG_LEVEL"]
    else:
        log_level = _cfg_get(cfg, ["app", "log_level"], "ERROR")
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.ERROR),
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    provider = args.provider or _cfg_get(cfg, ["app", "provider"]) or _prompt_choice(
        "请选择要调用的模型提供方：",
        [("claude", "Claude（Anthropic）"), ("glm", "GLM（智谱 OpenAI 兼容）")],
        default="claude",
    )
    key_source = args.key_source or _cfg_get(cfg, ["app", "key_source"]) or _prompt_choice(
        "请选择 API Key 来源：",
        [("env", "从环境变量读取"), ("paste", "运行时手动粘贴")],
        default="env",
    )

    # 是否启用 RAG：命令行 > 配置文件 > 交互式询问
    if args.no_rag:
        rag_enabled = False
    elif args.rag:
        rag_enabled = True
    else:
        cfg_rag = _cfg_get(cfg, ["rag", "enabled"], None)
        if cfg_rag is None:
            rag_enabled = (
                _prompt_choice(
                    "是否使用本地文档（RAG）回答？",
                    [("no", "否（普通对话）"), ("yes", "是（基于本地文档）")],
                    default="no",
                )
                == "yes"
            )
        else:
            rag_enabled = bool(cfg_rag)

    # 问题：--question > --prompt > 配置文件 prompt > 交互输入 > 代码默认
    cfg_prompt = _cfg_get(cfg, ["app", "prompt"], None)
    if args.question is not None:
        question = args.question
    elif args.prompt is not None:
        question = args.prompt
    else:
        default_q = cfg_prompt or DEFAULT_PROMPT
        typed = input(f"请输入你的问题（直接回车使用默认）：\n  默认: {default_q}\n> ").strip()
        question = typed or default_q

    max_tokens = args.max_tokens if args.max_tokens is not None else int(_cfg_get(cfg, ["app", "max_tokens"], 300))
    cfg_stream = bool(_cfg_get(cfg, ["app", "stream"], False))
    stream_enabled = False if args.no_stream else (True if args.stream else cfg_stream)

    # ---- RAG 分支：独立解析 GLM key，不依赖 chat 的 provider/key ----
    if rag_enabled:
        try:
            glm_key = _resolve_glm_key(cfg, key_source)
        except Exception as e:
            print(f"[错误] 获取 GLM API Key 失败：{e}", file=sys.stderr)
            return 2

        data_dir = args.data_dir or _cfg_get(cfg, ["rag", "data_dir"], "data")
        rag_top_k = args.rag_top_k if args.rag_top_k is not None else int(_cfg_get(cfg, ["rag", "top_k"], 3))
        rag_llm_model = args.rag_llm_model or _cfg_get(cfg, ["rag", "llm_model"], "glm-4-plus")
        rag_embed_model = args.rag_embed_model or _cfg_get(cfg, ["rag", "embed_model"], "embedding-3")
        retriever_mode = args.retriever or _cfg_get(cfg, ["rag", "retriever"], "vector")
        chunk_size = args.chunk_size if args.chunk_size is not None else int(_cfg_get(cfg, ["rag", "chunk_size"], 512))
        chunk_overlap = (
            args.chunk_overlap
            if args.chunk_overlap is not None
            else int(_cfg_get(cfg, ["rag", "chunk_overlap"], 50))
        )
        bm25_tokenizer_name = args.bm25_tokenizer or _cfg_get(cfg, ["rag", "bm25_tokenizer"], "jieba")
        rrf_k = args.rrf_k if args.rrf_k is not None else int(_cfg_get(cfg, ["rag", "rrf_k"], 60))

        # Chroma 持久化（W3D5 C1）
        if args.no_chroma:
            chroma_enabled = False
        elif args.chroma:
            chroma_enabled = True
        else:
            chroma_enabled = bool(_cfg_get(cfg, ["rag", "chroma", "enabled"], False))
        chroma_persist_dir = args.chroma_persist_dir or _cfg_get(
            cfg, ["rag", "chroma", "persist_dir"], "./chroma_db"
        )
        chroma_collection = args.chroma_collection or _cfg_get(
            cfg, ["rag", "chroma", "collection_name"], "rag_default"
        )

        # 回答模板优先级（W3D4 加入题型路由）：
        #   1) --rag-qa-template-file <path>        显式文件，最高优先级
        #   2) --question-type <type> / default_question_type
        #                                           语义路由，查 [rag.templates] 表
        #   3) [rag].qa_template                    老字段，向后兼容
        #   4) (都没有)                              LlamaIndex 默认英文 prompt
        qa_template: Optional[str] = None
        template_source: str = "llama_index_default"
        templates_map = _cfg_get(cfg, ["rag", "templates"], {}) or {}
        default_qtype = str(_cfg_get(cfg, ["rag", "default_question_type"], "") or "").strip()
        effective_qtype = (args.question_type or default_qtype).strip()

        def _load_template_file(path: str, src_label: str) -> Optional[str]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError as exc:
                print(
                    f"[错误] 读取 RAG 模板文件失败 (source={src_label}): {exc}",
                    file=sys.stderr,
                )
                return None

        if args.rag_qa_template_file:
            qa_template = _load_template_file(args.rag_qa_template_file, "file")
            if qa_template is None:
                return 2
            template_source = f"file({args.rag_qa_template_file})"
        elif effective_qtype:
            if effective_qtype not in templates_map:
                available = ", ".join(sorted(templates_map.keys())) or "(空，请在 [rag.templates] 配置)"
                print(
                    f"[错误] 未知 question_type: {effective_qtype!r}（可选：{available}）",
                    file=sys.stderr,
                )
                return 2
            mapped_path = str(templates_map[effective_qtype])
            qa_template = _load_template_file(mapped_path, f"question_type={effective_qtype}")
            if qa_template is None:
                return 2
            template_source = f"question_type={effective_qtype} -> {mapped_path}"
        else:
            cfg_tmpl = _cfg_get(cfg, ["rag", "qa_template"], None)
            if isinstance(cfg_tmpl, str) and cfg_tmpl.strip():
                qa_template = cfg_tmpl
                template_source = "toml.qa_template"

        chroma_log = (
            f"on(persist_dir={chroma_persist_dir}, collection={chroma_collection})"
            if chroma_enabled
            else "off"
        )
        logger.info(
            "rag=on retriever=%s data_dir=%s top_k=%s chunk=%s/%s bm25_tok=%s rrf_k=%s "
            "llm=%s embed=%s key=%s template=%s chroma=%s",
            retriever_mode, data_dir, rag_top_k, chunk_size, chunk_overlap,
            bm25_tokenizer_name, rrf_k,
            rag_llm_model, rag_embed_model, _mask_key(glm_key),
            template_source, chroma_log,
        )
        try:
            answer_with_rag(
                api_key=glm_key,
                question=question,
                data_dir=str(data_dir),
                llm_model=str(rag_llm_model),
                embed_model=str(rag_embed_model),
                retriever_mode=str(retriever_mode),
                top_k=int(rag_top_k),
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                bm25_tokenizer_name=str(bm25_tokenizer_name),
                rrf_k=int(rrf_k),
                streaming=stream_enabled,
                qa_template=qa_template,
                chroma_enabled=chroma_enabled,
                chroma_persist_dir=str(chroma_persist_dir),
                chroma_collection=str(chroma_collection),
            )
        except Exception as e:
            logger.error("%s", e)
            return 1
        return 0

    # ---- 普通 chat 分支 ----
    try:
        # 优先尊重 --key-source（命令行优先原则）；仅在选定来源拿不到 key 时，
        # 才回落到配置文件里写死的 api_key（不推荐明文保存）。
        try:
            api_key = _get_api_key(provider, key_source)
        except RuntimeError:
            cfg_key = ""
            if provider in ("claude", "glm"):
                cfg_key = str(_cfg_get(cfg, [provider, "api_key"], "")).strip()
            if not cfg_key:
                raise
            logger.info("使用配置文件中的 %s.api_key 作为兜底", provider)
            api_key = cfg_key
    except Exception as e:
        print(f"[错误] 获取 API Key 失败：{e}", file=sys.stderr)
        return 2

    if provider == "claude":
        model = args.claude_model if "CLAUDE_MODEL" in os.environ else _cfg_get(cfg, ["claude", "model"], args.claude_model)
    else:
        model = args.glm_model if "GLM_MODEL" in os.environ else _cfg_get(cfg, ["glm", "model"], args.glm_model)

    glm_base_url = os.getenv("GLM_BASE_URL") or _cfg_get(cfg, ["glm", "base_url"], "https://open.bigmodel.cn/api/paas/v4/")

    logger.info("provider=%s model=%s key=%s", provider, model, _mask_key(api_key))

    try:
        if provider == "claude":
            if stream_enabled:
                result = stream_claude(api_key=api_key, model=model, prompt=question, max_tokens=max_tokens)
            else:
                result = call_claude(api_key=api_key, model=model, prompt=question, max_tokens=max_tokens)
        else:
            if stream_enabled:
                result = stream_glm(
                    api_key=api_key,
                    model=model,
                    prompt=question,
                    max_tokens=max_tokens,
                    base_url=str(glm_base_url),
                    debug=args.debug,
                )
            else:
                result = call_glm(
                    api_key=api_key,
                    model=model,
                    prompt=question,
                    max_tokens=max_tokens,
                    base_url=str(glm_base_url),
                    debug=args.debug,
                )
    except Exception as e:
        logger.error("%s", e)
        return 1

    # 流式模式下，正文已在 stream_* 中实时打印；这里避免重复输出。
    if not stream_enabled:
        if result.text:
            print(result.text)
        else:
            logger.error("本次响应未提取到可显示的文本内容。可加 --debug/--log-level DEBUG 查看返回结构。")
            return 1

    if result.usage:
        logger.debug("usage: %s", result.usage)
    if result.request_id:
        logger.info("request_id: %s", result.request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
