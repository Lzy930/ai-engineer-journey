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


def _build_rag_engine(
    api_key: str,
    data_dir: str,
    llm_model: str,
    embed_model: str,
    top_k: int,
    streaming: bool,
    qa_template: str | None = None,
):
    try:
        from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
        from llama_index.llms.zhipuai import ZhipuAI
    except ImportError as e:
        raise RuntimeError(
            "缺少 RAG 依赖。请先安装：\n"
            "  .venv/bin/pip install llama-index "
            "llama-index-llms-zhipuai llama-index-embeddings-zhipuai\n"
            f"详情：{e}"
        ) from e

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"未找到数据目录：{data_dir}（用 --data-dir 指定，或在配置文件 [rag] 里设置）")

    Settings.llm = ZhipuAI(model=llm_model, api_key=api_key)
    Settings.embed_model = ZhipuAIEmbedding(model=embed_model, api_key=api_key)

    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        raise RuntimeError(f"目录 {data_dir} 内没有可加载的文档")
    logger.info("已加载 %d 个文档（来自 %s）", len(documents), data_dir)

    index = VectorStoreIndex.from_documents(documents)

    engine_kwargs: dict[str, Any] = {"streaming": streaming, "similarity_top_k": top_k}
    if qa_template:
        tmpl = qa_template.strip()
        # LlamaIndex 模板必须包含 {context_str} 和 {query_str} 两个占位符
        missing = [p for p in ("{context_str}", "{query_str}") if p not in tmpl]
        if missing:
            raise RuntimeError(
                f"RAG qa_template 缺少必须占位符：{missing}（必须同时包含 {{context_str}} 和 {{query_str}}）"
            )
        engine_kwargs["text_qa_template"] = PromptTemplate(tmpl)
        logger.info("已应用自定义 RAG qa_template（%d chars）", len(tmpl))

    return index.as_query_engine(**engine_kwargs)


def _log_rag_sources(response: Any) -> None:
    sources = getattr(response, "source_nodes", []) or []
    if not sources:
        return
    logger.info("命中片段数: %d", len(sources))
    for i, sn in enumerate(sources, 1):
        score = getattr(sn, "score", None)
        try:
            text = sn.node.get_content().strip().replace("\n", " ")
        except Exception:
            text = str(sn)
        snippet = text[:120] + ("..." if len(text) > 120 else "")
        score_s = f"{score:.3f}" if isinstance(score, float) else str(score)
        logger.info("  [%d] score=%s text=%s", i, score_s, snippet)


def answer_with_rag(
    api_key: str,
    question: str,
    *,
    data_dir: str,
    llm_model: str,
    embed_model: str,
    top_k: int,
    streaming: bool,
    qa_template: str | None = None,
) -> RunResult:
    try:
        engine = _build_rag_engine(
            api_key=api_key,
            data_dir=data_dir,
            llm_model=llm_model,
            embed_model=embed_model,
            top_k=top_k,
            streaming=streaming,
            qa_template=qa_template,
        )
        response = engine.query(question)

        text_parts: list[str] = []
        if streaming:
            gen = getattr(response, "response_gen", None)
            if gen is None:
                # 部分版本不暴露 response_gen，退回 str()
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

        _log_rag_sources(response)
        return RunResult(provider="rag(glm)", model=llm_model, text="".join(text_parts))
    except Exception as e:
        # 已经是 RuntimeError（依赖/数据目录）就直接抛；其它包成更友好的提示
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
        "--rag-qa-template-file",
        default=None,
        help="从文件读取 RAG 回答模板，覆盖配置文件 [rag].qa_template。",
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

        # 回答模板：命令行文件 > 配置文件 > 不设（用 LlamaIndex 默认）
        qa_template: Optional[str] = None
        if args.rag_qa_template_file:
            try:
                with open(args.rag_qa_template_file, "r", encoding="utf-8") as f:
                    qa_template = f.read()
            except OSError as e:
                print(f"[错误] 读取 RAG 模板文件失败：{e}", file=sys.stderr)
                return 2
        else:
            cfg_tmpl = _cfg_get(cfg, ["rag", "qa_template"], None)
            if isinstance(cfg_tmpl, str) and cfg_tmpl.strip():
                qa_template = cfg_tmpl

        logger.info(
            "rag=on data_dir=%s top_k=%s llm=%s embed=%s key=%s qa_template=%s",
            data_dir, rag_top_k, rag_llm_model, rag_embed_model, _mask_key(glm_key),
            "custom" if qa_template else "default",
        )
        try:
            answer_with_rag(
                api_key=glm_key,
                question=question,
                data_dir=str(data_dir),
                llm_model=str(rag_llm_model),
                embed_model=str(rag_embed_model),
                top_k=int(rag_top_k),
                streaming=stream_enabled,
                qa_template=qa_template,
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
