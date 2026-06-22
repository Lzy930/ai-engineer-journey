#!/usr/bin/env bash
# 跑 4 题 × 2 模板 = 8 次对比测试。
# 输出到 stdout，每段一个标题清楚分隔。

set -e

QUESTIONS=(
  "VFD-F17 是什么报警？"
  "E-7429 是什么事故？发生在什么时候？"
  "哪些供应商可能影响下月交付？"
  "RAG 模型为什么会胡编内容？"
)
TEMPLATES=(
  "prompts/qa_factual.txt"
  "prompts/qa_subjective.txt"
)
LABELS=(
  "factual(宽松)"
  "subjective(严格)"
)

run_one() {
  local q="$1"
  local t="$2"
  .venv/bin/python hello-world.py \
    --rag --retriever hybrid --rag-top-k 5 \
    --no-stream --max-tokens 400 --log-level ERROR \
    --rag-qa-template-file "$t" \
    --question "$q" 2>/dev/null
}

i=0
for q in "${QUESTIONS[@]}"; do
  i=$((i+1))
  echo "========================================"
  echo "Test $i: $q"
  echo "========================================"
  for k in 0 1; do
    echo ""
    echo "--- [${LABELS[$k]}] ---"
    run_one "$q" "${TEMPLATES[$k]}"
  done
  echo ""
done
