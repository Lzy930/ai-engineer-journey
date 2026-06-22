#!/usr/bin/env bash
# W3D4 回归：用 --question-type 路由跑 4 题 × 2 题型，再加 2 个边界测试。

set -e

QUESTIONS=(
  "VFD-F17 是什么报警？"
  "E-7429 是什么事故？发生在什么时候？"
  "哪些供应商可能影响下月交付？"
  "RAG 模型为什么会胡编内容？"
)
TYPES=("factual" "subjective")

run_one() {
  local q="$1"
  local t="$2"
  # log-level INFO 用于显示 template=... 路由日志（截到 stderr）
  .venv/bin/python hello-world.py \
    --rag --retriever hybrid --rag-top-k 5 \
    --no-stream --max-tokens 400 --log-level INFO \
    --question-type "$t" \
    --question "$q" 2>&1 1>/tmp/_qa_stdout.$$ \
    | grep "template=" | head -n1
  cat /tmp/_qa_stdout.$$
  rm -f /tmp/_qa_stdout.$$
}

i=0
for q in "${QUESTIONS[@]}"; do
  i=$((i+1))
  echo "========================================"
  echo "Test $i: $q"
  echo "========================================"
  for t in "${TYPES[@]}"; do
    echo ""
    echo "--- [--question-type $t] ---"
    run_one "$q" "$t"
  done
  echo ""
done

echo ""
echo "========================================"
echo "边界 1: 未知 question-type 应报错"
echo "========================================"
.venv/bin/python hello-world.py --rag --question-type causal \
  --question "test" --no-stream --log-level ERROR 2>&1 | head -n3
echo "(exit code: $?)"

echo ""
echo "========================================"
echo "边界 2: 不传 --question-type 走老 qa_template 字段（向后兼容）"
echo "========================================"
.venv/bin/python hello-world.py --rag --retriever hybrid --rag-top-k 5 \
  --no-stream --max-tokens 200 --log-level INFO \
  --question "VFD-F17 是什么报警？" 2>&1 | grep "template="
