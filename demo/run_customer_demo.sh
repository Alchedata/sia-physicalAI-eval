#!/usr/bin/env bash
set -euo pipefail

# run_customer_demo.sh
# Guided customer demo runner for SEPA-Eval synthetic data.
#
# Usage:
#   bash demo/run_customer_demo.sh
#   bash demo/run_customer_demo.sh --no-pause
#   bash demo/run_customer_demo.sh --regenerate
#   bash demo/run_customer_demo.sh --memory-dir demo/demo_eval_memory
#
# Defaults:
#   - Runs from repo root or any subdirectory inside this repo.
#   - Uses demo/demo_eval_memory as EvalMemory directory.

NO_PAUSE=0
REGENERATE=0
MEMORY_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pause)
      NO_PAUSE=1
      shift
      ;;
    --regenerate)
      REGENERATE=1
      shift
      ;;
    --memory-dir)
      MEMORY_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

pause() {
  if [[ "$NO_PAUSE" -eq 0 ]]; then
    read -r -p "Press Enter to continue..." _
  fi
}

# Resolve repo root by walking up until AlphaBrain directory is found.
START_DIR="$PWD"
ROOT_DIR=""
CUR_DIR="$PWD"
for _ in {1..8}; do
  if [[ -d "$CUR_DIR/AlphaBrain" && -d "$CUR_DIR/demo" ]]; then
    ROOT_DIR="$CUR_DIR"
    break
  fi
  CUR_DIR="$(dirname "$CUR_DIR")"
done

if [[ -z "$ROOT_DIR" ]]; then
  echo "Could not find repo root containing AlphaBrain/ and demo/."
  echo "Start this script from somewhere inside the project tree."
  exit 1
fi

if [[ -z "$MEMORY_DIR" ]]; then
  MEMORY_DIR="$ROOT_DIR/demo/demo_eval_memory"
elif [[ "$MEMORY_DIR" != /* ]]; then
  MEMORY_DIR="$ROOT_DIR/$MEMORY_DIR"
fi

ALPHABRAIN_DIR="$ROOT_DIR/AlphaBrain"
REPORT_OUT="$ROOT_DIR/demo/output/report.md"
HTML_OUT="$ROOT_DIR/demo/output/report.html"

mkdir -p "$(dirname "$REPORT_OUT")"

echo
 echo "============================================================"
echo "SEPA-Eval Customer Demo"
echo "============================================================"
echo "Repo root : $ROOT_DIR"
echo "AlphaBrain: $ALPHABRAIN_DIR"
echo "Memory dir: $MEMORY_DIR"
echo "Output    : $REPORT_OUT"
echo "Started in: $START_DIR"
echo "============================================================"
echo

# Optional regeneration for deterministic fresh run.
if [[ "$REGENERATE" -eq 1 ]]; then
  echo "[Setup] Regenerating synthetic traces for a fresh demo snapshot..."
  rm -rf "$MEMORY_DIR"
  (
    cd "$ALPHABRAIN_DIR"
    python ../demo/generate_libero_traces.py --output-dir "$MEMORY_DIR"
  )
  echo
fi

if [[ ! -d "$MEMORY_DIR" || ! -f "$MEMORY_DIR/eval.db" ]]; then
  echo "[Setup] Demo memory not found at: $MEMORY_DIR"
  echo "Generating synthetic traces now..."
  (
    cd "$ALPHABRAIN_DIR"
    python ../demo/generate_synthetic_traces.py --output-dir "$MEMORY_DIR"
  )
  echo
fi

run_cmd() {
  local label="$1"
  shift
  echo "[$label] $*"
  "$@"
  echo
}

cd "$ALPHABRAIN_DIR"

echo "Act 1: The Problem"
echo "Narration: 'Traditional benchmark SR can look strong while hiding failure modes.'"
pause

run_cmd "Status" \
  python -m sepa_eval --memory-dir "$MEMORY_DIR" status

echo "Act 2: Capability Frontier + Saturation"
echo "Narration: 'Spatial tasks are near-saturated; goal tasks still separate models.'"
pause

run_cmd "Report" \
  python -m sepa_eval --memory-dir "$MEMORY_DIR" report --output "$REPORT_OUT"

if [[ -f "$ROOT_DIR/demo/render_report_html.py" ]]; then
  run_cmd "HTML Dashboard" \
    python "$ROOT_DIR/demo/render_report_html.py" --input "$REPORT_OUT" --output "$HTML_OUT"
fi

echo "Key section preview: Saturation Map"
awk '/## 2. Saturation Map/{flag=1} /## 3. Failure Taxonomy/{flag=0} flag' "$REPORT_OUT" | head -40

echo
pause

echo "Act 3: Model Separation"
echo "Narration: 'On evolved / harder tasks, model gaps become obvious.'"
pause

run_cmd "Diff" \
  python -m sepa_eval --memory-dir "$MEMORY_DIR" diff QwenOFT-v2.1 NeuroVLA-v1.2

echo "Act 4: Failure Diagnosis"
echo "Narration: 'We do not just score failure; we categorize recurring failure patterns.'"
pause

echo "Key section preview: Failure Taxonomy"
awk '/## 3. Failure Taxonomy/{flag=1} /## 4. Evolved Task Summary/{flag=0} flag' "$REPORT_OUT"

echo
pause

echo "Act 5: Self-Evolution Evidence"
echo "Narration: 'SEPA-Eval promotes harder variants and queues borderline ones for review.'"
pause

run_cmd "Review Queue" \
  python -m sepa_eval --memory-dir "$MEMORY_DIR" review list

echo "Act 6: Closing the Loop"
echo "Narration: 'Failed rollouts can be exported into continual learning datasets.'"
pause

echo "[Export Command] python -m sepa_eval --memory-dir \"$MEMORY_DIR\" export-hard-cases --help"
python -m sepa_eval export-hard-cases --help | head -25

echo
 echo "============================================================"
echo "Demo complete"
echo "Report: $REPORT_OUT"
echo "Dashboard: $HTML_OUT"
echo "You can now open the report and use this terminal log as your talk track."
echo "============================================================"
