#!/usr/bin/env bash
# check-consistency.sh — guard against drift between articles.md and downstream caches.
#
# Seven checks (per README.md 开发须知):
#   C1 — references/articles.md numbering is contiguous 1..N
#   C2 — N matches downstream count claims (README, deep-research-tracker.md, references/AGENTS.md)
#        Files with standalone "<!-- check-consistency: skip-count -->" are exempted
#   C3 — thinking/, feedback/ *.md file count matches README "X 篇" claims
#   C4 — works/*-translation.md file count matches all translation count claims
#        (badges, <details> summary, Phase 5 mentions, AGENTS snapshot, table rows)
#   C5 — references/articles.md exclusion note "不计入 N 篇" matches C1 authority
#   C6 — translate pipeline local guard: 01-analysis.md must not claim abstract-only
#        when sources/<slug>/source-full.md exists. SKIPs on CI/clean clones.
#   C7 — thinking/ feedback/ prose must not bare-write library counts without
#        snapshot qualifiers (写作时点/当时/此前/首批/首轮/截至/快照)
#
# Usage:  bash scripts/check-consistency.sh        (run from repo root)
# Exits 0 on all-pass, 1 on any failure.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAIL=0
SKIP_MARK="<!-- check-consistency: skip-count -->"
has_skip_mark() { [ -f "$1" ] && grep -qE "^${SKIP_MARK}\$" "$1" 2>/dev/null; }

red()    { printf '\033[31m%s\033[0m' "$1"; }
green()  { printf '\033[32m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }

# ─── C1 ────────────────────────────────────────────────────────────────
echo "[C1] articles.md numbering is contiguous 1..N"
if [ ! -f "references/articles.md" ]; then
  echo "  $(red FAIL) — references/articles.md does not exist"
  FAIL=1
  AUTHORITY=""
else
  nums=$(grep -nE '^### [0-9]+\.' references/articles.md 2>/dev/null | sed -E 's/^[0-9]+:### ([0-9]+)\..*/\1/' || true)
  if [ -z "$nums" ]; then
    echo "  $(green PASS) — 0 entries (empty repository)"
    AUTHORITY="0"
  else
    sorted=$(echo "$nums" | sort -n)
    n=$(echo "$sorted" | wc -l | tr -d ' ')
    expected=$(seq 1 "$n")
    if [ "$sorted" = "$expected" ]; then
      echo "  $(green PASS) — $n contiguous entries (1..$n)"
      AUTHORITY="$n"
    else
      echo "  $(red FAIL) — numbering not contiguous"
      echo "  actual:   $(echo "$sorted" | tr '\n' ' ')"
      echo "  expected: $(echo "$expected" | tr '\n' ' ')"
      FAIL=1
      AUTHORITY=""
    fi
  fi
fi

# ─── C2 ────────────────────────────────────────────────────────────────
echo "[C2] downstream count claims match articles.md"
if [ -z "$AUTHORITY" ]; then
  echo "  $(yellow SKIP) — C1 failed, authority count unknown"
else
  check_count() {
    local file="$1" pattern="$2" label="$3"
    if [ ! -f "$file" ]; then
      echo "  $(yellow SKIP) — $label ($file): file does not exist"
      return
    fi
    if has_skip_mark "$file"; then
      echo "  $(yellow SKIP) — $label ($file): skip-count marker present"
      return
    fi
    local found
    found=$(grep -oE "$pattern" "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+' | head -1 || true)
    if [ -z "$found" ]; then
      echo "  $(yellow SKIP) — $label ($file): pattern '$pattern' not found"
    elif [ "$found" = "$AUTHORITY" ]; then
      echo "  $(green PASS) — $label ($file): $found"
    else
      echo "  $(red FAIL) — $label ($file): claims $found, articles.md says $AUTHORITY"
      FAIL=1
    fi
  }

  check_count "README.md"                        'articles-[0-9]+-'  "README.md badge"
  check_count "prompts/deep-research-tracker.md" '核心文章 [0-9]+ 篇' "deep-research-tracker.md header"
  check_count "references/AGENTS.md"             '[0-9]+ 篇文章'      "references/AGENTS.md overview"
fi

# ─── C3 ────────────────────────────────────────────────────────────────
echo "[C3] subdirectory file counts match README claims"
check_dir_count() {
  local dir="$1" claim_pattern="$2"
  if [ ! -d "$dir" ]; then
    echo "  $(yellow SKIP) — $dir: directory does not exist"
    return
  fi
  local actual
  actual=$(find "$dir" -maxdepth 1 -type f -name '*.md' ! -name 'AGENTS.md' 2>/dev/null | wc -l | tr -d ' ')
  local claim
  claim=$(grep -oE "$claim_pattern" README.md 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  if [ -z "$claim" ]; then
    echo "  $(yellow SKIP) — $dir: README claim pattern '$claim_pattern' not found"
  elif [ "$actual" = "$claim" ]; then
    echo "  $(green PASS) — $dir: $actual files = README claim $claim"
  else
    echo "  $(red FAIL) — $dir: $actual *.md files, README claims $claim 篇"
    FAIL=1
  fi
}

check_dir_count "thinking" '独立思考与质疑（[0-9]+ 篇'
check_dir_count "feedback" '踩坑与迭代心得（[0-9]+ 篇'

# ─── C4 ────────────────────────────────────────────────────────────────
echo "[C4] translation count claims match works/*-translation.md file count"
TRANSLATIONS=$(find works -maxdepth 1 -type f -name '*-translation.md' 2>/dev/null | wc -l | tr -d ' ')

check_against() {
  local file="$1" pattern="$2" label="$3" expected="$4"
  if [ ! -f "$file" ]; then
    echo "  $(yellow SKIP) — $label ($file): file does not exist"
    return
  fi
  local found
  found=$(grep -oE "$pattern" "$file" 2>/dev/null | head -1 | grep -oE '[0-9]+' | head -1 || true)
  if [ -z "$found" ]; then
    echo "  $(yellow SKIP) — $label ($file): pattern '$pattern' not found"
  elif [ "$found" = "$expected" ]; then
    echo "  $(green PASS) — $label ($file): $found"
  else
    echo "  $(red FAIL) — $label ($file): claims $found, expected $expected"
    FAIL=1
  fi
}

check_table_rows() {
  local file="$1" expected="$2"
  if [ ! -f "$file" ]; then
    echo "  $(yellow SKIP) — $file: file does not exist"
    return
  fi
  local rows
  rows=$(grep -cE '\(works/[^)]+-translation\.md\)' "$file" 2>/dev/null || true)
  if [ "$rows" = "$expected" ]; then
    echo "  $(green PASS) — $file translation table rows: $rows"
  else
    echo "  $(red FAIL) — $file translation table: $rows rows, $expected files"
    FAIL=1
  fi
}

check_against "README.md" 'translations-[0-9]+-' "README.md translations badge" "$TRANSLATIONS"
check_against "README.md" '<b>[0-9]+ 篇核心文章的中文翻译' "README.md <details> summary" "$TRANSLATIONS"
check_against "README.md" '[0-9]+ 篇翻译 \+ [0-9]+ 篇原创' "README.md Phase 5 mention" "$TRANSLATIONS"
check_against "AGENTS.md"  'works/，[0-9]+ 篇翻译' "AGENTS.md Phase 5 snapshot" "$TRANSLATIONS"
check_table_rows "README.md" "$TRANSLATIONS"

# ─── C5 ────────────────────────────────────────────────────────────────
echo "[C5] articles.md exclusion note matches authority"
if [ -z "$AUTHORITY" ]; then
  echo "  $(yellow SKIP) — C1 failed, authority count unknown"
else
  EXCLUDED=$(grep -oE '不计入 [0-9]+ 篇' references/articles.md 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  if [ -z "$EXCLUDED" ]; then
    echo "  $(yellow SKIP) — references/articles.md: '不计入 N 篇' note not found (optional for empty repo)"
  elif [ "$EXCLUDED" = "$AUTHORITY" ]; then
    echo "  $(green PASS) — references/articles.md: 不计入 $EXCLUDED 篇"
  else
    echo "  $(red FAIL) — references/articles.md: 不计入 $EXCLUDED 篇, AUTHORITY says $AUTHORITY"
    FAIL=1
  fi
fi

# ─── C6 ────────────────────────────────────────────────────────────────
echo "[C6] translate analysis files don't falsely claim abstract-only when full text exists"
shopt -s nullglob
analyses=(translate/*/translations/*/01-analysis.md)
if [ "${#analyses[@]}" -eq 0 ]; then
  echo "  $(yellow SKIP) — no translate/ candidate drafts present (CI checkout or clean clone)"
else
  c6_checked=0
  for analysis in "${analyses[@]}"; do
    slug=$(basename "$(dirname "$analysis")")
    base=${analysis%%/translations/*}
    fulltext="$base/sources/$slug/source-full.md"
    [ -f "$fulltext" ] || continue
    c6_checked=$((c6_checked + 1))
    # Split sentences to lines first (。→ newline) to avoid multibyte bracket expressions
    if awk '{gsub(/。/, "\n"); print}' "$analysis" | grep -qE '非全文|补抓.*全文|只抓.*摘要|只是摘要页'; then
      echo "  $(red FAIL) — $analysis: still claims abstract-only, but $fulltext exists"
      FAIL=1
    else
      echo "  $(green PASS) — $slug: analysis consistent with captured full text"
    fi
  done
  [ "$c6_checked" -eq 0 ] && echo "  $(yellow SKIP) — no slug has a source-full.md to check against"
fi

# ─── C7 ────────────────────────────────────────────────────────────────
echo "[C7] no bare library-count claims in thinking/ feedback/"
C7_PATTERN='[0-9]+ ?篇(文章|翻译)|[0-9]+ ?大概念'
C7_QUALIFIER='写作时点|当时|此前|首批|首轮|截至|快照'
c7_hits=0
c7_fails=0
while IFS= read -r hit; do
  [ -z "$hit" ] && continue
  c7_hits=$((c7_hits + 1))
  c7_file=${hit%%:*}
  c7_rest=${hit#*:}
  c7_line=${c7_rest%%:*}
  c7_text=${c7_rest#*:}
  if ! echo "$c7_text" | grep -qE "$C7_QUALIFIER"; then
    echo "  $(red FAIL) — $c7_file:$c7_line: bare count claim without snapshot qualifier"
    echo "    fix: drop the number (link references/articles.md) or qualify the line (写作时点/当时/此前/首批/首轮/截至/快照)"
    FAIL=1
    c7_fails=$((c7_fails + 1))
  fi
done <<EOF
$(grep -rnE "$C7_PATTERN" thinking feedback --include='*.md' 2>/dev/null || true)
EOF
if [ "$c7_fails" -eq 0 ]; then
  echo "  $(green PASS) — $c7_hits count mention(s), all snapshot-qualified or none present"
fi

# ─── Summary ───────────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "$(green '✓ consistency checks passed')"
  exit 0
else
  echo "$(red '✗ consistency checks failed') — fix the entries above and re-run"
  exit 1
fi
