#!/usr/bin/env bash
# Build the v2 manuscript: regenerate every number, gate on "no hand-typed result", then
# pdflatex -> bibtex -> pdflatex x2, with the Type-3 font check IEEE requires.
set -e
BIN="C:/Users/P520/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
PY="C:/Users/P520/timesfm-sales/.venv/Scripts/python.exe"
cd "$(dirname "$0")"
export PATH="$BIN:$PATH"

if [ "${1:-}" != "--no-numbers" ]; then
  "$PY" ../src/paper_numbers.py --results ../results_v2 >/dev/null && echo "numbers.tex regenerated"
  "$PY" ../src/paper_numbers.py --check --sec sec_v2 --main main_v2.tex || {
    echo "--- hand-typed results found in the prose (see above); build refused ---"; exit 1; }
fi

"$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error main_v2.tex >/dev/null || \
  { echo "--- pdflatex pass 1 failed ---"; grep -E "^! |^l\.[0-9]+|Undefined" main_v2.log | head -25; exit 1; }
"$BIN/bibtex.exe" main_v2 >/dev/null 2>&1 || true
"$BIN/pdflatex.exe" -interaction=nonstopmode main_v2.tex >/dev/null 2>&1 || true
"$BIN/pdflatex.exe" -interaction=nonstopmode main_v2.tex >/dev/null 2>&1 || true
PAGES=$("$BIN/pdfinfo.exe" main_v2.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "built main_v2.pdf (${PAGES:-?} pages)"
UNDEF=$(grep -c "Citation.*undefined" main_v2.log 2>/dev/null || true)
REF=$(grep -c "Reference.*undefined" main_v2.log 2>/dev/null || true)
OVER=$(grep -c 'Overfull .hbox' main_v2.log 2>/dev/null || true)
echo "  undefined citations: $UNDEF | undefined refs: $REF | overfull hboxes: $OVER"
T3=$("$BIN/pdffonts.exe" main_v2.pdf 2>/dev/null | grep -c "Type 3" || true)
if [ "${T3:-0}" -gt 0 ]; then echo "  WARNING: $T3 Type-3 font(s) embedded -- IEEE rejects these"; fi
# a float taller than the page is placed anyway and clipped at the page foot
BIG=$(grep -c "Float too large" main_v2.log 2>/dev/null || true)
[ "${BIG:-0}" -gt 0 ] && { echo "  WARNING: $BIG float(s) too large for the page (clipped):"; grep "Float too large" main_v2.log | sed 's/^/    /'; }
# unresolved number macros or references print as ??
QQ=$("$BIN/pdftotext.exe" main_v2.pdf - 2>/dev/null | grep -c "??" || true)
[ "${QQ:-0}" -gt 0 ] && echo "  unresolved number macros or references printed as ??: $QQ"
# generated tables/figures that are still missing print a placeholder in the draft; the
# release build (--release) refuses them, as it refuses undefined number macros and
# unresolved references.
MISS=$(grep -c "^MISSING" main_v2.log 2>/dev/null || true)
[ "${MISS:-0}" -gt 0 ] && { echo "  missing generated inputs: $MISS"; grep "^MISSING" main_v2.log | sort -u | sed 's/^/    /'; }
if [ "${1:-}" = "--release" ] || [ "${2:-}" = "--release" ]; then
  [ "${MISS:-0}" -eq 0 ] && [ "${REF:-0}" -eq 0 ] && [ "${UNDEF:-0}" -eq 0 ] && [ "${T3:-0}" -eq 0 ] \
    && [ "${BIG:-0}" -eq 0 ] && [ "${QQ:-0}" -eq 0 ] \
    || { echo "--- release build refused ---"; exit 1; }
  echo "release build OK"
fi
