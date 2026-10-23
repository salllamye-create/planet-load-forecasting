#!/usr/bin/env bash
# Build the Elsevier submission version (main_els.tex) from the numbers, tables and figures that
# build_v2.sh regenerates: pdflatex -> bibtex -> pdflatex x2, then the same checks as the IEEE
# build (undefined citations/references, overfull boxes, missing generated inputs, Type-3 fonts)
# plus a check for the submission placeholders (country, repository URL).
#   build_els.sh            draft build
#   build_els.sh --release  refuses missing inputs, undefined references or citations, Type-3
#                           fonts and placeholders
set -e
BIN="C:/Users/P520/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"
export PATH="$BIN:$PATH"
M=main_els

"$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error $M.tex >/dev/null || \
  { echo "--- pdflatex pass 1 failed ---"; grep -E "^! |^l\.[0-9]+|Undefined" $M.log | head -25; exit 1; }
"$BIN/bibtex.exe" $M >/dev/null 2>&1 || true
"$BIN/pdflatex.exe" -interaction=nonstopmode $M.tex >/dev/null 2>&1 || true
"$BIN/pdflatex.exe" -interaction=nonstopmode $M.tex >/dev/null 2>&1 || true
PAGES=$("$BIN/pdfinfo.exe" $M.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "built $M.pdf (${PAGES:-?} pages)"
UNDEF=$(grep -c "Citation.*undefined" $M.log 2>/dev/null || true)
REF=$(grep -c "Reference.*undefined" $M.log 2>/dev/null || true)
OVER=$(grep -c 'Overfull .hbox' $M.log 2>/dev/null || true)
echo "  undefined citations: $UNDEF | undefined refs: $REF | overfull hboxes: $OVER"
T3=$("$BIN/pdffonts.exe" $M.pdf 2>/dev/null | grep -c "Type 3" || true)
[ "${T3:-0}" -gt 0 ] && echo "  WARNING: $T3 Type-3 font(s) embedded (bitmapped text)"
MISS=$(grep -c "^MISSING" $M.log 2>/dev/null || true)
[ "${MISS:-0}" -gt 0 ] && { echo "  missing generated inputs: $MISS"; grep "^MISSING" $M.log | sort -u | sed 's/^/    /'; }
QQ=$("$BIN/pdftotext.exe" $M.pdf - 2>/dev/null | grep -c "??" || true)
[ "${QQ:-0}" -gt 0 ] && echo "  unresolved number macros or references printed as ??: $QQ"
PH=$(grep -c -E "ELSCOUNTRY|ELSREPOSITORY|<anonymised>" $M.tex || true)
[ "${PH:-0}" -gt 0 ] && echo "  submission placeholders still in $M.tex: $PH"
if [ "${1:-}" = "--release" ]; then
  [ "${MISS:-0}" -eq 0 ] && [ "${REF:-0}" -eq 0 ] && [ "${UNDEF:-0}" -eq 0 ] && [ "${T3:-0}" -eq 0 ] \
    && [ "${PH:-0}" -eq 0 ] && [ "${QQ:-0}" -eq 0 ] \
    || { echo "--- release build refused ---"; exit 1; }
  echo "release build OK"
fi
