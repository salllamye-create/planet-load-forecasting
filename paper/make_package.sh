#!/usr/bin/env bash
# Assemble the submission / arXiv source package from the built manuscript:
#   paper/release/<stamp>/els/   main_els.tex + .bbl, numbers.tex, sec_v2, tables_v2, figures, bib
#   paper/release/<stamp>/ieee/  the same for main_v2.tex
# Each package is test-built in place (pdflatex x2 on the shipped .bbl, as arXiv does), then zipped.
set -e
BIN="C:/Users/P520/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"
export PATH="$BIN:$PATH"
STAMP=$(date '+%Y%m%d-%H%M')
OUT="release/$STAMP"
mkdir -p "$OUT"

pack() {  # pack <main stem> <subdir>
  local M=$1 D="$OUT/$2"
  [ -f "$M.bbl" ] || { echo "missing $M.bbl: build $M first"; exit 1; }
  mkdir -p "$D/sec_v2" "$D/tables_v2" "$D/figures_v2"
  cp "$M.tex" "$M.bbl" numbers.tex refs.bib refs_v2.bib refs_rev.bib "$D/"
  cp sec_v2/*.tex "$D/sec_v2/"
  cp tables_v2/*.tex "$D/tables_v2/"
  # only the figures the manuscript includes
  grep -o -h -E "figv\{[A-Za-z0-9_]+\}" "$M.tex" sec_v2/*.tex | sed -E 's/figv\{(.*)\}/\1/' | sort -u | while read -r f; do
    cp "../figures_v2/$f.pdf" "$D/figures_v2/"
  done
  # figures live beside the source in the package
  sed -i 's#\\graphicspath{{../figures_v2/}{../figures/}}#\\graphicspath{{figures_v2/}}#; s#\\IfFileExists{../figures_v2/#\\IfFileExists{figures_v2/#' "$D/$M.tex"
  ( cd "$D" && "$BIN/pdflatex.exe" -interaction=nonstopmode -halt-on-error "$M.tex" >/dev/null \
      && "$BIN/pdflatex.exe" -interaction=nonstopmode "$M.tex" >/dev/null 2>&1 \
      && echo "  $2: test build OK ($("$BIN/pdfinfo.exe" "$M.pdf" | awk '/^Pages/{print $2}') pages)" \
      || { echo "  $2: test build FAILED"; grep -E "^! " "$M.log" | head -5; exit 1; } )
  ( cd "$D" && rm -f "$M.aux" "$M.log" "$M.out" "$M.toc" "$M.spl" && cd .. && zip -qr "$2-$STAMP.zip" "$2" )
  echo "  $OUT/$2-$STAMP.zip"
}

pack main_els els
pack main_v2 ieee
echo "package complete: $OUT"
