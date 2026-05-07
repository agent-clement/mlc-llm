#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# TeXLive.net's multipart API requires the main file to be named document.tex
# and rejects subdirectory filenames. Keep the source tree arXiv-shaped, but
# flatten the submitted copy for the online compiler.
sed 's#figures/##g' "${ROOT}/paper.tex" > "${TMP_DIR}/document.tex"
cp "${ROOT}/references.bib" "${TMP_DIR}/references.bib"
cp "${ROOT}/figures/"*.tex "${TMP_DIR}/"

curl -L --fail --show-error --silent 'https://texlive.net/cgi-bin/latexcgi' \
  -F 'engine=pdflatex' \
  -F 'bibcmd=bibtex' \
  -F 'return=pdf' \
  -F "filename[]=document.tex" -F "filecontents[]=@${TMP_DIR}/document.tex;type=text/plain" \
  -F "filename[]=references.bib" -F "filecontents[]=@${TMP_DIR}/references.bib;type=text/plain" \
  -F "filename[]=system_pipeline.tex" -F "filecontents[]=@${TMP_DIR}/system_pipeline.tex;type=text/plain" \
  -F "filename[]=image_path.tex" -F "filecontents[]=@${TMP_DIR}/image_path.tex;type=text/plain" \
  -F "filename[]=kernel_breakdown.tex" -F "filecontents[]=@${TMP_DIR}/kernel_breakdown.tex;type=text/plain" \
  -F "filename[]=optimization_timeline.tex" -F "filecontents[]=@${TMP_DIR}/optimization_timeline.tex;type=text/plain" \
  -F "filename[]=runtime_visibility.tex" -F "filecontents[]=@${TMP_DIR}/runtime_visibility.tex;type=text/plain" \
  -F "filename[]=throughput_parity.tex" -F "filecontents[]=@${TMP_DIR}/throughput_parity.tex;type=text/plain" \
  -F "filename[]=memory_usage.tex" -F "filecontents[]=@${TMP_DIR}/memory_usage.tex;type=text/plain" \
  -F "filename[]=model_structure.tex" -F "filecontents[]=@${TMP_DIR}/model_structure.tex;type=text/plain" \
  -o "${ROOT}/paper-online.pdf"

file "${ROOT}/paper-online.pdf"
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo "${ROOT}/paper-online.pdf" | sed -n '1,20p'
fi
