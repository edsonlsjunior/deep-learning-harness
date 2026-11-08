#!/bin/bash
cd "$(dirname "$0")" || exit 1
for candidate in "$HOME/.local/bin/python3.12" python3.12 python3.11 python3 /opt/homebrew/bin/python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,11))' 2>/dev/null; then
      "$candidate" start.py "$@"
      result=$?
      if [ "$result" -ne 0 ]; then read -r -p "Pressione Enter para fechar…"; fi
      exit "$result"
    fi
  fi
done
echo "Instale Python 3.11+ e Node.js LTS. Depois abra este arquivo novamente."
read -r -p "Pressione Enter para fechar…"
exit 1
