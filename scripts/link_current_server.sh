#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

while IFS=$'\t' read -r group required logical bytes digest original description; do
  [[ "$group" == "group" ]] && continue
  mkdir -p "$(dirname "$logical")"
  if [[ -e "$logical" || -L "$logical" ]]; then
    continue
  fi
  ln -s "$original" "$logical"
  echo "linked $logical"
done < external_manifest.tsv

python scripts/check_external.py --fast
