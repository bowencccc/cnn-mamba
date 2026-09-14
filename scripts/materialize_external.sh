#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DESTINATION [raw|checkpoint|all]" >&2
  exit 2
fi

destination=$1
selected_group=${2:-all}
case "$selected_group" in raw|checkpoint|all) ;; *) echo "invalid group: $selected_group" >&2; exit 2 ;; esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$destination"
destination=$(cd "$destination" && pwd)
cd "$repo_root"

while IFS=$'\t' read -r group required logical bytes digest original description; do
  [[ "$group" == "group" ]] && continue
  [[ "$selected_group" != "all" && "$group" != "$selected_group" ]] && continue
  if [[ ! -f "$logical" ]]; then
    echo "missing: $logical" >&2
    exit 1
  fi
  mkdir -p "$destination/$(dirname "$logical")"
  cp -L --reflink=auto "$logical" "$destination/$logical"
  echo "copied $logical"
done < external_manifest.tsv

cp external_manifest.tsv "$destination/external_manifest.tsv"
echo "external bundle ready: $destination"
