#!/usr/bin/env bash
# Usage: caucus "generate names..." | ./scripts/check-names.sh
# Or:    ./scripts/check-names.sh < names.txt
#
# Checks each name against crates.io and PyPI for availability.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
DIM='\033[2m'
RESET='\033[0m'

check_crates() {
  local name="$1"
  # crates.io API returns 404 for nonexistent crates
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" "https://crates.io/api/v1/crates/$name")
  [[ "$status" == "404" ]]
}

check_pypi() {
  local name="$1"
  local status
  status=$(curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/$name/json")
  [[ "$status" == "404" ]]
}

printf "\n%-20s %-12s %-12s %s\n" "NAME" "CRATES.IO" "PYPI" "STATUS"
printf "%-20s %-12s %-12s %s\n" "----" "---------" "----" "------"

both_free=()

while IFS= read -r line; do
  # Strip numbering, punctuation, whitespace, lowercase it
  name=$(echo "$line" | sed 's/^[0-9]*[.)]\s*//' | tr -d '[:punct:]' | xargs | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
  [[ -z "$name" ]] && continue

  crates="?"
  pypi="?"

  if check_crates "$name"; then
    crates="${GREEN}free${RESET}"
    crates_free=true
  else
    crates="${RED}taken${RESET}"
    crates_free=false
  fi

  if check_pypi "$name"; then
    pypi="${GREEN}free${RESET}"
    pypi_free=true
  else
    pypi="${RED}taken${RESET}"
    pypi_free=false
  fi

  if $crates_free && $pypi_free; then
    status="${GREEN}★ AVAILABLE${RESET}"
    both_free+=("$name")
  elif $crates_free || $pypi_free; then
    status="${YELLOW}partial${RESET}"
  else
    status="${DIM}taken${RESET}"
  fi

  printf "%-20s %-23s %-23s %b\n" "$name" "$crates" "$pypi" "$status"

  # Rate limit to avoid getting blocked
  sleep 0.3
done

if [[ ${#both_free[@]} -gt 0 ]]; then
  printf "\n${GREEN}Available on both crates.io and PyPI:${RESET}\n"
  for name in "${both_free[@]}"; do
    echo "  $name"
  done
fi
