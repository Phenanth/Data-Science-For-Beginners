#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"

# 1) 基本检查
git rev-parse --is-inside-work-tree >/dev/null

# 工作区必须干净
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit/stash your changes first." >&2
  exit 1
fi

# 2) 确保 upstream 存在
if ! git remote | grep -qx upstream; then
  echo "Remote 'upstream' not found.
Add it once:
  git remote add upstream <UPSTREAM_URL>
  git remote set-url --push upstream DISABLE   # optional to prevent push" >&2
  exit 1
fi

# 3) 获取最新
git fetch --prune upstream
git fetch --prune origin

# 4) 切分支
git checkout "$BRANCH"

# 5) rebase 到 upstream
git rebase "upstream/$BRANCH"

# 6) 推回你的 fork
git push origin "$BRANCH" --force-with-lease

echo "✅ Sync done: origin/$BRANCH = upstream/$BRANCH + your commits (rebased)."
echo ""
read -n 1 -s -r -p "Press any key to exit..."
echo