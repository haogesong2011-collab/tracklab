#!/bin/sh
# 用项目虚拟环境启动，不要使用系统 PATH 里的 python3。
cd "$(dirname "$0")" || exit 1

PY=""
for c in .venv/bin/python3 .venv/bin/python .venv/bin/python3.13; do
  if [ -x "$c" ]; then
    PY=$c
    break
  fi
done

if [ -z "$PY" ]; then
  echo "虚拟环境不完整（找不到 .venv 里的 Python）。请整段复制执行："
  echo ""
  echo "  cd ~/Desktop/tracklab"
  echo "  rm -rf .venv"
  echo "  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv"
  echo "  ls .venv/bin"
  echo "  .venv/bin/python3 -m pip install -U pip"
  echo "  .venv/bin/python3 -m pip install -r requirements.txt"
  echo "  .venv/bin/python3 -m app"
  echo ""
  echo "若 ls .venv/bin 里没有 python3，把实际文件名换成 python 或 python3.13。"
  if [ -d .venv/bin ]; then
    echo "当前 .venv/bin 内容："
    ls -la .venv/bin
  fi
  exit 1
fi

if ! "$PY" -c "import PySide6" 2>/dev/null; then
  echo "正在安装依赖到虚拟环境（$PY）…"
  "$PY" -m pip install -U pip
  "$PY" -m pip install -r requirements.txt || exit 1
fi
exec "$PY" -m app "$@"
