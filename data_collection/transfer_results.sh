#!/bin/bash
# ==========================================
# Move all result files from /home to /orange
# Author: ji757406.ucf
# ==========================================

SRC="/home/ji757406.ucf/router/results"
DEST="/orange/sgao1/ji757406.ucf/route/results"

echo "🔍 Checking source and destination..."
if [ ! -d "$SRC" ]; then
  echo "❌ Source directory not found: $SRC"
  exit 1
fi

mkdir -p "$DEST"

echo "🚀 Moving files from $SRC → $DEST ..."
echo

# Step 1: 用 rsync 拷贝（带进度，保留属性）
rsync -ah --progress "$SRC"/ "$DEST"/

# Step 2: 校验 rsync 返回码
if [ $? -ne 0 ]; then
  echo "❌ Error during rsync copy! Original files will NOT be deleted."
  exit 1
fi

echo
echo "✅ Copy phase complete. Verifying..."
sleep 1

# Step 3: 简单比对文件数确保传输完整
src_count=$(find "$SRC" -type f | wc -l)
dst_count=$(find "$DEST" -type f | wc -l)

echo "Source files: $src_count"
echo "Target files: $dst_count"

if [ "$dst_count" -lt "$src_count" ]; then
  echo "⚠️ File count mismatch! Aborting deletion to stay safe."
  exit 1
fi


echo "✅ Transfer complete!"
echo "📦 Source: $SRC"
echo "📁 Target: $DEST"