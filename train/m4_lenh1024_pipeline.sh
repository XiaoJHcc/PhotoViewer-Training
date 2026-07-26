#!/usr/bin/env bash
# m4_lenh1024_pipeline.sh — 梯3：ViT-L/16@1024 导出 → parity → 全库重提 → M4 复测
# 2026-07-19 挂后台无人值守（决策 8 梯3；用户裁定：机械活后台跑）。
# 已知坑：DatasetBuilder 主流程结束后进程悬挂（ONNX 清理期）——run_dotnet 看门狗
# 检测到 GATE 打印后等 30s 再 kill；M4 复测前校验双路 CLS 覆盖 9418。
set -u
cd /d/Git/PhotoViewer
export PYTHONUTF8=1
PY=Tools/.venv/Scripts/python.exe
ONNX=D:/PhotoDB/dataset/models/dinov3_vitl16_1024.onnx
MID=~/.cache/modelscope/hub/models/facebook/dinov3-vitl16-pretrain-lvd1689m   # HF 门控 401，走本机 ModelScope 缓存（梯2 同款）
NEWID=dinov3_vitl16_f32_1024_v1
LOGD=Training/train/out/m4_lenh1024
mkdir -p "$LOGD"

run_dotnet() {  # $1=log 文件，其余=命令；GATE 出现即收尾
  local log="$1"; shift
  "$@" > "$log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if grep -q "GATE" "$log" 2>/dev/null; then sleep 30; kill $pid 2>/dev/null; break; fi
    sleep 15
  done
  wait $pid 2>/dev/null
  grep -E "GATE|ERROR|FAIL" "$log" | tail -5
}

echo "===== [1/4] export ViT-L@1024 ====="
$PY Training/onnx/export_dinov3_onnx.py --model-id $MID --output $ONNX --image-size 1024 || exit 1

echo "===== [2/4] parity ====="
$PY Training/onnx/verify_onnx_parity.py --model-id $MID --onnx $ONNX --image-size 1024 --samples 100 || exit 1

echo "===== [3/4] 全库重提（manifest + 旧批） ====="
run_dotnet "$LOGD/extract_manifest.log" dotnet run --project Training/DatasetBuilder -- \
  --manifest D:/PhotoDB/dataset/manifest.2026-07-19.json \
  --model-file $ONNX --model-id $NEWID --no-patch
run_dotnet "$LOGD/extract_20240212.log" dotnet run --project Training/DatasetBuilder -- \
  D:/PhotoDB/20240212 --db D:/PhotoDB/dataset/photos_dataset.db \
  --model-file $ONNX --model-id $NEWID --no-patch

echo "===== [4/4] 覆盖校验 + M4 复测 ====="
$PY - <<'EOF'
import sqlite3, sys
c = sqlite3.connect('file:D:/PhotoDB/dataset/photos_dataset.db?mode=ro', uri=True)
cov = {m: n for m, n in c.execute(
    "SELECT model_id, COUNT(*) FROM photo_features WHERE model_id LIKE 'dinov3_vitl16_f32_1024%' GROUP BY model_id")}
print("coverage:", cov)
need = {'dinov3_vitl16_f32_1024_v1': 9418, 'dinov3_vitl16_f32_1024_v1+clhe2.0ycc1.0': 9418}
sys.exit(0 if all(cov.get(k, 0) >= v for k, v in need.items()) else 1)
EOF
if [ $? -ne 0 ]; then echo "[ERROR] 覆盖不足，M4 不跑"; exit 1; fi

$PY Training/train/m4_baseline.py --model-id "dinov3_vitl16_f32_1024_v1+clhe2.0ycc1.0" \
  --out Training/train/out/m4_lenh1024
echo "===== 梯3 管线完成 ====="
