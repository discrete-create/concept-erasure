#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

echo "=============================================="
echo "开始执行：download-sd.py（Stable Diffusion模型）"
echo "=============================================="
python3 download/download-sd.py

if [ $? -ne 0 ]; then
    echo "错误：download-sd.py 执行失败，终止后续操作"
    exit 1
fi

echo -e "\n=============================================="
echo "download-sd.py 执行完成！"
echo "==============================================\n"


echo "=============================================="
echo "开始执行：download-llm.py（Qwen3-VL-8B模型）"
echo "=============================================="
python3 download/download-llm.py

if [ $? -ne 0 ]; then
    echo "错误：download-llm.py 执行失败"
    exit 1
fi

echo -e "\n=============================================="
echo "download-llm.py 执行完成！"
echo "=============================================="