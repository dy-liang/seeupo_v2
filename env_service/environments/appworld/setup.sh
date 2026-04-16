#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
APPWORLD_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. 环境变量配置
echo "📁 设置环境变量..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export APPWORLD_ROOT="$APPWORLD_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"



# 3. Conda 环境创建
if ! conda info --envs | grep -w "appworld" &>/dev/null; then
    echo "🐍 创建 Conda 环境 appworld（Python 3.11）..."
    conda create -n appworld python=3.12.0 -y
else
    echo "⚠️ Conda 环境 appworld 已存在，请删除或修改。（本次已跳过创建）。"
fi

# 4. 安装依赖
echo "📦 安装 libcst..."
conda install -n appworld -y libcst

echo "📋 安装 Python 依赖..."
conda run -n appworld pip install -r "$SCRIPT_DIR/requirements.txt"
# https://github.com/modelscope/AgentEvolver/issues/6
conda run -n appworld pip uninstall -y click
conda run -n appworld pip install click==8.2.0

# 5. 初始化 appworld
echo "📁 初始化 appworld..."
conda run -n appworld appworld install

# 6. 下载数据
echo "📦 下载数据（失败则使用备用下载）..."
if ! conda run -n appworld appworld download data; then
    echo "⚠️ 自动下载失败，尝试从备用地址获取数据..."
    wget -O "$APPWORLD_ROOT/appworld_data.zip" "https://dail-wlcb.oss-accelerate.aliyuncs.com/eric.czq/appworld_data.zip"
    mkdir -p /tmp/unziptemp
    unzip "$APPWORLD_ROOT/appworld_data.zip" -d /tmp/unziptemp
    mv /tmp/unziptemp/*/* "$APPWORLD_ROOT"
    rm -rf /tmp/unziptemp "$APPWORLD_ROOT/appworld_data.zip"
fi

echo "✅ 设置完成！"

echo ""
echo "👉 启动方法："
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate appworld"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash appworld.sh"
echo "----------------------------------------"
