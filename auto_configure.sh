#!/bin/bash

echo "🛠️ 特征提取自动配置脚本"

# === 读取用户输入 ===
read -p "请输入 PETSC_DIR 路径（如 /home/user/petsc-3.20.0）: " PETSC_DIR
read -p "请输入 PETSC_ARCH 名称（如 arch-linux-c-debug）: " PETSC_ARCH
read -p "请输入 Python 脚本中的矩阵数据路径 path （如 /home/user/data/smatrix/）: " MATRIX_PATH
read -p "请输入 Python 脚本中的特征输出目录 feature_output_path （如 ./feature/）: " FEATURE_OUTPUT_PATH

# === 修改 Makefile 中的 PETSC_DIR 和 PETSC_ARCH ===
MAKEFILE=makefile
if [ ! -f "$MAKEFILE" ]; then
    echo "❌ 错误：找不到 makefile 文件"
    exit 1
fi

echo "📄 正在修改 makefile 中的 PETSC_DIR 和 PETSC_ARCH..."

# 宽松匹配
sed -i "s|^\s*PETSC_DIR\s*[?]*=.*|PETSC_DIR = $PETSC_DIR|" "$MAKEFILE"
sed -i "s|^\s*PETSC_ARCH\s*[?]*=.*|PETSC_ARCH = $PETSC_ARCH|" "$MAKEFILE"

echo "✅ Makefile 修改完成"

# === 修改 Python 脚本中的 path 和 feature_output_path  ===
PYFILE=getFeature_multi_bypath.py
if [ ! -f "$PYFILE" ]; then
    echo "❌ 错误：找不到 getFeature_multi_bypath.py 文件"
    exit 1
fi

echo "📄 正在修改 Python 脚本中的 path 和 feature_output_path 变量..."

# 使用sed高级捕获，保留行首缩进部分，再改后半部分
sed -i -E "s|^([[:space:]]*)path = \".*\"|\1path = \"$MATRIX_PATH\"|" "$PYFILE"
sed -i -E "s|^([[:space:]]*)feature_output_path = \".*\"|\1feature_output_path = \"$FEATURE_OUTPUT_PATH\"|" "$PYFILE"

echo "✅ Python 脚本修改完成"

# === 自动make编译 ===
echo -e "\n🛠️ 开始 make 编译..."

make

if [ $? -ne 0 ]; then
    echo "❌ make 编译失败，请检查错误信息！"
    exit 1
fi

echo "✅ make 编译成功！"

# === 自动运行 getFeature_multi_bypath.py ===
echo -e "\n🚀 开始执行 getFeature_multi_bypath.py ..."

python getFeature_multi_bypath.py

if [ $? -ne 0 ]; then
    echo "❌ 执行 Python 脚本失败，请检查错误信息！"
    exit 1
fi

echo "🎉 全部完成！特征提取任务已运行完毕！"
