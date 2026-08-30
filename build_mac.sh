#!/bin/bash
# 皮肤恢复评估系统打包脚本 (macOS)
# 运行方式: bash build_mac.sh

set -e

echo "=========================================="
echo "  皮肤恢复评估系统 - macOS 打包"
echo "=========================================="

# 检查 PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "安装 PyInstaller..."
    pip install pyinstaller
fi

# 检查静态文件
if [ ! -f "static/three.min.js" ]; then
    echo "下载 Three.js..."
    mkdir -p static
    curl -o static/three.min.js https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js
fi

# 清理旧构建
rm -rf build dist *.spec.bak 2>/dev/null || true

echo ""
echo "开始打包..."
pyinstaller build.spec --clean --noconfirm

echo ""
echo "=========================================="
echo "  打包完成！"
echo "=========================================="

if [ -d "dist/皮肤恢复评估系统.app" ]; then
    echo "生成: dist/皮肤恢复评估系统.app"
    echo ""
    echo "要创建 DMG 安装镜像，运行:"
    echo "  hdiutil create -volname '皮肤恢复评估系统' -srcfolder dist/皮肤恢复评估系统.app -ov -format UDZO dist/皮肤恢复评估系统.dmg"
else
    echo "生成: dist/SkinRecoveryAssessor"
fi

echo ""
echo "运行测试:"
echo "  ./dist/SkinRecoveryAssessor"
