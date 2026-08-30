@echo off
REM 皮肤恢复评估系统打包脚本 (Windows)
REM 运行方式: 双击运行或在命令行执行 build_win.bat

echo ==========================================
echo   皮肤恢复评估系统 - Windows 打包
echo ==========================================

REM 检查 PyInstaller
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo 安装 PyInstaller...
    pip install pyinstaller
)

REM 检查静态文件
if not exist "static\three.min.js" (
    echo 下载 Three.js...
    mkdir static 2>nul
    powershell -Command "Invoke-WebRequest -Uri 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js' -OutFile 'static\three.min.js'"
)

REM 清理旧构建
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo 开始打包...
pyinstaller build.spec --clean --noconfirm

echo.
echo ==========================================
echo   打包完成！
echo ==========================================
echo 生成: dist\SkinRecoveryAssessor.exe
echo.
echo 如需创建安装包，可使用 Inno Setup:
echo   https://jrsoftware.org/isinfo.php
echo.
pause
