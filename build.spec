# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Skin Recovery Assessor

import sys
from pathlib import Path

block_cipher = None

# 静态文件（Three.js）
static_dir = Path('static')
datas = []
if static_dir.exists():
    datas.append(('static', 'static'))

a = Analysis(
    ['web_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'numpy',
        'cv2',
        'PIL',
        'pillow_heif',
        'matplotlib',
        'skimage',
        'scipy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['open3d', 'tkinter', 'PyQt5', 'PyQt6', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SkinRecoveryAssessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 显示控制台以便调试和关闭
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# macOS .app bundle (仅 macOS 有效)
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='皮肤恢复评估系统.app',
        icon=None,
        bundle_identifier='com.skinrecovery.assessor',
        info_plist={
            'CFBundleDisplayName': '皮肤恢复评估系统',
            'CFBundleName': 'Skin Recovery Assessor',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )
