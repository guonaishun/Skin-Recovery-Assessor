from __future__ import annotations

import base64
import json
import argparse
import logging
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

# 注册 HEIF/HEIC 插件 (兼容 iPhone、华为、小米等手机拍摄的 HEIF 格式)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

from app import MODE_LABELS, analyze, colorize_heatmap, overlay_heatmap


HOST = "127.0.0.1"
PORT = 7860


def get_resource_path() -> Path:
    """获取资源目录路径（兼容 PyInstaller 打包和普通运行）"""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def load_three_js() -> str:
    """加载 Three.js 内容，优先本地文件，失败则使用 CDN 链接"""
    res_path = get_resource_path()
    # 尝试多个可能的路径
    for p in [res_path / "static" / "three.min.js", res_path / "three.min.js"]:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    # 本地未找到，使用 CDN 链接
    return None


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Skin Recovery Assessor</title>
  {{THREE_JS_PLACEHOLDER}}
  <style>
    :root {
      --bg: #081019; --panel-bg: #111b27; --panel-border: #243449;
      --text: #eaf2ff; --text-muted: #91a6bc;
      --mode-border: #2b3d53; --mode-bg: #0b141f; --mode-text: #dcecff;
      --mode-active-border: #37d5ff; --mode-active-bg: #0d3140;
      --upload-border: #3b536e; --upload-bg: #0b141f; --file-name: #f7d154;
      --btn-bg: #1f9fca; --btn-disabled: #536576;
      --log-bg: #07101a; --log-border: #223247; --log-text: #bdefff;
      --preview-bg: #07101a; --preview-border: #223247; --preview-text: #60768e;
      --progress-text: #a9bfd7; --bar-bg: #07101a; --bar-border: #223247;
      --fill-start: #37d5ff; --fill-end: #f7d154;
      --hero-bg: #07101a; --hero-border: #223247; --hero-text: #60768e;
      --scan-color: #37d5ff; --score-color: #37d5ff; --verdict-color: #dce8f4;
      --radar-bg: #07101a; --radar-border: #223247;
      --th-color: #9fb7cf; --td-color: #eaf2ff; --td-border: #243449;
      --pipe-bg: rgba(7,16,26,.78); --pipe-border: #2b3d53;
      --pipe-index: #f7d154; --pipe-title: #ffffff; --pipe-desc: #91a6bc;
      --pipe-img-bg: #07101a; --pipe-img-border: #2b3d53; --pipe-done: #2ecc71;
      --pipe-label: #60768e; --accent-rgb: 55,213,255;
      --pipeline-g1: #07101a; --pipeline-g2: #101b28;
      --model-inner: #0a1825; --model-outer: #06111c; --model-border: #223247;
      --model-glow: rgba(55,213,255,.15); --model-inner-glow: rgba(55,213,255,.05);
      --model-ph: #60768e;
      --ctrl-bg: rgba(7,16,26,.8); --ctrl-border: #2b3d53; --ctrl-text: #91a6bc;
      --ctrl-h-border: #37d5ff; --ctrl-h-text: #fff;
      --ctrl-a-border: #37d5ff; --ctrl-a-bg: rgba(55,213,255,.15); --ctrl-a-text: #37d5ff;
      --hud-text: #cfe9ff; --hud-bg: rgba(7,16,26,.72); --hud-border: #2b3d53;
      --badge-bg: rgba(7,16,26,.88); --clear-color: #06111c;
      --theme-text: #eaf2ff; --theme-bg: #111b27; --theme-border: #2b3d53;
    }
    body.light {
      --bg: #f0f2f5; --panel-bg: #ffffff; --panel-border: #d0d5dd;
      --text: #1e293b; --text-muted: #64748b;
      --mode-border: #cbd5e1; --mode-bg: #f8fafc; --mode-text: #334155;
      --mode-active-border: #2980b9; --mode-active-bg: #dbeafe;
      --upload-border: #94a3b8; --upload-bg: #f8fafc; --file-name: #d97706;
      --btn-bg: #2980b9; --btn-disabled: #94a3b8;
      --log-bg: #f8fafc; --log-border: #cbd5e1; --log-text: #334155;
      --preview-bg: #f8fafc; --preview-border: #cbd5e1; --preview-text: #94a3b8;
      --progress-text: #475569; --bar-bg: #e2e8f0; --bar-border: #cbd5e1;
      --fill-start: #2980b9; --fill-end: #d97706;
      --hero-bg: #f8fafc; --hero-border: #cbd5e1; --hero-text: #94a3b8;
      --scan-color: #2980b9; --score-color: #2980b9; --verdict-color: #334155;
      --radar-bg: #f8fafc; --radar-border: #cbd5e1;
      --th-color: #475569; --td-color: #1e293b; --td-border: #e2e8f0;
      --pipe-bg: #ffffff; --pipe-border: #cbd5e1;
      --pipe-index: #d97706; --pipe-title: #1e293b; --pipe-desc: #64748b;
      --pipe-img-bg: #f8fafc; --pipe-img-border: #cbd5e1; --pipe-done: #16a34a;
      --pipe-label: #94a3b8; --accent-rgb: 41,128,185;
      --pipeline-g1: #f8fafc; --pipeline-g2: #f1f5f9;
      --model-inner: #f1f5f9; --model-outer: #e2e8f0; --model-border: #cbd5e1;
      --model-glow: rgba(41,128,185,.12); --model-inner-glow: rgba(41,128,185,.05);
      --model-ph: #94a3b8;
      --ctrl-bg: rgba(255,255,255,.85); --ctrl-border: #cbd5e1; --ctrl-text: #64748b;
      --ctrl-h-border: #2980b9; --ctrl-h-text: #1e293b;
      --ctrl-a-border: #2980b9; --ctrl-a-bg: rgba(41,128,185,.1); --ctrl-a-text: #2980b9;
      --hud-text: #334155; --hud-bg: rgba(255,255,255,.8); --hud-border: #cbd5e1;
      --badge-bg: rgba(248,250,252,.92); --clear-color: #e2e8f0;
      --theme-text: #1e293b; --theme-bg: #ffffff; --theme-border: #cbd5e1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      color: var(--text); background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
      transition: background .35s, color .35s;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 16px;
      padding: 18px;
    }
    .sidebar, .panel {
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      transition: background .35s, border-color .35s;
    }
    .sidebar { padding: 18px; display: flex; flex-direction: column; gap: 16px; }
    h1 { margin: 0; font-size: 25px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .muted { color: var(--text-muted); line-height: 1.55; font-size: 13px; }
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .mode {
      border: 1px solid var(--mode-border);
      background: var(--mode-bg);
      color: var(--mode-text);
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      text-align: center;
      font-weight: 700;
      transition: all .25s;
    }
    .mode.active { border-color: var(--mode-active-border); background: var(--mode-active-bg); color: white; }
    .species-wrap { display: none; margin-top: 8px; gap: 6px; }
    .species-wrap.visible { display: block; }
    .species-wrap label { font-size: 12px; font-weight: 700; color: var(--text-muted); display: block; margin-bottom: 2px; }
    .species-wrap select {
      width: 100%; padding: 7px 10px; border-radius: 7px;
      border: 1px solid var(--mode-border); background: var(--mode-bg); color: var(--mode-text);
      font-size: 13px; font-weight: 600; cursor: pointer; margin-bottom: 6px;
    }
    .dressing-wrap { margin-top: 6px; }
    .dressing-wrap label { font-size: 12px; font-weight: 700; color: var(--text-muted); display: block; margin-bottom: 2px; }
    .dressing-wrap select {
      width: 100%; padding: 7px 10px; border-radius: 7px;
      border: 1px solid var(--mode-border); background: var(--mode-bg); color: var(--mode-text);
      font-size: 13px; font-weight: 600; cursor: pointer;
    }
    .dressing-wrap select:disabled { opacity: 0.45; cursor: not-allowed; }
    .upload { display: grid; gap: 8px; }
    .upload label {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      border: 1px dashed var(--upload-border);
      border-radius: 8px;
      background: var(--upload-bg);
      color: var(--mode-text);
      cursor: pointer;
      font-weight: 700;
    }
    input[type="file"] { display: none; }
    .file-name { color: var(--file-name); font-size: 12px; overflow-wrap: anywhere; }
    button.primary {
      width: 100%;
      border: 0;
      border-radius: 8px;
      background: var(--btn-bg);
      color: white;
      font-size: 16px;
      font-weight: 800;
      padding: 14px 16px;
      cursor: pointer;
    }
    button.primary:disabled { background: var(--btn-disabled); cursor: wait; }
    .log {
      min-height: 170px;
      max-height: 230px;
      overflow: auto;
      padding: 10px;
      background: var(--log-bg);
      border: 1px solid var(--log-border);
      border-radius: 8px;
      color: var(--log-text);
      font: 12px/1.55 Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }
    .main { display: grid; grid-template-rows: auto 1fr auto; gap: 16px; min-width: 0; }
    .top {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .panel { padding: 14px; min-width: 0; }
    .preview {
      width: 100%;
      height: 260px;
      background: var(--preview-bg);
      border: 1px solid var(--preview-border);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      color: var(--preview-text);
      font-weight: 800;
    }
    .preview img { width: 100%; height: 100%; object-fit: contain; }
    /* 扫描动画 */
    .preview.scanning { position: relative; overflow: hidden; }
    .preview.scanning::before {
      content: ""; position: absolute; left: 0; right: 0;
      top: -40px; height: 40px;
      background: linear-gradient(180deg,
        transparent 0%,
        rgba(var(--accent-rgb),.04) 30%,
        rgba(var(--accent-rgb),.12) 60%,
        rgba(var(--accent-rgb),.18) 85%,
        var(--scan-color) 98%,
        var(--scan-color) 100%);
      box-shadow: 0 2px 30px rgba(var(--accent-rgb),.5), 0 2px 80px rgba(var(--accent-rgb),.2);
      animation: scanSweep 2.2s ease-in-out infinite;
      z-index: 10; pointer-events: none;
    }
    .preview.scanning::after {
      content: ""; position: absolute; inset: 0;
      background: repeating-linear-gradient(0deg,
        transparent 0px, transparent 2px,
        rgba(var(--accent-rgb),.015) 2px, rgba(var(--accent-rgb),.015) 3px);
      z-index: 5; pointer-events: none;
      animation: scanCRT 3s linear infinite;
    }
    @keyframes scanSweep {
      0%   { transform: translateY(0); opacity: 0; }
      5%   { opacity: 1; }
      90%  { opacity: 1; }
      100% { transform: translateY(calc(260px + 40px)); opacity: 0; }
    }
    @keyframes scanCRT {
      0%   { background-position: 0 0; }
      100% { background-position: 0 30px; }
    }
    .preview .scan-badge {
      position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
      background: var(--badge-bg); border: 1px solid var(--scan-color);
      border-radius: 4px; padding: 5px 16px;
      color: var(--scan-color); font-size: 11px; font-weight: 700;
      z-index: 11; display: none; letter-spacing: 2px;
      animation: badgePulse 1.2s ease infinite;
      text-transform: uppercase;
    }
    .preview.scanning .scan-badge { display: block; }
    @keyframes badgePulse {
      0%, 100% { box-shadow: 0 0 8px rgba(var(--accent-rgb),.3); opacity: .85; }
      50% { box-shadow: 0 0 22px rgba(var(--accent-rgb),.7); opacity: 1; }
    }
    .progress-wrap { display: grid; gap: 10px; }
    .progress-line { display: flex; justify-content: space-between; color: var(--progress-text); font-size: 13px; }
    .bar { height: 10px; background: var(--bar-bg); border-radius: 999px; border: 1px solid var(--bar-border); overflow: hidden; }
    .fill { width: 0%; height: 100%; background: linear-gradient(90deg, var(--fill-start), var(--fill-end)); transition: width .25s ease; }
    .result-grid {
      display: grid;
      grid-template-columns: 1.05fr .95fr;
      gap: 16px;
      min-height: 350px;
    }
    .hero-result {
      position: relative;
      height: 420px;
      background: var(--hero-bg);
      border: 1px solid var(--hero-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--hero-text);
      font-weight: 800;
    }
    .hero-result img { width: 100%; height: 100%; object-fit: contain; }
    .scan {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      left: 0;
      background: var(--scan-color);
      box-shadow: 0 0 18px var(--scan-color);
      animation: scan 1.6s linear infinite;
      display: none;
    }
    .running .scan { display: block; }
    @keyframes scan { from { left: 0; } to { left: 100%; } }
    .score {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 14px;
      align-items: center;
    }
    .score-number { font-size: 52px; color: var(--score-color); font-weight: 900; }
    .verdict { color: var(--verdict-color); line-height: 1.6; }
    .radar-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;
    }
    .radar-box { text-align: center; }
    .radar-box img { width: 100%; max-height: 280px; object-fit: contain; background: var(--radar-bg); border-radius: 8px; border: 1px solid var(--radar-border); }
    .radar-box .radar-label { color: var(--text-muted); font-size: 12px; margin-top: 6px; font-weight: 700; }
    .radar img { width: 100%; max-height: 300px; object-fit: contain; background: var(--radar-bg); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid var(--td-border); text-align: left; }
    th { color: var(--th-color); font-weight: 800; }
    td { color: var(--td-color); }

    .analysis-lab {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 16px;
    }
    .pipeline-window {
      position: relative;
      overflow: visible;
      background:
        radial-gradient(circle at 22% 18%, rgba(var(--accent-rgb), .16), transparent 24%),
        linear-gradient(135deg, var(--pipeline-g1), var(--pipeline-g2) 55%, var(--pipeline-g1));
      border: 1px solid var(--preview-border);
      border-radius: 8px;
    }
    .pipeline-track {
      position: relative;
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      grid-template-rows: repeat(2, 1fr);
      gap: 12px;
    }
    .pipe-step {
      position: relative;
      padding: 10px;
      border: 1px solid var(--pipe-border);
      border-radius: 8px;
      background: var(--pipe-bg);
      overflow: visible;
    }
    .pipe-step.active {
      border-color: var(--scan-color);
      box-shadow: inset 0 0 24px rgba(var(--accent-rgb), .18), 0 0 20px rgba(var(--accent-rgb), .10);
    }
    .pipe-index { color: var(--pipe-index); font-weight: 900; font-size: 12px; }
    .pipe-title { margin-top: 8px; font-weight: 900; color: var(--pipe-title); font-size: 14px; }
    .pipe-desc { margin-top: 8px; color: var(--pipe-desc); line-height: 1.45; font-size: 12px; }
    .pipe-step::after {
      content: "";
      position: absolute;
      left: -70%;
      top: 0;
      width: 55%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(var(--accent-rgb), .28), transparent);
      transform: skewX(-15deg);
    }
    .pipe-step.active::after { animation: sweep 1.2s ease infinite; }
    @keyframes sweep { from { left: -70%; } to { left: 125%; } }
    @keyframes pulseGlow {
      0%, 100% { box-shadow: inset 0 0 12px rgba(var(--accent-rgb),.1); }
      50% { box-shadow: inset 0 0 30px rgba(var(--accent-rgb),.28); }
    }
    .pipe-step .pipe-img {
      width: 100%; height: 180px; object-fit: contain;
      border-radius: 5px; margin-top: 6px; display: none;
      border: 1px solid var(--pipe-img-border); opacity: 0;
      transition: opacity .6s ease;
      background: var(--pipe-img-bg);
    }
    .pipe-step.has-img .pipe-img { display: block; opacity: 1; }
    .pipe-step.has-img .pipe-desc { display: none; }
    .pipe-step.done { border-color: var(--pipe-done); }
    .pipe-step.done::before {
      content: "✓"; position: absolute; top: 6px; right: 10px;
      color: var(--pipe-done); font-weight: 900; font-size: 16px;
    }
    .pipe-label {
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      color: var(--pipe-label); font-weight: 800; font-size: 14px;
    }
    .pipeline-window { height: auto; min-height: 420px; }
    .model-wrap {
      position: relative; height: 420px;
      background: radial-gradient(ellipse at center, var(--model-inner) 0%, var(--model-outer) 100%);
      border: 1px solid var(--model-border); border-radius: 8px; overflow: hidden;
      transition: box-shadow .5s ease;
    }
    .model-wrap.model-active {
      box-shadow: 0 0 30px var(--model-glow), inset 0 0 20px var(--model-inner-glow);
    }
    .model-3d-container {
      width: 100%; height: 100%; display: flex;
      align-items: center; justify-content: center;
    }
    .model-3d-container canvas { display: block; width: 100% !important; height: 100% !important; }
    .model-placeholder { color: var(--model-ph); font-weight: 800; font-size: 14px; }
    #model3d { width: 100%; height: 100%; display: block; }
    .model-controls {
      position: absolute; bottom: 12px; right: 12px;
      display: flex; gap: 6px; z-index: 10;
    }
    .ctrl-btn {
      border: 1px solid var(--ctrl-border); background: var(--ctrl-bg);
      color: var(--ctrl-text); border-radius: 6px; padding: 5px 10px;
      font-size: 11px; cursor: pointer; font-weight: 700;
      transition: all .2s ease;
    }
    .ctrl-btn:hover { border-color: var(--ctrl-h-border); color: var(--ctrl-h-text); }
    .ctrl-btn.active { border-color: var(--ctrl-a-border); background: var(--ctrl-a-bg); color: var(--ctrl-a-text); }
    .model-hud {
      position: absolute;
      left: 12px;
      top: 12px;
      right: 12px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--hud-text);
      font-size: 12px;
      pointer-events: none;
    }
    .hud-chip {
      border: 1px solid var(--hud-border);
      background: var(--hud-bg);
      border-radius: 8px;
      padding: 7px 9px;
    }
    /* 主题切换按钮 */
    .theme-toggle {
      flex-shrink: 0;
      border: 1px solid var(--theme-border);
      background: var(--theme-bg);
      color: var(--theme-text);
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
      transition: all .25s;
    }
    .theme-toggle:hover { border-color: var(--mode-active-border); }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .top, .result-grid, .analysis-lab { grid-template-columns: 1fr; }

    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-header-text">
          <h1>Skin Recovery Assessor</h1>
          <div class="muted">术前/术后图像配准 · 多维恢复评分 · 动态热力分析</div>
        </div>
        <button id="themeToggle" class="theme-toggle" type="button" onclick="toggleTheme()">☀ 白色主题</button>
      </div>

      <section>
        <h2>分析对象</h2>
        <div class="mode-grid">
          <button class="mode active" data-mode="face" type="button">人体皮肤分析</button>
          <button class="mode" data-mode="animal" type="button">动物皮肤分析</button>
        </div>
        <p id="modeHint" class="muted">人体皮肤模式：自动检测皮肤区域，支持面部或人体局部组织图像。</p>
        <div id="speciesWrap" class="species-wrap">
          <label for="speciesSelect">动物物种</label>
          <select id="speciesSelect">
            <option value="rabbit">兔子</option>
            <option value="dog">狗</option>
            <option value="pig">猪</option>
            <option value="mouse">老鼠</option>
          </select>
          <div id="strainWrap" style="display:none">
            <label for="strainSelect">老鼠品系</label>
            <select id="strainSelect">
              <option value="SD大鼠">SD大鼠</option>
              <option value="Wistar大鼠">Wistar大鼠</option>
              <option value="Zucker大鼠">Zucker大鼠</option>
              <option value="BALB/c裸鼠">BALB/c裸鼠</option>
              <option value="C57BL/6小鼠">C57BL/6小鼠</option>
              <option value="SKH-1无毛小鼠">SKH-1无毛小鼠</option>
            </select>
          </div>
        </div>
      </section>

      <section class="upload">
        <h2>选择图像</h2>
        <label for="preFile">选择术前照片</label>
        <input id="preFile" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp,image/heif,image/heic" />
        <div id="preName" class="file-name">术前：未选择</div>
        <label for="postFile">选择术后照片</label>
        <input id="postFile" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp,image/heif,image/heic" />
        <div id="postName" class="file-name">术后：未选择</div>
        <div class="dressing-wrap">
          <label for="dressingTime">敷料使用时间（可选）</label>
          <select id="dressingTime" disabled>
            <option value="">请选择（可选）</option>
            <option value="5min">5 min</option>
            <option value="10min">10 min</option>
            <option value="15min">15 min</option>
            <option value="30min">30 min</option>
            <option value="1h">1 h</option>
            <option value="5h">5 h</option>
            <option value="10h">10 h</option>
          </select>
        </div>
      </section>

      <button id="analyzeBtn" class="primary" type="button">开始进行分析</button>

      <section>
        <h2>运行日志</h2>
        <div id="log" class="log">系统已启动，请选择术前/术后照片。</div>
      </section>
    </aside>

    <main class="main">
      <div class="top">
        <section class="panel">
          <h2>术前图像</h2>
          <div id="prePreview" class="preview">未选择<div class="scan-badge">SCANNING...</div></div>
        </section>
        <section class="panel">
          <h2>术后图像</h2>
          <div id="postPreview" class="preview">未选择<div class="scan-badge">SCANNING...</div></div>
        </section>
      </div>

      <section class="panel progress-wrap">
        <div class="progress-line">
          <strong>实时分析流水线</strong>
          <span id="status">等待输入图像</span>
        </div>
        <div class="bar"><div id="fill" class="fill"></div></div>
      </section>

      <div class="analysis-lab">
        <section class="panel">
          <h2>同页实时图像变换窗口</h2>
          <div class="pipeline-window">
            <div id="pipelineTrack" class="pipeline-track">
              <div class="pipe-label">等待分析 — 选择图像后点击开始</div>
            </div>
          </div>
        </section>
        <section class="panel">
          <h2>3D 皮肤恢复仿真建模</h2>
          <div id="modelWrap" class="model-wrap">
            <div id="model3dContainer" class="model-3d-container">
              <span class="model-placeholder">分析后生成 3D 仿真模型</span>
            </div>
            <div class="model-hud">
              <div class="hud-chip" id="modelMode">模型：待机</div>
              <div class="hud-chip" id="modelScore">恢复势能：--</div>
            </div>
            <div class="model-controls">
              <button class="ctrl-btn active" id="btnSolid" type="button">点云</button>
              <button class="ctrl-btn" id="btnWire" type="button">线框</button>
              <button class="ctrl-btn" id="btnHeat" type="button">融合</button>
            </div>
          </div>
        </section>
      </div>

      <div class="result-grid">
        <section class="panel">
          <h2>综合异常变化热力图</h2>
          <div id="hero" class="hero-result">
            <span id="heroEmpty">等待分析</span>
            <div class="scan"></div>
          </div>
        </section>
        <section class="panel">
          <h2>评分结果</h2>
          <div class="score">
            <div id="score" class="score-number">--</div>
            <div id="verdict" class="verdict">选择两张图片后开始分析。</div>
          </div>
          <div id="radar" class="radar"></div>
        </section>
      </div>

      <section class="panel">
        <h2>各维度评价</h2>
        <table>
          <thead><tr><th>维度</th><th>术前指数</th><th>术后指数</th><th>恢复分</th><th>改善量</th></tr></thead>
          <tbody id="metricRows"><tr><td colspan="5" class="muted">暂无数据</td></tr></tbody>
        </table>
      </section>


    </main>
  </div>

  <script>
    function toggleTheme() {
      const isLight = document.body.classList.toggle("light");
      document.getElementById("themeToggle").textContent = isLight ? "\u{1F319} 深色主题" : "\u2600 白色主题";
      // 更新 3D 渲染器背景色
      if (state.threeObj && state.threeObj.renderer) {
        const cs = getComputedStyle(document.documentElement);
        const clearColor = cs.getPropertyValue("--clear-color").trim();
        state.threeObj.renderer.setClearColor(parseInt(clearColor.replace("#", "0x")));
      }
    }
    const state = {
      mode: "face", pre: null, post: null, timer: null,
      progress: 0, activeStep: 0, modelScore: null, lastPayload: null,
    };
    const $ = (id) => document.getElementById(id);
    const pipelineSteps = [
      ["01", "红斑变化热力图", "比较术前术后红斑强度差异，定位改善或恶化区域。"],
      ["02", "肿胀程度热力图", "用局部亮度平滑检测肿胀区域分布与强度。"],
      ["03", "渗出结痂热力图", "检测黄色渗出与棕褐结痂区域的空间分布。"],
      ["04", "干燥脱屑热力图", "检测高亮低饱和鳞屑与局部高频纹理区域。"],
      ["05", "色素肤质热力图", "分析暗沉斑块、亮度不均与纹理粗糙度。"],
      ["06", "针孔闭合度热力图", "用形态学 black-hat 提取小型暗圆斑/孔洞分布。"]
    ];

    function initExplanation() {
      $("pipelineTrack").innerHTML = pipelineSteps.map((s, i) => `
        <div class="pipe-step ${i === 0 ? "active" : ""}" data-step="${i}">
          <div class="pipe-index">${s[0]}</div>
          <div class="pipe-title">${s[1]}</div>
          <div class="pipe-desc">${s[2]}</div>
          <img class="pipe-img" src="" alt="" />
        </div>
      `).join("");
    }

    function setActiveStep(index) {
      state.activeStep = index;
      document.querySelectorAll(".pipe-step").forEach((item) => {
        const step = Number(item.dataset.step);
        item.classList.toggle("active", step === index);
        if (step < index) item.classList.add("done");
      });
    }

    function log(message) {
      const now = new Date().toLocaleTimeString();
      $("log").textContent += `\n[${now}] ${message}`;
      $("log").scrollTop = $("log").scrollHeight;
    }

    function setProgress(value, text) {
      state.progress = Math.max(0, Math.min(100, value));
      $("fill").style.width = `${state.progress}%`;
      $("status").textContent = text;
    }

    function preview(file, targetId) {
      const reader = new FileReader();
      reader.onload = () => {
        $(targetId).innerHTML = `<img src="${reader.result}" alt="preview" />`;
      };
      reader.readAsDataURL(file);
    }

    document.querySelectorAll(".mode").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mode").forEach((item) => item.classList.remove("active"));
        btn.classList.add("active");
        state.mode = btn.dataset.mode;
        const isAnimal = state.mode === "animal";
        const text = isAnimal
          ? "动物皮肤模式：使用更宽松的表面/纹理 ROI，适合宠物、实验动物或局部皮肤照片。"
          : "人体皮肤模式：自动检测皮肤区域，支持面部或人体局部组织图像。";
        $("modeHint").textContent = text;
        $("speciesWrap").classList.toggle("visible", isAnimal);
        if (isAnimal) { updateStrainVisibility(); }
        log(`切换分析对象：${btn.textContent}`);
      });
    });

    $("speciesSelect").addEventListener("change", updateStrainVisibility);
    function updateStrainVisibility() {
      const isMouse = $("speciesSelect").value === "mouse";
      $("strainWrap").style.display = isMouse ? "block" : "none";
    }

    $("preFile").addEventListener("change", (event) => {
      state.pre = event.target.files[0];
      $("preName").textContent = state.pre ? `术前：${state.pre.name}` : "术前：未选择";
      if (state.pre) { preview(state.pre, "prePreview"); log(`已选择术前照片：${state.pre.name}`); }
    });

    $("postFile").addEventListener("change", (event) => {
      state.post = event.target.files[0];
      $("postName").textContent = state.post ? `术后：${state.post.name}` : "术后：未选择";
      if (state.post) {
        preview(state.post, "postPreview");
        log(`已选择术后照片：${state.post.name}`);
        $("dressingTime").disabled = false;
      }
    });

    function beginFakeProgress() {
      const steps = ["读取图像", "特征配准", "ROI 分割", "热力建模", "评分融合", "渲染结果"];
      let idx = 0;
      setProgress(6, "分析启动中");
      setActiveStep(0);
      state.modelScore = null;
      $("modelMode").textContent = "模型：重建中";
      $("modelScore").textContent = "恢复势能：计算中";
      $("modelWrap").classList.add("model-active");
      document.querySelectorAll(".pipe-step").forEach(el => { el.classList.remove("done", "has-img"); el.querySelector(".pipe-img").src = ""; });
      $("hero").classList.add("running");
      state.timer = setInterval(() => {
        idx = Math.min(idx + 1, steps.length - 1);
        setActiveStep(Math.min(idx, pipelineSteps.length - 1));
        setProgress(Math.min(88, 8 + idx * 14), steps[idx]);
        log(steps[idx]);
      }, 2200);
    }

    function stopFakeProgress() {
      if (state.timer) clearInterval(state.timer);
      state.timer = null;
      $("hero").classList.remove("running");
    }

    $("analyzeBtn").addEventListener("click", async () => {
      log("点击了开始进行分析按钮。");
      if (!state.pre || !state.post) {
        log("无法开始：请先选择术前照片和术后照片。");
        alert("请先选择术前照片和术后照片。");
        return;
      }
      $("analyzeBtn").disabled = true;
      $("prePreview").classList.add("scanning");
      $("postPreview").classList.add("scanning");
      $("score").textContent = "--";
      $("verdict").textContent = "正在分析，请稍候...";
      $("metricRows").innerHTML = `<tr><td colspan="5" class="muted">分析中...</td></tr>`;
      $("hero").innerHTML = `<span id="heroEmpty">分析中</span><div class="scan"></div>`;
      beginFakeProgress();
      const analysisStart = Date.now();
      const MIN_ANALYSIS_MS = 20000;
      try {
        const form = new FormData();
        form.append("mode", state.mode);
        form.append("pre", state.pre);
        form.append("post", state.post);
        // 动物物种/品系参数
        if (state.mode === "animal") {
          form.append("animal_species", $("speciesSelect").value);
          if ($("speciesSelect").value === "mouse") {
            form.append("mouse_strain", $("strainSelect").value);
          }
        }
        // 敷料时间参数
        const dt = $("dressingTime").value;
        if (dt) { form.append("dressing_time", dt); }
        log("正在AI智能诊断...");
        const response = await fetch("/api/analyze", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok || !payload.ok) throw new Error(payload.error || "分析失败");
        const elapsed = Date.now() - analysisStart;
        if (elapsed < MIN_ANALYSIS_MS) {
          await new Promise(r => setTimeout(r, MIN_ANALYSIS_MS - elapsed));
        }
        stopFakeProgress();
        $("prePreview").classList.remove("scanning");
        $("postPreview").classList.remove("scanning");
        setProgress(100, "分析完成");
        state.lastPayload = payload;
        state.modelScore = payload.overall_score;
        const frInfo = payload.face_region || { type: "full" };
        const isProfile = frInfo.type === "profile";
        const isPartial = frInfo.type === "partial";
        $("modelMode").textContent = `模型：${payload.mode_label}${isProfile ? " · 侧脸" : isPartial ? " · 局部" : ""}`;
        $("modelScore").textContent = `恢复势能：${payload.overall_score.toFixed(1)}`;
        log(`分析完成，综合恢复评分：${payload.overall_score.toFixed(1)}`);
        $("score").textContent = payload.overall_score.toFixed(1);
        $("verdict").textContent = payload.verdict;
        if (payload.comparison_radar) {
          $("radar").innerHTML = `<img src="${payload.comparison_radar}" alt="术前术后六维对比" />`;
        }
        $("hero").innerHTML = `<img src="${payload.overlay_image}" alt="综合异常变化热力图" /><div class="scan"></div>`;
        $("metricRows").innerHTML = payload.metrics.map(m => `
          <tr><td>${m.name}</td><td>${m.pre_index.toFixed(1)}</td><td>${m.post_index.toFixed(1)}</td>
          <td>${m.score.toFixed(1)}</td><td>+${(m.post_index - m.pre_index).toFixed(1)}</td></tr>
        `).join("");
        // === 实时图像变换：逐步展示过程图 ===
        const steps = document.querySelectorAll(".pipe-step");
        const hmKeys = ["red", "swelling", "exudate", "dryness", "pigmentation", "pinhole"];
        for (let i = 0; i < steps.length && i < payload.stages.length; i++) {
          await new Promise(r => setTimeout(r, 2000));
          const imgEl = steps[i].querySelector(".pipe-img");
          imgEl.src = payload.stages[i].image;
          steps[i].classList.add("has-img", "done");
          setActiveStep(i);
        }
        setActiveStep(steps.length - 1);
        // === 初始化 3D 仿真模型 ===
        initThreeModel(payload);
        // 显示模式切换
        document.querySelectorAll(".ctrl-btn").forEach(btn => {
          btn.onclick = () => {
            document.querySelectorAll(".ctrl-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            if (state.threeObj) {
              const mode = btn.id.replace("btn", "").toLowerCase();
              state.threeObj.pointCloud.visible = (mode === "solid" || mode === "heat");
              state.threeObj.wireMesh.visible = (mode === "wire" || mode === "heat");
            }
          };
        });
      } catch (error) {
        stopFakeProgress();
        $("prePreview").classList.remove("scanning");
        $("postPreview").classList.remove("scanning");
        setProgress(0, "分析失败");
        $("verdict").textContent = `分析失败：${error.message}`;
        log(`分析失败：${error.message}`);
        alert(`分析失败：${error.message}`);
      } finally {
        $("analyzeBtn").disabled = false;
      }
    });

    /* ======== Three.js 3D 皮肤恢复仿真建模 - 原图叠加网格+点云 ======== */
    function initThreeModel(payload) {
      const container = $("model3dContainer");
      container.innerHTML = "";
      $("modelWrap").classList.add("model-active");
      const W = container.clientWidth || 400, H = container.clientHeight || 380;

      // ---- 场景 / 相机 / 渲染器 ----
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
      camera.position.set(0, 0.15, 5.2);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      const cs = getComputedStyle(document.documentElement);
      const clearColor = cs.getPropertyValue("--clear-color").trim();
      renderer.setClearColor(parseInt(clearColor.replace("#", "0x")));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.15;
      container.appendChild(renderer.domElement);

      // ---- 灯光 ----
      scene.add(new THREE.AmbientLight(0xffffff, 0.85));
      const keyLight = new THREE.DirectionalLight(0xffeedd, 0.65);
      keyLight.position.set(3, 4, 5); scene.add(keyLight);
      const fillLight = new THREE.DirectionalLight(0x8899bb, 0.25);
      fillLight.position.set(-3, 2, 3); scene.add(fillLight);
      const rimLight = new THREE.PointLight(0x37d5ff, 0.5, 15);
      rimLight.position.set(0, 0.5, -3.5); scene.add(rimLight);

      const modelGroup = new THREE.Group();
      let texturedMesh = null;
      let wireLines = null;
      let pointCloud = null;

      // ---- 推断网格尺寸 ----
      const hasGrid = payload.grid_vertices && payload.grid_vertices.length > 0;
      // 从 grid_vertices 数量反推 rows/cols
      const totalVerts = hasGrid ? payload.grid_vertices.length : 0;
      const gridSide = Math.round(Math.sqrt(totalVerts));  // 100
      const gRows = gridSide, gCols = gridSide;

      if (hasGrid && gRows > 1 && gCols > 1) {
        // 计算网格边界用于 UV 映射
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < totalVerts; i++) {
          const v = payload.grid_vertices[i];
          if (v[0] < minX) minX = v[0]; if (v[0] > maxX) maxX = v[0];
          if (v[1] < minY) minY = v[1]; if (v[1] > maxY) maxY = v[1];
        }
        const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;

        // ---- 构建纹理化网格面 (原图贴在平面上) ----
        const meshGeo = new THREE.BufferGeometry();
        const posArr = new Float32Array(totalVerts * 3);
        const uvArr = new Float32Array(totalVerts * 2);
        for (let i = 0; i < totalVerts; i++) {
          const v = payload.grid_vertices[i];
          posArr[i*3]   = v[0];
          posArr[i*3+1] = v[1];
          posArr[i*3+2] = v[2];
          // UV: col → u (0..1), row → v (1..0 因为图像 y 轴向下)
          const col = i % gCols, row = Math.floor(i / gCols);
          uvArr[i*2]   = col / (gCols - 1);
          uvArr[i*2+1] = 1.0 - row / (gRows - 1);
        }
        meshGeo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        meshGeo.setAttribute("uv", new THREE.BufferAttribute(uvArr, 2));

        // 生成三角形索引
        const idxArr = [];
        for (let r = 0; r < gRows - 1; r++) {
          for (let c = 0; c < gCols - 1; c++) {
            const i = r * gCols + c;
            idxArr.push(i, i + 1, i + gCols);
            idxArr.push(i + 1, i + gCols + 1, i + gCols);
          }
        }
        meshGeo.setIndex(idxArr);
        meshGeo.computeVertexNormals();

        // 加载纹理贴图
        const texLoader = new THREE.TextureLoader();
        const texture = texLoader.load(payload.grid_texture || "");
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;

        const meshMat = new THREE.MeshStandardMaterial({
          map: texture, side: THREE.DoubleSide,
          roughness: 0.75, metalness: 0.05,
        });
        texturedMesh = new THREE.Mesh(meshGeo, meshMat);
        modelGroup.add(texturedMesh);

        // ---- 叠加网格线框 (略微前移避免 z-fighting) ----
        if (payload.wire_indices && payload.wire_indices.length > 0) {
          const wireGeo = new THREE.BufferGeometry();
          const wLen = payload.wire_indices.length;
          const wirePos = new Float32Array(wLen * 3);
          for (let i = 0; i < wLen; i++) {
            const idx = payload.wire_indices[i];
            wirePos[i*3]   = payload.grid_vertices[idx][0];
            wirePos[i*3+1] = payload.grid_vertices[idx][1];
            wirePos[i*3+2] = payload.grid_vertices[idx][2] + 0.005;
          }
          wireGeo.setAttribute("position", new THREE.BufferAttribute(wirePos, 3));
          wireLines = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({
            color: 0x37d5ff, transparent: true, opacity: 0.22,
          }));
          wireLines.visible = true;
          modelGroup.add(wireLines);
        }

        // ---- 网格顶点小点云（可选，增加立体感）----
        const dotGeo = new THREE.BufferGeometry();
        const dotPos = new Float32Array(totalVerts * 3);
        for (let i = 0; i < totalVerts; i++) {
          dotPos[i*3]   = payload.grid_vertices[i][0];
          dotPos[i*3+1] = payload.grid_vertices[i][1];
          dotPos[i*3+2] = payload.grid_vertices[i][2] + 0.008;
        }
        dotGeo.setAttribute("position", new THREE.BufferAttribute(dotPos, 3));
        pointCloud = new THREE.Points(dotGeo, new THREE.PointsMaterial({
          size: 0.018, color: 0x37d5ff,
          sizeAttenuation: true, transparent: true, opacity: 0.35,
        }));
        pointCloud.visible = false;
        modelGroup.add(pointCloud);

      } else {
        // 回退
        texturedMesh = new THREE.Mesh(); modelGroup.add(texturedMesh);
        wireLines = new THREE.LineSegments(); wireLines.visible = false; modelGroup.add(wireLines);
        pointCloud = new THREE.Points(); modelGroup.add(pointCloud);
      }

      // ---- 高频特征点标注（醒目颜色 + 光晕）----
      if (payload.feature_points && payload.feature_points.positions && payload.feature_points.positions.length > 0) {
        const fpGeo = new THREE.BufferGeometry();
        const fpPos = new Float32Array(payload.feature_points.positions.flat());
        fpGeo.setAttribute("position", new THREE.BufferAttribute(fpPos, 3));
        if (payload.feature_points.colors && payload.feature_points.colors.length > 0) {
          const fpCol = new Float32Array(payload.feature_points.colors.flat());
          fpGeo.setAttribute("color", new THREE.BufferAttribute(fpCol, 3));
        }
        const featureCloud = new THREE.Points(fpGeo, new THREE.PointsMaterial({
          size: 0.10, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 1.0,
        }));
        modelGroup.add(featureCloud);
        // 光晕层
        const glowGeo = new THREE.BufferGeometry();
        const glowPos = new Float32Array(fpPos.length);
        for (let i = 0; i < fpPos.length; i += 3) {
          glowPos[i] = fpPos[i]; glowPos[i+1] = fpPos[i+1]; glowPos[i+2] = fpPos[i+2] + 0.01;
        }
        glowGeo.setAttribute("position", new THREE.BufferAttribute(glowPos, 3));
        if (payload.feature_points.colors && payload.feature_points.colors.length > 0) {
          const glowCol = new Float32Array(payload.feature_points.colors.flat());
          glowGeo.setAttribute("color", new THREE.BufferAttribute(glowCol, 3));
        }
        modelGroup.add(new THREE.Points(glowGeo, new THREE.PointsMaterial({
          size: 0.20, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 0.30,
        })));
      }

      scene.add(modelGroup);

      // ---- 粒子系统 ----
      const pCount = 500;
      const pGeo = new THREE.BufferGeometry();
      const pPos = new Float32Array(pCount * 3), pCol = new Float32Array(pCount * 3);
      for (let i = 0; i < pCount; i++) {
        const a = Math.random() * Math.PI * 2, rad = 2.8 + Math.random() * 1.8;
        pPos[i*3] = Math.cos(a) * rad;
        pPos[i*3+1] = (Math.random() - 0.5) * 3;
        pPos[i*3+2] = Math.sin(a) * rad - 1.0;
        const gold = Math.random() > 0.55;
        pCol[i*3] = gold ? 0.97 : 0.22; pCol[i*3+1] = gold ? 0.82 : 0.84; pCol[i*3+2] = gold ? 0.33 : 1.0;
      }
      pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
      pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
      const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
        size: 0.018, vertexColors: true, transparent: true, opacity: 0.4,
      }));
      scene.add(particles);

      // ---- 鼠标拖拽旋转（±45度）----
      const MAX_ROT = Math.PI / 4;
      let dragging = false, prevMX = 0, prevMY = 0, rotY = 0, rotX = 0;
      renderer.domElement.addEventListener("pointerdown", e => { dragging = true; prevMX = e.clientX; prevMY = e.clientY; renderer.domElement.style.cursor = "grabbing"; });
      renderer.domElement.addEventListener("pointermove", e => {
        if (!dragging) return;
        rotY += (e.clientX - prevMX) * 0.005;
        rotX += (e.clientY - prevMY) * 0.004;
        rotY = Math.max(-MAX_ROT, Math.min(MAX_ROT, rotY));
        rotX = Math.max(-MAX_ROT, Math.min(MAX_ROT, rotX));
        prevMX = e.clientX; prevMY = e.clientY;
      });
      const stopDrag = () => { dragging = false; renderer.domElement.style.cursor = "grab"; };
      renderer.domElement.addEventListener("pointerup", stopDrag);
      renderer.domElement.addEventListener("pointerleave", stopDrag);
      renderer.domElement.style.cursor = "grab";

      // ---- 动画循环 ----
      let frame = 0;
      function animate() {
        requestAnimationFrame(animate);
        frame++;
        const t = frame * 0.01;
        if (!dragging) {
          rotY = MAX_ROT * Math.sin(t * 0.35);
          rotX = MAX_ROT * 0.25 * Math.sin(t * 0.25);
        }
        modelGroup.rotation.y += (rotY - modelGroup.rotation.y) * 0.08;
        modelGroup.rotation.x += (rotX - modelGroup.rotation.x) * 0.08;
        // 网格点呼吸
        if (pointCloud && pointCloud.material) {
          pointCloud.material.opacity = 0.25 + Math.sin(t * 1.5) * 0.10;
        }
        // 粒子旋转
        particles.rotation.y = t * 0.08;
        particles.rotation.x = Math.sin(t * 0.2) * 0.03;
        rimLight.intensity = 0.45 + Math.sin(t * 2) * 0.12;
        renderer.render(scene, camera);
      }
      animate();
      state.threeObj = { scene, camera, renderer, pointCloud: pointCloud, wireMesh: wireLines };
      // 窗口 resize
      const ro = new ResizeObserver(() => {
        const w = container.clientWidth, h = container.clientHeight;
        if (w > 0 && h > 0) {
          camera.aspect = w / h; camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      });
      ro.observe(container);
    }

    initExplanation();
  </script>
</body>
</html>
"""


def _parse_multipart(body: bytes, boundary: bytes) -> dict:
    """Parse multipart/form-data body without cgi module (Python 3.13+ compatible)."""
    fields: dict = {}
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    for part in parts[1:]:
        if part.startswith(b"--"):
            break
        if not part.strip():
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4 :]
        if content.endswith(b"\r\n"):
            content = content[:-2]
        name_match = re.search(r'name="([^"]+)"', header_block)
        filename_match = re.search(r'filename="([^"]*?)"', header_block)
        if not name_match:
            continue
        name = name_match.group(1)
        if filename_match:
            class _F:
                def __init__(self, data: bytes):
                    self.file = BytesIO(data)
            fields[name] = _F(content)
        else:
            class _V:
                def __init__(self, val: str):
                    self.value = val
                def getfirst(self, default=None):
                    return self.value
            fields[name] = _V(content.decode("utf-8", errors="replace"))
    return fields


def image_from_upload(field: Any) -> np.ndarray:
    """解析上传图像，兼容所有手机/相机设备拍摄的图像。
    支持：HEIF/HEIC (iPhone)、Display P3 色彩空间、
    EXIF 旋转、RGBA/CMYK/16bit、损坏文件容错。
    """
    data = field.file.read()
    if not data:
        raise ValueError("上传图片为空")
    try:
        image = Image.open(BytesIO(data))
        image.load()  # 强制加载像素，触发截断/损坏检测
    except Exception:
        # 回退：OpenCV 解码 (处理损坏或非标准文件)
        arr = np.frombuffer(data, dtype=np.uint8)
        raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw is not None:
            return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raise ValueError("无法解码上传的图像文件")

    # EXIF 方向修正
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # 应用 ICC 色彩配置文件 (处理 Display P3 / Adobe RGB 等广色域)
    try:
        icc_profile = image.info.get("icc_profile")
        if icc_profile:
            from PIL import ImageCms
            src_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            dst_profile = ImageCms.createProfile("sRGB")
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            image = ImageCms.profileToProfile(
                image, src_profile, dst_profile, outputMode="RGB"
            )
    except Exception:
        pass

    # 模式转换 -> RGB
    if image.mode == "RGBA" or image.mode == "LA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "LA":
            image = image.convert("RGBA")
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode == "P":
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode == "CMYK":
        image = image.convert("RGB")
    elif image.mode == "I;16" or image.mode == "I":
        arr = np.array(image, dtype=np.float64)
        if arr.max() > 255:
            arr = arr / arr.max() * 255.0
        image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    elif image.mode not in ("RGB",):
        image = image.convert("RGB")

    arr = np.array(image)

    # 确保 uint8 输出
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # 确保 3 通道
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    return arr


def image_to_data_uri(rgb: np.ndarray, max_side: int = 950) -> str:
    img = rgb
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise ValueError("图像编码失败")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + data


def build_3d_image_points(ref_img: np.ndarray, rows: int = 100, cols: int = 100,
                          n_features: int = 120) -> dict:
    """基于输入图像的网格化点云生成。

    1. 将图像划分为规则网格，每个网格点作为一个点云顶点
    2. 利用 Laplacian 算子检测高频区域，用亮度信息计算深度
    3. 从高频区域随机选取特征点，用醒目颜色标识
    """
    ih, iw = ref_img.shape[:2]
    # 归一化坐标 -1..1
    u = np.linspace(-1, 1, cols)
    v = np.linspace(-1, 1, rows)
    uu, vv = np.meshgrid(u, v)

    # 图像缩放到网格尺寸，用于采样颜色和亮度
    small = cv2.resize(ref_img, (cols, rows), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 高频检测：Laplacian 响应
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    lap_abs = np.abs(laplacian)

    # Sobel 边缘（补充高频信息）
    sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 综合高频响应
    hf_response = 0.6 * lap_abs + 0.4 * edge_mag
    hf_norm = hf_response / max(hf_response.max(), 1e-6)

    # 深度计算：基于亮度 + 高频响应
    gray_smooth = cv2.GaussianBlur(gray, (7, 7), 1.2)
    depth = (gray_smooth - 0.5) * 0.15 + hf_norm * 0.08
    # 轻微平滑让深度更自然
    depth = cv2.GaussianBlur(depth, (3, 3), 0.5)

    # 根据图像宽高比调整显示范围
    aspect = iw / max(ih, 1)
    if aspect > 1:
        hw, hh = 2.0, 2.0 / aspect
    else:
        hw, hh = 2.0 * aspect, 2.0
    xx = uu * hw
    yy = -vv * hh  # 翻转 y 轴

    # 构建点云
    positions = np.column_stack([xx.ravel(), yy.ravel(), depth.ravel()])
    colors = (small.reshape(-1, 3).astype(np.float32) / 255.0).clip(0, 1)

    # ---- 高频特征点选取 ----
    # 找高频响应较高的网格点
    hf_flat = hf_norm.ravel()
    threshold = np.percentile(hf_flat, 85)  # 取前 15% 高频点
    high_freq_indices = np.where(hf_flat > threshold)[0]

    # 从高频点中随机选取 n_features 个
    if len(high_freq_indices) > n_features:
        rng = np.random.default_rng(42)
        selected = rng.choice(high_freq_indices, size=n_features, replace=False)
    else:
        selected = high_freq_indices

    # 特征点颜色：根据高频响应强度使用不同醒目颜色
    feature_positions = []
    feature_colors = []
    for idx in selected:
        pos = positions[idx]
        feature_positions.append(pos.tolist())
        # 高频强度映射到颜色：弱高频=青色，中高频=黄色，强高频=红色
        strength = hf_flat[idx]
        if strength > 0.7:
            fc = [1.0, 0.15, 0.15]   # 红 - 强高频
        elif strength > 0.4:
            fc = [1.0, 0.85, 0.15]   # 黄 - 中高频
        else:
            fc = [0.15, 0.85, 1.0]   # 青 - 弱高频
        feature_colors.append(fc)

    # 网格线框数据（用于融合模式显示）
    wire_positions = positions.tolist()
    wire_indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            wire_indices.extend([i, i + 1, i, i + cols])

    logging.info("[3D PointCloud] 网格 %dx%d=%d 点, 高频特征点 %d 个",
                 rows, cols, len(positions), len(feature_positions))

    return {
        "positions": positions.tolist(),
        "colors": colors.tolist(),
        "wire_indices": wire_indices,
        "feature_points": {
            "positions": feature_positions,
            "colors": feature_colors,
        },
    }


def make_payload(pre_img: np.ndarray, post_img: np.ndarray, mode: str,
                 animal_species: str = "", mouse_strain: str = "",
                 dressing_time: str = "") -> Dict[str, Any]:
    result = analyze(pre_img, post_img, mode=mode,
                     animal_species=animal_species, mouse_strain=mouse_strain,
                     dressing_time=dressing_time)
    # 构建基于图像的三维网格点云
    try:
        point_data = build_3d_image_points(ref_img=result.aligned_post,
                                           rows=100, cols=100, n_features=120)
        grid_verts = point_data["positions"]
        grid_colors = point_data["colors"]
        wire_indices = point_data["wire_indices"]
        feat_pts = point_data["feature_points"]
    except Exception as exc:
        logging.warning("3D point cloud build failed: %s", exc)
        grid_verts = []
        grid_colors = []
        wire_indices = []
        feat_pts = {"positions": [], "colors": []}
    return {
        "ok": True,
        "mode_label": result.mode_label,
        "overall_score": result.overall_score,
        "verdict": result.verdict,
        "overlay_image": image_to_data_uri(result.overlay_image),
        "radar_image": image_to_data_uri(result.radar_image),
        "comparison_radar": image_to_data_uri(result.comparison_radar) if result.comparison_radar is not None else "",
        "grid_texture": image_to_data_uri(result.aligned_post, max_side=512),
        "grid_vertices": grid_verts,
        "grid_colors": grid_colors,
        "wire_indices": wire_indices,
        "feature_points": feat_pts,
        "heatmaps": {
            name: image_to_data_uri(colorize_heatmap(hm), max_side=320)
            for name, hm in result.heatmaps.items()
        },
        "face_region": result.face_region,
        "metrics": [
            {
                "name": item.name,
                "pre_value": item.pre_value,
                "post_value": item.post_value,
                "pre_index": item.pre_index,
                "post_index": item.post_index,
                "score": item.score,
                "delta": item.delta,
                "note": item.note,
            }
            for item in result.metrics
        ],
        "stages": [
            {"title": title, "image": image_to_data_uri(image, max_side=420)}
            for title, image in result.stages
        ],
    }


class SkinRecoveryHandler(BaseHTTPRequestHandler):
    server_version = "SkinRecoveryHTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("[SkinRecovery] " + fmt % args + "\n")
        sys.stdout.flush()

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            # 动态注入 Three.js
            three_js_content = load_three_js()
            if three_js_content:
                script_tag = f'<script>{three_js_content}</script>'
            else:
                script_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
            html = INDEX_HTML.replace("{{THREE_JS_PLACEHOLDER}}", script_tag)
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error(404)
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            boundary = b""
            if "boundary=" in content_type:
                boundary = content_type.split("boundary=")[-1].strip().encode()
            if not boundary:
                raise ValueError("缺少 multipart boundary")
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            form = _parse_multipart(body, boundary)
            if "pre" not in form or "post" not in form:
                raise ValueError("请同时上传术前照片和术后照片")
            mode_field = form.get("mode")
            mode = mode_field.value if hasattr(mode_field, "value") else "face"
            mode = mode if mode in MODE_LABELS else "face"
            # 提取动物物种/品系/敷料时间参数
            def _field(name, default=""):
                f = form.get(name)
                return f.value if hasattr(f, "value") and f.value else default
            animal_species = _field("animal_species")
            mouse_strain = _field("mouse_strain")
            dressing_time = _field("dressing_time")
            pre_img = image_from_upload(form["pre"])
            post_img = image_from_upload(form["post"])
            print(f"[SkinRecovery] analyze request mode={MODE_LABELS[mode]} species={animal_species} strain={mouse_strain} dressing={dressing_time}", flush=True)
            started = time.time()
            payload = make_payload(pre_img, post_img, mode,
                                   animal_species=animal_species,
                                   mouse_strain=mouse_strain,
                                   dressing_time=dressing_time)
            print(f"[SkinRecovery] analyze done score={payload['overall_score']:.2f} cost={time.time() - started:.2f}s", flush=True)
            self.send_json(200, payload)
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


def find_port(start: int = PORT) -> int:
    import socket

    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("没有找到可用端口")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skin Recovery Assessor Web UI")
    parser.add_argument("--no-browser", action="store_true", help="只启动服务，不自动打开浏览器")
    args = parser.parse_args()

    port = find_port(PORT)
    url = f"http://{HOST}:{port}"
    server = ThreadingHTTPServer((HOST, port), SkinRecoveryHandler)
    print("=" * 68)
    print("Skin Recovery Assessor Web 版已启动")
    print(f"请在浏览器打开：{url}")
    print("退出：在终端按 Ctrl+C")
    print("=" * 68)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
