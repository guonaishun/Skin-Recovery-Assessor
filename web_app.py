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
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: #eaf2ff;
      background: #081019;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 16px;
      padding: 18px;
    }
    .sidebar, .panel {
      background: #111b27;
      border: 1px solid #243449;
      border-radius: 8px;
    }
    .sidebar { padding: 18px; display: flex; flex-direction: column; gap: 16px; }
    h1 { margin: 0; font-size: 25px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .muted { color: #91a6bc; line-height: 1.55; font-size: 13px; }
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .mode {
      border: 1px solid #2b3d53;
      background: #0b141f;
      color: #dcecff;
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      text-align: center;
      font-weight: 700;
    }
    .mode.active { border-color: #37d5ff; background: #0d3140; color: white; }
    .upload { display: grid; gap: 8px; }
    .upload label {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      border: 1px dashed #3b536e;
      border-radius: 8px;
      background: #0b141f;
      color: #dcecff;
      cursor: pointer;
      font-weight: 700;
    }
    input[type="file"] { display: none; }
    .file-name { color: #f7d154; font-size: 12px; overflow-wrap: anywhere; }
    button.primary {
      width: 100%;
      border: 0;
      border-radius: 8px;
      background: #1f9fca;
      color: white;
      font-size: 16px;
      font-weight: 800;
      padding: 14px 16px;
      cursor: pointer;
    }
    button.primary:disabled { background: #536576; cursor: wait; }
    .log {
      min-height: 170px;
      max-height: 230px;
      overflow: auto;
      padding: 10px;
      background: #07101a;
      border: 1px solid #223247;
      border-radius: 8px;
      color: #bdefff;
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
      background: #07101a;
      border: 1px solid #223247;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      color: #60768e;
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
        rgba(55,213,255,.04) 30%,
        rgba(55,213,255,.12) 60%,
        rgba(0,255,204,.18) 85%,
        #37d5ff 98%,
        #00ffcc 100%);
      box-shadow: 0 2px 30px rgba(55,213,255,.5), 0 2px 80px rgba(55,213,255,.2);
      animation: scanSweep 2.2s ease-in-out infinite;
      z-index: 10; pointer-events: none;
    }
    .preview.scanning::after {
      content: ""; position: absolute; inset: 0;
      background: repeating-linear-gradient(0deg,
        transparent 0px, transparent 2px,
        rgba(55,213,255,.015) 2px, rgba(55,213,255,.015) 3px);
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
      background: rgba(7, 16, 26, .88); border: 1px solid #37d5ff;
      border-radius: 4px; padding: 5px 16px;
      color: #37d5ff; font-size: 11px; font-weight: 700;
      z-index: 11; display: none; letter-spacing: 2px;
      animation: badgePulse 1.2s ease infinite;
      text-transform: uppercase;
    }
    .preview.scanning .scan-badge { display: block; }
    @keyframes badgePulse {
      0%, 100% { box-shadow: 0 0 8px rgba(55,213,255,.3); opacity: .85; }
      50% { box-shadow: 0 0 22px rgba(55,213,255,.7); opacity: 1; }
    }
    .progress-wrap { display: grid; gap: 10px; }
    .progress-line { display: flex; justify-content: space-between; color: #a9bfd7; font-size: 13px; }
    .bar { height: 10px; background: #07101a; border-radius: 999px; border: 1px solid #223247; overflow: hidden; }
    .fill { width: 0%; height: 100%; background: linear-gradient(90deg, #37d5ff, #f7d154); transition: width .25s ease; }
    .result-grid {
      display: grid;
      grid-template-columns: 1.05fr .95fr;
      gap: 16px;
      min-height: 350px;
    }
    .hero-result {
      position: relative;
      height: 420px;
      background: #07101a;
      border: 1px solid #223247;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #60768e;
      font-weight: 800;
    }
    .hero-result img { width: 100%; height: 100%; object-fit: contain; }
    .scan {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 2px;
      left: 0;
      background: #37d5ff;
      box-shadow: 0 0 18px #37d5ff;
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
    .score-number { font-size: 52px; color: #37d5ff; font-weight: 900; }
    .verdict { color: #dce8f4; line-height: 1.6; }
    .radar-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;
    }
    .radar-box { text-align: center; }
    .radar-box img { width: 100%; max-height: 280px; object-fit: contain; background: #07101a; border-radius: 8px; border: 1px solid #223247; }
    .radar-box .radar-label { color: #91a6bc; font-size: 12px; margin-top: 6px; font-weight: 700; }
    .radar img { width: 100%; max-height: 300px; object-fit: contain; background: #07101a; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 8px; border-bottom: 1px solid #243449; text-align: left; }
    th { color: #9fb7cf; font-weight: 800; }
    td { color: #eaf2ff; }

    .analysis-lab {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 16px;
    }
    .pipeline-window {
      position: relative;
      overflow: visible;
      background:
        radial-gradient(circle at 22% 18%, rgba(55, 213, 255, .16), transparent 24%),
        linear-gradient(135deg, #07101a, #101b28 55%, #07101a);
      border: 1px solid #223247;
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
      border: 1px solid #2b3d53;
      border-radius: 8px;
      background: rgba(7, 16, 26, .78);
      overflow: visible;
    }
    .pipe-step.active {
      border-color: #37d5ff;
      box-shadow: inset 0 0 24px rgba(55, 213, 255, .18), 0 0 20px rgba(55, 213, 255, .10);
    }
    .pipe-index { color: #f7d154; font-weight: 900; font-size: 12px; }
    .pipe-title { margin-top: 8px; font-weight: 900; color: #ffffff; font-size: 14px; }
    .pipe-desc { margin-top: 8px; color: #91a6bc; line-height: 1.45; font-size: 12px; }
    .pipe-step::after {
      content: "";
      position: absolute;
      left: -70%;
      top: 0;
      width: 55%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(55, 213, 255, .28), transparent);
      transform: skewX(-15deg);
    }
    .pipe-step.active::after { animation: sweep 1.2s ease infinite; }
    @keyframes sweep { from { left: -70%; } to { left: 125%; } }
    @keyframes pulseGlow {
      0%, 100% { box-shadow: inset 0 0 12px rgba(55,213,255,.1); }
      50% { box-shadow: inset 0 0 30px rgba(55,213,255,.28); }
    }
    .pipe-step .pipe-img {
      width: 100%; height: 180px; object-fit: contain;
      border-radius: 5px; margin-top: 6px; display: none;
      border: 1px solid #2b3d53; opacity: 0;
      transition: opacity .6s ease;
      background: #07101a;
    }
    .pipe-step.has-img .pipe-img { display: block; opacity: 1; }
    .pipe-step.has-img .pipe-desc { display: none; }
    .pipe-step.done { border-color: #2ecc71; }
    .pipe-step.done::before {
      content: "✓"; position: absolute; top: 6px; right: 10px;
      color: #2ecc71; font-weight: 900; font-size: 16px;
    }
    .pipe-label {
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      color: #60768e; font-weight: 800; font-size: 14px;
    }
    .pipeline-window { height: auto; min-height: 420px; }
    .model-wrap {
      position: relative; height: 420px;
      background: radial-gradient(ellipse at center, #0a1825 0%, #06111c 100%);
      border: 1px solid #223247; border-radius: 8px; overflow: hidden;
      transition: box-shadow .5s ease;
    }
    .model-wrap.model-active {
      box-shadow: 0 0 30px rgba(55,213,255,.15), inset 0 0 20px rgba(55,213,255,.05);
    }
    .model-3d-container {
      width: 100%; height: 100%; display: flex;
      align-items: center; justify-content: center;
    }
    .model-3d-container canvas { display: block; width: 100% !important; height: 100% !important; }
    .model-placeholder { color: #60768e; font-weight: 800; font-size: 14px; }
    #model3d { width: 100%; height: 100%; display: block; }
    .model-controls {
      position: absolute; bottom: 12px; right: 12px;
      display: flex; gap: 6px; z-index: 10;
    }
    .ctrl-btn {
      border: 1px solid #2b3d53; background: rgba(7,16,26,.8);
      color: #91a6bc; border-radius: 6px; padding: 5px 10px;
      font-size: 11px; cursor: pointer; font-weight: 700;
      transition: all .2s ease;
    }
    .ctrl-btn:hover { border-color: #37d5ff; color: #fff; }
    .ctrl-btn.active { border-color: #37d5ff; background: rgba(55,213,255,.15); color: #37d5ff; }
    .model-hud {
      position: absolute;
      left: 12px;
      top: 12px;
      right: 12px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: #cfe9ff;
      font-size: 12px;
      pointer-events: none;
    }
    .hud-chip {
      border: 1px solid #2b3d53;
      background: rgba(7, 16, 26, .72);
      border-radius: 8px;
      padding: 7px 9px;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .top, .result-grid, .analysis-lab { grid-template-columns: 1fr; }

    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div>
        <h1>Skin Recovery Assessor</h1>
        <div class="muted">术前/术后图像配准 · 多维恢复评分 · 动态热力分析</div>
      </div>

      <section>
        <h2>分析对象</h2>
        <div class="mode-grid">
          <button class="mode active" data-mode="face" type="button">人脸分析</button>
          <button class="mode" data-mode="animal" type="button">动物皮肤分析</button>
        </div>
        <p id="modeHint" class="muted">人脸模式：优先定位面部区域，并结合肤色 ROI 进行分析。</p>
      </section>

      <section class="upload">
        <h2>选择图像</h2>
        <label for="preFile">选择术前照片</label>
        <input id="preFile" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp,image/heif,image/heic" />
        <div id="preName" class="file-name">术前：未选择</div>
        <label for="postFile">选择术后照片</label>
        <input id="postFile" type="file" accept="image/jpeg,image/png,image/bmp,image/tiff,image/webp,image/heif,image/heic" />
        <div id="postName" class="file-name">术后：未选择</div>
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
        const text = state.mode === "animal"
          ? "动物皮肤模式：使用更宽松的表面/纹理 ROI，适合宠物、实验动物或局部皮肤照片。"
          : "人脸模式：优先定位面部区域，并结合肤色 ROI 进行分析。";
        $("modeHint").textContent = text;
        log(`切换分析对象：${btn.textContent}`);
      });
    });

    $("preFile").addEventListener("change", (event) => {
      state.pre = event.target.files[0];
      $("preName").textContent = state.pre ? `术前：${state.pre.name}` : "术前：未选择";
      if (state.pre) { preview(state.pre, "prePreview"); log(`已选择术前照片：${state.pre.name}`); }
    });

    $("postFile").addEventListener("change", (event) => {
      state.post = event.target.files[0];
      $("postName").textContent = state.post ? `术后：${state.post.name}` : "术后：未选择";
      if (state.post) { preview(state.post, "postPreview"); log(`已选择术后照片：${state.post.name}`); }
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

    /* ======== Three.js 3D 皮肤恢复仿真建模 - 三维点云重建 ======== */
    function initThreeModel(payload) {
      const container = $("model3dContainer");
      container.innerHTML = "";
      $("modelWrap").classList.add("model-active");
      const W = container.clientWidth || 400, H = container.clientHeight || 380;
      const fr = payload.face_region || { x: 0, y: 0, w: 1, h: 1, type: "full" };
      const isProfile = fr.type === "profile";
      const isPartial = fr.type === "partial";

      // ---- 场景 / 相机 / 渲染器 ----
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(36, W / H, 0.1, 100);
      camera.position.set(0, 0.08, isProfile ? 3.8 : 4.5);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(W, H);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x06111c, 1);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.15;
      container.appendChild(renderer.domElement);

      // ---- 灯光系统 ----
      scene.add(new THREE.AmbientLight(0x667788, 0.6));
      const keyLight = new THREE.DirectionalLight(0xffeedd, 0.85);
      keyLight.position.set(3, 4, 5); scene.add(keyLight);
      const fillLight = new THREE.DirectionalLight(0x8899bb, 0.35);
      fillLight.position.set(-3, 2, 3); scene.add(fillLight);
      const rimLight = new THREE.PointLight(0x37d5ff, 0.7, 15);
      rimLight.position.set(0, 0.5, -3.5); scene.add(rimLight);

      // ---- 3D 人脸点云 (Open3D PointCloud 风格) ----
      const faceGroup = new THREE.Group();
      let pointCloud = null;
      let wireMesh = null;

      if (payload.face_vertices && payload.face_vertices.length > 0) {
        // 构建点云 BufferGeometry
        const geo = new THREE.BufferGeometry();
        const posArr = new Float32Array(payload.face_vertices.flat());
        geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
        if (payload.face_colors && payload.face_colors.length > 0) {
          const colArr = new Float32Array(payload.face_colors.flat());
          geo.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
        }
        const ptMat = new THREE.PointsMaterial({
          size: 0.038, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 0.93,
        });
        pointCloud = new THREE.Points(geo, ptMat);
        faceGroup.add(pointCloud);

        // 半透明线框支撑网格
        const wireGeo = new THREE.PlaneGeometry(2.6, 3.2, 48, 48);
        const wPos = wireGeo.attributes.position;
        for (let i = 0; i < wPos.count; i++) {
          const x = wPos.getX(i), y = wPos.getY(i);
          const u = x / 1.3, v = y / 1.6;
          const r2 = u*u + v*v*0.85;
          let z = 0.48 * Math.exp(-r2 * 0.8);
          z += 0.16 * Math.exp(-(u*u + (v-0.2)**2) / 0.028);
          z += 0.07 * Math.exp(-((u-0.32)**2 + (v+0.12)**2) / 0.035);
          z += 0.07 * Math.exp(-((u+0.32)**2 + (v+0.12)**2) / 0.035);
          z -= 0.04 * Math.exp(-(u*u + (v-0.48)**2) / 0.022);
          z += 0.04 * Math.exp(-(u*u*0.7 + (v-0.75)**2) / 0.055);
          wPos.setZ(i, z);
        }
        wireGeo.computeVertexNormals();
        wireMesh = new THREE.Mesh(wireGeo, new THREE.MeshBasicMaterial({
          color: 0x37d5ff, wireframe: true, transparent: true, opacity: 0.14,
        }));
        wireMesh.visible = true;
        faceGroup.add(wireMesh);
      } else {
        // 回退：前端生成点云
        const rows = 80, cols = 64;
        const positions = [], colors = [];
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const u = (c / (cols-1)) * 2 - 1;
            const v = (r / (rows-1)) * 2 - 1;
            const r2 = u*u + v*v*0.85;
            let z = 0.48 * Math.exp(-r2*0.8);
            z += 0.16 * Math.exp(-(u*u + (v-0.2)**2)/0.028);
            z -= 0.04 * Math.exp(-(u*u + (v-0.48)**2)/0.022);
            positions.push(u * 1.3, v * 1.6, z);
            colors.push(0.84, 0.64, 0.54);
          }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({
          size: 0.038, vertexColors: true, sizeAttenuation: true,
        }));
        faceGroup.add(pointCloud);
        wireMesh = new THREE.Mesh(); wireMesh.visible = false;
        faceGroup.add(wireMesh);
      }

      // ---- 关键特征点标注 ----
      if (payload.feature_points && payload.feature_points.positions && payload.feature_points.positions.length > 0) {
        const fpGeo = new THREE.BufferGeometry();
        const fpPos = new Float32Array(payload.feature_points.positions.flat());
        fpGeo.setAttribute("position", new THREE.BufferAttribute(fpPos, 3));
        if (payload.feature_points.colors && payload.feature_points.colors.length > 0) {
          const fpCol = new Float32Array(payload.feature_points.colors.flat());
          fpGeo.setAttribute("color", new THREE.BufferAttribute(fpCol, 3));
        }
        const fpMat = new THREE.PointsMaterial({
          size: 0.12, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 1.0,
        });
        const featureCloud = new THREE.Points(fpGeo, fpMat);
        faceGroup.add(featureCloud);
        // 特征点光环
        const glowGeo = new THREE.BufferGeometry();
        glowGeo.setAttribute("position", new THREE.BufferAttribute(fpPos.slice(), 3));
        const glowMat = new THREE.PointsMaterial({
          size: 0.22, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 0.3,
        });
        const glowCloud = new THREE.Points(glowGeo, glowMat);
        if (payload.feature_points.colors && payload.feature_points.colors.length > 0) {
          const glowCol = new Float32Array(payload.feature_points.colors.flat());
          glowGeo.setAttribute("color", new THREE.BufferAttribute(glowCol, 3));
        }
        faceGroup.add(glowCloud);
      }
      if (payload.back_vertices && payload.back_vertices.length > 0) {
        const backGeo = new THREE.BufferGeometry();
        const backPosArr = new Float32Array(payload.back_vertices.flat());
        backGeo.setAttribute("position", new THREE.BufferAttribute(backPosArr, 3));
        if (payload.back_colors && payload.back_colors.length > 0) {
          const backColArr = new Float32Array(payload.back_colors.flat());
          backGeo.setAttribute("color", new THREE.BufferAttribute(backColArr, 3));
        }
        const backPtMat = new THREE.PointsMaterial({
          size: 0.038, vertexColors: true,
          sizeAttenuation: true, transparent: true, opacity: 0.93,
        });
        const backCloud = new THREE.Points(backGeo, backPtMat);
        faceGroup.add(backCloud);
      }

      // 外发光环
      const ringGeo = new THREE.TorusGeometry(2.1, 0.015, 16, 120);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x37d5ff, transparent: true, opacity: 0.35 });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.z = -0.6;
      faceGroup.add(ring);

      // 底座光晕盘
      const baseGeo = new THREE.CircleGeometry(1.8, 64);
      const baseMat = new THREE.MeshBasicMaterial({ color: 0x37d5ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide });
      const base = new THREE.Mesh(baseGeo, baseMat);
      base.rotation.x = -Math.PI / 2; base.position.y = -1.8;
      faceGroup.add(base);

      scene.add(faceGroup);

      // ---- 粒子系统 ----
      const pCount = 800;
      const pGeo = new THREE.BufferGeometry();
      const pPos = new Float32Array(pCount * 3), pCol = new Float32Array(pCount * 3);
      for (let i = 0; i < pCount; i++) {
        const a = Math.random() * Math.PI * 2, rad = 2.2 + Math.random() * 2.5;
        pPos[i*3] = Math.cos(a) * rad;
        pPos[i*3+1] = (Math.random() - 0.5) * 4;
        pPos[i*3+2] = Math.sin(a) * rad - 1.5;
        const gold = Math.random() > 0.55;
        pCol[i*3] = gold ? 0.97 : 0.22; pCol[i*3+1] = gold ? 0.82 : 0.84; pCol[i*3+2] = gold ? 0.33 : 1.0;
      }
      pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
      pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
      const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
        size: 0.022, vertexColors: true, transparent: true, opacity: 0.5,
      }));
      scene.add(particles);

      // ---- 鼠标拖拽旋转 ----
      let dragging = false, prevMX = 0, prevMY = 0, rotY = 0, rotX = 0;
      renderer.domElement.addEventListener("pointerdown", e => { dragging = true; prevMX = e.clientX; prevMY = e.clientY; renderer.domElement.style.cursor = "grabbing"; });
      renderer.domElement.addEventListener("pointermove", e => {
        if (!dragging) return;
        rotY += (e.clientX - prevMX) * 0.006;
        rotX += (e.clientY - prevMY) * 0.004;
        rotX = Math.max(-0.6, Math.min(0.6, rotX));
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
        if (!dragging) rotY += 0.003;
        faceGroup.rotation.y += (rotY - faceGroup.rotation.y) * 0.08;
        faceGroup.rotation.x += (rotX - faceGroup.rotation.x) * 0.08;
        // 点云呼吸效果
        if (pointCloud && pointCloud.material) {
          pointCloud.material.size = 0.038 + Math.sin(t * 1.5) * 0.004;
        }
        // 光环脉冲
        ringMat.opacity = 0.25 + Math.sin(t * 1.8) * 0.12;
        ring.rotation.z = t * 0.15;
        baseMat.opacity = 0.04 + Math.sin(t * 1.5) * 0.02;
        // 粒子旋转
        particles.rotation.y = t * 0.1;
        particles.rotation.x = Math.sin(t * 0.25) * 0.04;
        // 轮光动态
        rimLight.intensity = 0.65 + Math.sin(t * 2) * 0.15;
        renderer.render(scene, camera);
      }
      animate();
      state.threeObj = { scene, camera, renderer, pointCloud, wireMesh };
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


def _face_depth(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """计算面部三维深度值（图像坐标：vv=-1 顶部/额头，vv=+1 底部/下巴）。"""
    r2 = uu**2 + vv**2 * 0.85
    zz = 0.48 * np.exp(-r2 * 0.8)
    # 鼻子（vv=0.2 = 面部中央偏下）
    zz += 0.16 * np.exp(-(uu**2 + (vv - 0.2)**2) / 0.028)
    # 额骨隆起（vv=-0.55 = 额头）
    zz += 0.06 * np.exp(-(uu**2 * 0.7 + (vv + 0.55)**2) / 0.055)
    # 下巴隆起（vv=0.75）
    zz += 0.04 * np.exp(-(uu**2 * 0.7 + (vv - 0.75)**2) / 0.055)
    # 左颧骨
    zz += 0.07 * np.exp(-((uu - 0.32)**2 + (vv + 0.12)**2) / 0.035)
    # 右颧骨
    zz += 0.07 * np.exp(-((uu + 0.32)**2 + (vv + 0.12)**2) / 0.035)
    # 左眼窝
    zz -= 0.055 * np.exp(-((uu - 0.28)**2 + (vv - 0.02)**2) / 0.018)
    # 右眼窝
    zz -= 0.055 * np.exp(-((uu + 0.28)**2 + (vv - 0.02)**2) / 0.018)
    # 嘴巴凹槽（vv=0.48）
    zz -= 0.04 * np.exp(-(uu**2 + (vv - 0.48)**2) / 0.022)
    # 人中
    zz += 0.05 * np.exp(-(uu**2 + (vv - 0.34)**2) / 0.045)
    # 太阳穴
    zz += 0.04 * np.exp(-((uu - 0.55)**2 + (vv + 0.1)**2) / 0.06)
    zz += 0.04 * np.exp(-((uu + 0.55)**2 + (vv + 0.1)**2) / 0.06)
    return zz


def build_3d_face_points(rows: int = 80, cols: int = 64,
                         ref_img: Optional[np.ndarray] = None,
                         face_region: Optional[dict] = None) -> dict:
    """构建三维人脸点云 + 后脑点云，形成立体头部。
    依据输入图像的人脸检测结果调整模型比例，并用图像亮度调制深度细节。
    """
    face_w, face_h = 2.6, 3.2

    # 根据人脸检测结果调整模型宽高比
    if face_region and face_region.get('w', 0) > 0 and face_region.get('h', 0) > 0:
        aspect = face_region['w'] / max(face_region['h'], 0.01)
        # 标准脸宽高比 ≈ 0.75，按偏差调整
        deviation = aspect / 0.75
        face_w = 2.6 * np.clip(deviation, 0.82, 1.18)

    hw, hh = face_w / 2, face_h / 2
    u = np.linspace(-1, 1, cols)
    v = np.linspace(-1, 1, rows)
    uu, vv = np.meshgrid(u, v)
    xx = uu.copy()
    yy = -vv.copy()  # 图像坐标 → 显示坐标：翻转 y 轴

    # 脸型轮廓：下巴处（vv>0.1）收窄
    jaw_taper = 1.0 - 0.42 * np.power(np.maximum(0, vv - 0.1), 1.4)
    cheek_w = 1.0 + 0.06 * np.exp(-np.power(vv + 0.15, 2) / 0.12)
    xx = xx * np.maximum(0.12, jaw_taper) * cheek_w

    zz = _face_depth(uu, vv)

    # 用图像亮度信息调制深度：亮区（额头/鼻梁）凸起，暗区（眼窝/鼻翼）凹陷
    if ref_img is not None and ref_img.size > 0:
        gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray_small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
        # 亮度偏离均值部分作为深度细节（平滑处理后取低频成分）
        gray_smooth = cv2.GaussianBlur(gray_small, (5, 5), 1.0)
        brightness_detail = (gray_smooth - 0.5) * 0.06
        zz = zz + brightness_detail

    xx = xx * hw
    yy = yy * hh

    front_verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 从参考图像采样正面顶点颜色
    if ref_img is not None and ref_img.size > 0:
        ih, iw = ref_img.shape[:2]
        cu = ((uu.ravel() + 1) * 0.5 * (iw - 1)).astype(int).clip(0, iw - 1)
        cv_ = ((vv.ravel() + 1) * 0.5 * (ih - 1)).astype(int).clip(0, ih - 1)
        front_colors = (ref_img[cv_, cu].astype(np.float32) / 255.0).clip(0, 1)
    else:
        n = len(front_verts)
        base = np.array([0.84, 0.64, 0.54], dtype=np.float32)
        front_colors = np.tile(base, (n, 1))

    # ---- 关键特征点：鼻尖、双眼、嘴角、下巴、额头 ----
    # 图像坐标：vv=-1 顶/额头，vv=+1 底/下巴
    feature_specs = [
        # (name, uu, vv, color_rgb_01)
        ("鼻尖",    0.0,   0.20, [1.0, 0.25, 0.25]),    # 红
        ("左眼",   -0.28, -0.02, [0.25, 0.85, 1.0]),    # 青
        ("右眼",    0.28, -0.02, [0.25, 0.85, 1.0]),    # 青
        ("左嘴角", -0.18,  0.48, [1.0, 0.85, 0.15]),    # 黄
        ("右嘴角",  0.18,  0.48, [1.0, 0.85, 0.15]),    # 黄
        ("下巴",    0.0,   0.78, [0.40, 1.0, 0.40]),    # 绿
        ("额头",    0.0,  -0.55, [0.85, 0.55, 1.0]),    # 紫
        ("左颚骨", -0.52,  0.10, [1.0, 0.65, 0.20]),    # 橙
        ("右颚骨",  0.52,  0.10, [1.0, 0.65, 0.20]),    # 橙
    ]
    feat_positions, feat_colors = [], []
    for name, fu, fv, fc in feature_specs:
        fx = fu * max(0.12, 1.0 - 0.42 * max(0, fv - 0.1)**1.4) * (1.0 + 0.06 * np.exp(-(fv + 0.15)**2 / 0.12)) * hw
        fy = -fv * hh  # 显示坐标
        fz = float(_face_depth(np.array([fu]), np.array([fv]))[0])
        feat_positions.append([float(fx), float(fy), float(fz)])
        feat_colors.append(fc)
    feature_points = {"positions": feat_positions, "colors": feat_colors}

    # ---- 后脑建模：半椭球壳，与正面无缝衔接 ----
    back_rows, back_cols = 60, 48
    bu = np.linspace(-1, 1, back_cols)
    bv = np.linspace(-1, 1, back_rows)
    buu, bvv = np.meshgrid(bu, bv)
    # 半球深度：中心最凸，边缘趋向 0（与正面边缘对齐）
    r_back = np.sqrt(buu**2 + bvv**2 * 0.85)
    back_depth = -0.38 * np.sqrt(np.maximum(0, 1 - np.minimum(r_back, 1.0)**2))
    # 颅骨后部微微凸起
    back_depth += 0.06 * np.exp(-(buu**2 + (bvv + 0.3)**2) / 0.5)
    # 应用同样的脸型轮廓
    bxx = buu.copy()
    byy = -bvv.copy()
    bjaw = 1.0 - 0.42 * np.power(np.maximum(0, bvv - 0.1), 1.4)
    bcheek = 1.0 + 0.06 * np.exp(-np.power(bvv + 0.15, 2) / 0.12)
    bxx = bxx * np.maximum(0.12, bjaw) * bcheek
    bxx = bxx * hw
    byy = byy * hh
    back_verts = np.column_stack([bxx.ravel(), byy.ravel(), back_depth.ravel()])

    # 后脑颜色：肤色渐变到深色（模拟头发/头皮）
    n_back = len(back_verts)
    back_colors = np.zeros((n_back, 3), dtype=np.float32)
    if ref_img is not None and ref_img.size > 0:
        edge_r = np.sqrt(buu.ravel()**2 + bvv.ravel()**2)
        edge_factor = np.clip(edge_r / 0.9, 0, 1)
        skin_base = front_colors.mean(axis=0)
        dark = np.array([0.18, 0.13, 0.10], dtype=np.float32)
        for i in range(3):
            back_colors[:, i] = skin_base[i] * (1 - edge_factor) + dark[i] * edge_factor
    else:
        back_colors[:] = [0.28, 0.22, 0.18]

    logging.info("[3D PointCloud] 正面 %d + 后脑 %d = %d 顶点",
                 len(front_verts), n_back, len(front_verts) + n_back)

    return {
        "positions": front_verts.tolist(),
        "colors": front_colors.tolist(),
        "back_positions": back_verts.tolist(),
        "back_colors": back_colors.tolist(),
        "feature_points": feature_points,
    }


def make_payload(pre_img: np.ndarray, post_img: np.ndarray, mode: str) -> Dict[str, Any]:
    result = analyze(pre_img, post_img, mode=mode)
    # 构建三维人脸点云 (Open3D PointCloud 风格)
    try:
        point_data = build_3d_face_points(rows=80, cols=64, ref_img=result.aligned_post,
                                             face_region=result.face_region)
        face_verts = point_data["positions"]
        face_colors = point_data["colors"]
        back_verts = point_data["back_positions"]
        back_colors = point_data["back_colors"]
        feat_pts = point_data["feature_points"]
    except Exception as exc:
        logging.warning("3D point cloud build failed: %s", exc)
        face_verts = []
        face_colors = []
        back_verts = []
        back_colors = []
        feat_pts = {"positions": [], "colors": []}
    return {
        "ok": True,
        "mode_label": result.mode_label,
        "overall_score": result.overall_score,
        "verdict": result.verdict,
        "overlay_image": image_to_data_uri(result.overlay_image),
        "radar_image": image_to_data_uri(result.radar_image),
        "comparison_radar": image_to_data_uri(result.comparison_radar) if result.comparison_radar is not None else "",
        "face_texture": image_to_data_uri(result.aligned_post, max_side=512),
        "displacement_map": image_to_data_uri(
            cv2.cvtColor(
                overlay_heatmap(
                    np.full_like(result.aligned_post, 128),
                    np.clip(
                        sum(result.heatmaps.values()) / len(result.heatmaps), 0, 1
                    ),
                    result.skin_mask,
                    alpha=0.9,
                ),
                cv2.COLOR_RGB2GRAY,
            ),
            max_side=256,
        ),
        "face_vertices": face_verts,
        "face_colors": face_colors,
        "back_vertices": back_verts,
        "back_colors": back_colors,
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
            pre_img = image_from_upload(form["pre"])
            post_img = image_from_upload(form["post"])
            print(f"[SkinRecovery] analyze request mode={MODE_LABELS[mode]}", flush=True)
            started = time.time()
            payload = make_payload(pre_img, post_img, mode)
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
