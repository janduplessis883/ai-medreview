#!/usr/bin/env python3
"""app.py — AI MedReview4 animated landing page.

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI MedReview4",
    page_icon=":material/clinical_notes:",
    layout="wide",
)

PAGE = """<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  html, body {
    height: 100%;
    overflow: hidden;
    background: radial-gradient(ellipse at 30% 20%, #2e3d52 0%, #232F3E 55%, #131a24 100%);
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  }

  #particles {
    position: fixed;
    inset: 0;
    z-index: 0;
  }

  .stage {
    position: relative;
    z-index: 1;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 26px;
  }

  .line {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    font-weight: 800;
    letter-spacing: 0.04em;
  }

  .line span {
    display: inline-block;
    opacity: 0;
    transform: translateY(50px) rotateX(90deg);
    animation: rise 0.65s cubic-bezier(0.22, 1.4, 0.36, 1) forwards;
  }

  .line span.space { width: 0.32em; }

  @keyframes rise {
    to { opacity: 1; transform: translateY(0) rotateX(0); }
  }

  /* Line 1 — Friends & Family Test Intelligence */
  .line-1 {
    font-size: clamp(1.1rem, 3vw, 2rem);
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
  }
  .line-1 span { color: #FFD580; }

  /* Line 2 — AI MedReview4 */
  .line-2 {
    font-size: clamp(3rem, 9vw, 7.5rem);
  }
  .line-2 span {
    background: linear-gradient(120deg, #FFD580, #FF9900, #E68A00, #FFD580);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation:
      rise 0.65s cubic-bezier(0.22, 1.4, 0.36, 1) forwards,
      shimmer 6s linear infinite;
  }

  @keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
  }

  /* Line 3 — Jev */
  .line-3 {
    font-size: clamp(1.6rem, 4.5vw, 3.2rem);
    letter-spacing: 0.3em;
  }
  .line-3 span { color: #067D62; text-shadow: 0 0 24px rgba(6, 125, 98, 0.55); }

  .rule {
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #FF9900, #FFD814, transparent);
    animation: expand 1.1s ease forwards;
  }

  @keyframes expand { to { width: min(460px, 62vw); } }

  .pulse-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #FF9900;
    opacity: 0;
    animation: fadeIn 0.6s ease forwards, pulse 2.2s ease-in-out infinite;
  }

  @keyframes fadeIn { to { opacity: 1; } }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 153, 0, 0.55); }
    50%      { box-shadow: 0 0 0 16px rgba(255, 153, 0, 0); }
  }
</style>
</head>
<body>
  <canvas id="particles"></canvas>

  <div class="stage">
    <div class="line line-1" id="line1"></div>
    <div class="rule" id="rule"></div>
    <div class="line line-2" id="line2"></div>
    <div class="line line-3" id="line3"></div>
    <div class="pulse-dot" id="dot"></div>
  </div>

<script>
  const LINES = [
    { id: "line1", text: "Friends & Family Test Intelligence", start: 0.2, step: 0.045 },
    { id: "line2", text: "AI MedReview4",                      start: 1.8, step: 0.11  },
    { id: "line3", text: "Jev",                                start: 3.4, step: 0.22  },
  ];

  for (const { id, text, start, step } of LINES) {
    const el = document.getElementById(id);
    [...text].forEach((ch, i) => {
      const span = document.createElement("span");
      if (ch === " ") {
        span.className = "space";
      } else {
        span.textContent = ch;
        span.style.animationDelay = `${start + step * i}s, 0s`;
      }
      el.appendChild(span);
    });
  }

  // Decorative elements timed after the text
  document.getElementById("rule").style.animationDelay = "1.5s";
  document.getElementById("dot").style.animationDelay = "4.2s, 4.8s";

  // Floating particle field (orange-tinted)
  const canvas = document.getElementById("particles");
  const ctx = canvas.getContext("2d");
  let W, H, dots;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function init() {
    resize();
    dots = Array.from({ length: 90 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 2.2 + 0.4,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      a: Math.random() * 0.45 + 0.12,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, W, H);
    for (const d of dots) {
      d.x += d.vx;
      d.y += d.vy;
      if (d.x < -5) d.x = W + 5;
      if (d.x > W + 5) d.x = -5;
      if (d.y < -5) d.y = H + 5;
      if (d.y > H + 5) d.y = -5;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 175, 26, ${d.a})`;
      ctx.fill();
    }
    requestAnimationFrame(tick);
  }

  window.addEventListener("resize", resize);
  init();
  tick();
</script>
</body>
</html>
"""

st.html(PAGE, unsafe_allow_javascript=True)
