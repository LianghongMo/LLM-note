// Interactive figures for the lecture pages.
// `core` holds the numerics (loadable from Node for testing); the widgets
// below draw on <canvas> with colours taken from the page's CSS tokens.
(function () {
  'use strict';

  // ======================================================================
  // Numerics
  // ======================================================================

  function mulberry32(seed) {
    return function () {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    let spare = null;
    return function () {
      if (spare !== null) {
        const s = spare;
        spare = null;
        return s;
      }
      let u = 0;
      while (u === 0) u = rand();
      const r = Math.sqrt(-2 * Math.log(u));
      const th = 2 * Math.PI * rand();
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  }

  // ---- Marchenko–Pastur law and ridge risk (Lecture 2) ----
  function mpEdges(g) {
    const s = Math.sqrt(g);
    return [(1 - s) * (1 - s), (1 + s) * (1 + s)];
  }

  function mpDensity(t, g) {
    const [lm, lp] = mpEdges(g);
    if (t <= lm || t >= lp || t <= 0) return 0;
    return Math.sqrt((lp - t) * (t - lm)) / (2 * Math.PI * g * t);
  }

  // \int f d\mu_gamma: atom (1-1/gamma)_+ at zero plus the continuous part,
  // integrated with t = a + b cos(theta) to remove the square-root edges.
  function mpIntegrate(f, g, q) {
    q = q || 600;
    const a = 1 + g;
    const b = 2 * Math.sqrt(g);
    let s = 0;
    for (let i = 0; i < q; i++) {
      const th = ((i + 0.5) * Math.PI) / q;
      const t = a + b * Math.cos(th);
      const sn = Math.sin(th);
      s += ((b * b * sn * sn) / (2 * Math.PI * g * t)) * f(t);
    }
    return (s * Math.PI) / q + Math.max(1 - 1 / g, 0) * f(0);
  }

  function ridgeRisk(lam, g, r2, s2) {
    const bias = r2 * mpIntegrate((t) => (lam * lam) / ((t + lam) * (t + lam)), g);
    const variance = s2 * g * mpIntegrate((t) => t / ((t + lam) * (t + lam)), g);
    return { risk: bias + variance, bias, variance };
  }

  // ---- 1D diffusion with an Ornstein–Uhlenbeck forward process (Lecture 1) ----
  const MIX = { w: [0.3, 0.45, 0.25], mu: [-2.0, 0.3, 2.4], sd: [0.35, 0.45, 0.25] };
  const T_MAX = 2.5;

  function mixAt(t) {
    // x_t = alpha_t x_0 + sigma_t eps, alpha_t = e^{-t}, sigma_t^2 = 1 - e^{-2t}
    const a = Math.exp(-t);
    const s2 = -Math.expm1(-2 * t);
    return MIX.w.map((w, k) => {
      const v = a * a * MIX.sd[k] * MIX.sd[k] + s2;
      return { m: a * MIX.mu[k], v, logc: Math.log(w) - 0.5 * Math.log(2 * Math.PI * v) };
    });
  }

  function mixDensity(x, comps) {
    let p = 0;
    for (const c of comps) p += Math.exp(c.logc - ((x - c.m) * (x - c.m)) / (2 * c.v));
    return p;
  }

  function mixScore(x, comps) {
    // d/dx log rho_t(x), evaluated with log-sum-exp for stability
    let mx = -Infinity;
    const lp = new Array(comps.length);
    for (let k = 0; k < comps.length; k++) {
      const c = comps[k];
      lp[k] = c.logc - ((x - c.m) * (x - c.m)) / (2 * c.v);
      if (lp[k] > mx) mx = lp[k];
    }
    let num = 0;
    let den = 0;
    for (let k = 0; k < comps.length; k++) {
      const r = Math.exp(lp[k] - mx);
      den += r;
      num += (r * -(x - comps[k].m)) / comps[k].v;
    }
    return num / den;
  }

  function sampleMix(n, gauss, rand) {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u = rand();
      let k = 0;
      let acc = MIX.w[0];
      while (u > acc && k < MIX.w.length - 1) acc += MIX.w[++k];
      out[i] = MIX.mu[k] + MIX.sd[k] * gauss();
    }
    return out;
  }

  // Probability-flow ODE for dx = -x dt + sqrt(2) dW:  dx/dt = -x - s_t(x).
  // One Heun step backwards in time, from t to t - dt.
  function pfOdeStepBack(pos, t, dt) {
    const c1 = mixAt(t);
    const c2 = mixAt(Math.max(t - dt, 0));
    for (let i = 0; i < pos.length; i++) {
      const x = pos[i];
      const k1 = -x - mixScore(x, c1);
      const xe = x - dt * k1;
      const k2 = -xe - mixScore(xe, c2);
      pos[i] = x - 0.5 * dt * (k1 + k2);
    }
  }

  // ---- Deep linear network (Lecture 2) ----
  // log(||z^(L)|| / ||x||) for M independent networks. For a fresh Gaussian W
  // with variance 1/N, W v ~ N(0, ||v||^2 I / N) exactly, so each layer
  // needs N Gaussian draws rather than an N x N matrix.
  function simDeepLinear(N, L, M, gauss) {
    const out = new Float64Array(M);
    for (let m = 0; m < M; m++) {
      let logr = 0;
      for (let l = 0; l < L; l++) {
        let s = 0;
        for (let i = 0; i < N; i++) {
          const z = gauss();
          s += z * z;
        }
        logr += 0.5 * Math.log(s / N);
      }
      out[m] = logr;
    }
    return out;
  }

  // ---- Linear ResNet z <- (I + s_L W) z (Lecture 2) ----
  // Returns P paths of log(||z_l||^2 / ||x||^2), sampled at <= maxPts depths.
  function simResNet(N, L, beta, P, gauss, maxPts) {
    const s = Math.pow(L, -beta);
    const every = Math.max(1, Math.ceil(L / (maxPts || 400)));
    const paths = [];
    const z = new Float64Array(N);
    for (let p = 0; p < P; p++) {
      let nz = 0;
      for (let i = 0; i < N; i++) {
        z[i] = gauss();
        nz += z[i] * z[i];
      }
      nz = Math.sqrt(nz);
      for (let i = 0; i < N; i++) z[i] /= nz;
      let logn2 = 0; // z is kept at unit norm; the norm lives in logn2
      const xs = [0];
      const ys = [0];
      const c = s / Math.sqrt(N);
      for (let l = 1; l <= L; l++) {
        let n2 = 0;
        for (let i = 0; i < N; i++) {
          z[i] += c * gauss();
          n2 += z[i] * z[i];
        }
        logn2 += Math.log(n2);
        const inv = 1 / Math.sqrt(n2);
        for (let i = 0; i < N; i++) z[i] *= inv;
        if (l % every === 0 || l === L) {
          xs.push(l / L);
          ys.push(logn2);
        }
      }
      paths.push({ xs, ys });
    }
    return { paths, s };
  }

  function quantile(sorted, q) {
    const i = (sorted.length - 1) * q;
    const lo = Math.floor(i);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  const core = {
    mulberry32, gaussian, mpEdges, mpDensity, mpIntegrate, ridgeRisk,
    mixAt, mixDensity, mixScore, sampleMix, pfOdeStepBack, T_MAX,
    simDeepLinear, simResNet, quantile,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = core;
  if (typeof document === 'undefined') return;

  // ======================================================================
  // Canvas plotting
  // ======================================================================

  function tokens() {
    const s = getComputedStyle(document.documentElement);
    const g = (n) => s.getPropertyValue(n).trim();
    return {
      ink: g('--ink'), muted: g('--ink-2'), rule: g('--rule'), accent: g('--accent'),
      accentSoft: g('--accent-soft'), theory: g('--theory'), sheet: g('--sheet'),
      mono: g('--font-mono') || 'monospace',
    };
  }

  function niceStep(span, n) {
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const e = raw / mag;
    return mag * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1);
  }

  function linTicks(lo, hi, n) {
    const step = niceStep(hi - lo, n);
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
      ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    return ticks.map((v) => ({ v, label: v.toFixed(dec) }));
  }

  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) {
      const v = Math.pow(10, e);
      out.push({ v, label: e < 0 ? v.toFixed(-e) : String(v) });
    }
    return out;
  }

  class Plot {
    constructor(canvas, margin) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.margin = Object.assign({ l: 46, r: 14, t: 24, b: 38 }, margin || {});
    }

    begin(x0, x1, y0, y1, opt) {
      opt = opt || {};
      const dpr = window.devicePixelRatio || 1;
      const w = this.canvas.clientWidth;
      const h = this.canvas.clientHeight;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      const c = this.ctx;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, w, h);
      this.w = w;
      this.h = h;
      this.pw = w - this.margin.l - this.margin.r;
      this.ph = h - this.margin.t - this.margin.b;
      this.xlog = !!opt.xlog;
      this.x0 = x0; this.x1 = x1; this.y0 = y0; this.y1 = y1;
      this.col = tokens();
      this.font = `11px ${this.col.mono}`;
      return this;
    }

    X(x) {
      const m = this.margin;
      if (this.xlog) {
        const a = Math.log10(this.x0);
        const b = Math.log10(this.x1);
        return m.l + ((Math.log10(x) - a) / (b - a)) * this.pw;
      }
      return m.l + ((x - this.x0) / (this.x1 - this.x0)) * this.pw;
    }

    Y(y) {
      return this.margin.t + (1 - (y - this.y0) / (this.y1 - this.y0)) * this.ph;
    }

    axes(o) {
      const c = this.ctx;
      const m = this.margin;
      const col = this.col;
      c.font = this.font;
      c.lineWidth = 1;
      const xt = o.xticks || (this.xlog ? logTicks(this.x0, this.x1) : linTicks(this.x0, this.x1, o.nx || 6));
      const yt = o.yticks || linTicks(this.y0, this.y1, o.ny || 4);
      // grid
      c.strokeStyle = col.rule;
      c.globalAlpha = 0.7;
      c.beginPath();
      for (const t of yt) {
        const y = Math.round(this.Y(t.v)) + 0.5;
        c.moveTo(m.l, y);
        c.lineTo(m.l + this.pw, y);
      }
      c.stroke();
      c.globalAlpha = 1;
      // frame: bottom and left axes
      c.strokeStyle = col.muted;
      c.beginPath();
      c.moveTo(m.l + 0.5, m.t);
      c.lineTo(m.l + 0.5, m.t + this.ph + 0.5);
      c.lineTo(m.l + this.pw, m.t + this.ph + 0.5);
      c.stroke();
      // tick labels
      c.fillStyle = col.muted;
      c.textAlign = 'center';
      c.textBaseline = 'top';
      for (const t of xt) {
        const x = this.X(t.v);
        if (x < m.l - 1 || x > m.l + this.pw + 1) continue;
        c.beginPath();
        c.moveTo(Math.round(x) + 0.5, m.t + this.ph);
        c.lineTo(Math.round(x) + 0.5, m.t + this.ph + 4);
        c.stroke();
        c.fillText(t.label, x, m.t + this.ph + 6);
      }
      c.textAlign = 'right';
      c.textBaseline = 'middle';
      for (const t of yt) c.fillText(t.label, m.l - 6, this.Y(t.v));
      // axis titles
      if (o.xlabel) {
        c.textAlign = 'center';
        c.textBaseline = 'bottom';
        c.fillText(o.xlabel, m.l + this.pw / 2, this.h - 2);
      }
      if (o.title) {
        c.textAlign = 'left';
        c.textBaseline = 'top';
        c.fillStyle = col.ink;
        c.fillText(o.title, m.l, 4);
      }
      return this;
    }

    clip(fn) {
      const c = this.ctx;
      c.save();
      c.beginPath();
      c.rect(this.margin.l + 1, this.margin.t - 2, this.pw, this.ph + 2);
      c.clip();
      fn();
      c.restore();
      return this;
    }

    line(xs, ys, o) {
      o = o || {};
      const c = this.ctx;
      c.save();
      c.strokeStyle = o.color || this.col.ink;
      c.lineWidth = o.width || 1.5;
      c.globalAlpha = o.alpha == null ? 1 : o.alpha;
      c.setLineDash(o.dash || []);
      c.lineJoin = 'round';
      c.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        if (!isFinite(ys[i])) { pen = false; continue; }
        const x = this.X(xs[i]);
        const y = this.Y(ys[i]);
        pen ? c.lineTo(x, y) : c.moveTo(x, y);
        pen = true;
      }
      c.stroke();
      c.restore();
      return this;
    }

    bars(edges, heights, o) {
      const c = this.ctx;
      c.save();
      c.fillStyle = o.color;
      c.globalAlpha = o.alpha == null ? 1 : o.alpha;
      const yb = this.Y(Math.max(this.y0, 0));
      for (let i = 0; i < heights.length; i++) {
        if (heights[i] <= 0) continue;
        const xa = this.X(edges[i]);
        const xb = this.X(edges[i + 1]);
        const y = this.Y(heights[i]);
        c.fillRect(xa + 0.5, y, Math.max(xb - xa - 1, 0.5), yb - y);
      }
      c.restore();
      return this;
    }

    area(xs, ys, o) {
      const c = this.ctx;
      c.save();
      c.fillStyle = o.color;
      c.globalAlpha = o.alpha == null ? 1 : o.alpha;
      c.beginPath();
      c.moveTo(this.X(xs[0]), this.Y(0));
      for (let i = 0; i < xs.length; i++) c.lineTo(this.X(xs[i]), this.Y(ys[i]));
      c.lineTo(this.X(xs[xs.length - 1]), this.Y(0));
      c.closePath();
      c.fill();
      c.restore();
      return this;
    }

    vline(x, o) {
      o = o || {};
      return this.line([x, x], [o.from == null ? this.y0 : o.from, o.to == null ? this.y1 : o.to], o);
    }

    hline(y, o) {
      return this.line([this.x0, this.x1], [y, y], o || {});
    }

    dot(x, y, o) {
      const c = this.ctx;
      c.save();
      c.fillStyle = (o && o.color) || this.col.ink;
      c.beginPath();
      c.arc(this.X(x), this.Y(y), (o && o.r) || 3.5, 0, 2 * Math.PI);
      c.fill();
      c.restore();
      return this;
    }

    text(x, y, str, o) {
      o = o || {};
      const c = this.ctx;
      c.save();
      c.font = this.font;
      c.fillStyle = o.color || this.col.ink;
      c.textAlign = o.align || 'left';
      c.textBaseline = o.baseline || 'alphabetic';
      let px = this.X(x) + (o.dx || 0);
      const w = c.measureText(str).width;
      // keep labels inside the plotting area
      const lo = this.margin.l + 2;
      const hi = this.margin.l + this.pw - 2;
      const left = c.textAlign === 'left' ? px : c.textAlign === 'right' ? px - w : px - w / 2;
      if (left + w > hi) px -= left + w - hi;
      if (left < lo) px += lo - left;
      c.fillText(str, px, this.Y(y) + (o.dy || 0));
      c.restore();
      return this;
    }
  }

  function histogram(data, lo, hi, nb) {
    const edges = new Float64Array(nb + 1);
    const h = new Float64Array(nb);
    const bw = (hi - lo) / nb;
    for (let i = 0; i <= nb; i++) edges[i] = lo + i * bw;
    let n = 0;
    for (let i = 0; i < data.length; i++) {
      const k = Math.floor((data[i] - lo) / bw);
      if (k >= 0 && k < nb) h[k]++;
      n++;
    }
    for (let i = 0; i < nb; i++) h[i] /= n * bw; // density normalisation
    return { edges, h };
  }

  function linspace(a, b, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
    return out;
  }

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? (0).toFixed(d) : v.toFixed(d)).replace('-', '−');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const legend = (color, label, dashed) =>
    `<span><span class="key" style="background:${dashed ? `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)` : color}"></span>${label}</span>`;

  // ======================================================================
  // Widgets
  // ======================================================================

  const widgets = {};

  // ---- Lecture 1: forward noising and reverse probability-flow ODE ----
  widgets.diffusion = function () {
    const M = 5000;
    const slider = $('diff-t');
    const out = $('diff-t-out');
    const runBtn = $('diff-run');
    const resetBtn = $('diff-reset');
    const pd = new Plot($('diff-density'));
    const ps = new Plot($('diff-score'), { t: 22, b: 34 });
    const X0 = -4.5;
    const X1 = 4.5;
    const grid = linspace(X0, X1, 361);

    let seed = 7;
    let x0, eps, pos;
    // the slider is quadratic in t: fine control where the modes separate
    const uToT = (u) => T_MAX * u * u;
    let mode = 'forward';
    let t = uToT(parseFloat(slider.value));
    let running = false;

    function resample() {
      const rand = mulberry32(seed++);
      const g = gaussian(rand);
      x0 = sampleMix(M, g, rand);
      eps = new Float64Array(M);
      for (let i = 0; i < M; i++) eps[i] = g();
      pos = new Float64Array(M);
    }

    function forward() {
      mode = 'forward';
      const a = Math.exp(-t);
      const s = Math.sqrt(-Math.expm1(-2 * t));
      for (let i = 0; i < M; i++) pos[i] = a * x0[i] + s * eps[i];
    }

    function draw() {
      const comps = mixAt(t);
      const dens = grid.map((x) => mixDensity(x, comps));
      const hist = histogram(pos, X0, X1, 90);
      pd.begin(X0, X1, 0, 0.55).axes({ title: 'density ρₜ(x) and samples', xlabel: 'x', ny: 5 });
      pd.clip(() => {
        pd.bars(hist.edges, hist.h, { color: pd.col.accent, alpha: 0.32 });
        pd.line(grid, dens, { color: pd.col.theory, width: 2 });
      });

      const score = grid.map((x) => mixScore(x, comps));
      ps.begin(X0, X1, -6, 6).axes({ title: 'score sₜ(x) = ∂ₓ log ρₜ(x)', ny: 2, xticks: [] });
      ps.clip(() => {
        ps.hline(0, { color: ps.col.muted, width: 1, dash: [3, 3] });
        ps.line(grid, score, { color: ps.col.accent, width: 1.8 });
      });

      out.textContent = t.toFixed(2);
      $('diff-readout').innerHTML = [
        legend(pd.col.accent, mode === 'forward' ? 'samples: xₜ = αₜx₀ + σₜε' : 'samples: reverse ODE from N(0,1)'),
        legend(pd.col.theory, 'exact ρₜ'),
        `<span>αₜ = <b>${fmt(Math.exp(-t), 3)}</b></span>`,
        `<span>σₜ = <b>${fmt(Math.sqrt(-Math.expm1(-2 * t)), 3)}</b></span>`,
      ].join('');
    }

    function run() {
      if (running) return;
      running = true;
      runBtn.disabled = slider.disabled = resetBtn.disabled = true;
      mode = 'reverse';
      const g = gaussian(mulberry32(seed++));
      for (let i = 0; i < M; i++) pos[i] = g();
      // integrate on the slider's quadratic schedule, t_k = T (1 - k/steps)^2,
      // so the steps (and the animation) concentrate at small t
      const steps = 500;
      const tk = (k) => uToT(1 - k / steps);
      let k = 0;
      t = T_MAX;
      const finish = () => {
        t = 0;
        slider.value = '0';
        draw();
        running = false;
        runBtn.disabled = slider.disabled = resetBtn.disabled = false;
      };
      if (reduceMotion) {
        for (; k < steps; k++) pfOdeStepBack(pos, tk(k), tk(k) - tk(k + 1));
        finish();
        return;
      }
      const frame = () => {
        for (let j = 0; j < 3 && k < steps; j++, k++) pfOdeStepBack(pos, tk(k), tk(k) - tk(k + 1));
        t = tk(k);
        slider.value = String(1 - k / steps);
        if (k < steps) {
          draw();
          requestAnimationFrame(frame);
        } else {
          finish();
        }
      };
      draw();
      requestAnimationFrame(frame);
    }

    slider.addEventListener('input', () => {
      t = uToT(parseFloat(slider.value));
      forward();
      draw();
    });
    runBtn.addEventListener('click', run);
    resetBtn.addEventListener('click', () => {
      resample();
      forward();
      draw();
    });

    resample();
    forward();
    return draw;
  };

  // ---- Lecture 2: deep linear network ----
  widgets.deeplinear = function () {
    const NS = [4, 8, 16, 32, 64, 128];
    const nIn = $('dl-n');
    const lIn = $('dl-l');
    const plot = new Plot($('dl-hist'));
    let seed = 11;
    let data = null;
    let params = null;

    function simulate() {
      const N = NS[parseInt(nIn.value, 10)];
      const L = parseInt(lIn.value, 10);
      const M = Math.max(400, Math.min(4000, Math.floor(8e6 / (L * N))));
      const g = gaussian(mulberry32(seed));
      data = simDeepLinear(N, L, M, g);
      params = { N, L, M };
      $('dl-n-out').textContent = String(N);
      $('dl-l-out').textContent = String(L);
    }

    function draw() {
      const { N, L, M } = params;
      const xi = L / N;
      const mu = -xi / 2;
      const sd = Math.sqrt(xi / 2);
      const sorted = Float64Array.from(data).sort();
      const lo = Math.min(quantile(sorted, 0.002), mu - 4 * sd);
      const hi = Math.max(quantile(sorted, 0.998), mu + 4 * sd);
      const hist = histogram(data, lo, hi, 60);
      const xs = linspace(lo, hi, 300);
      const gauss = xs.map((x) => Math.exp(-((x - mu) * (x - mu)) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI)));
      const ymax = Math.max(Math.max(...hist.h), Math.max(...gauss)) * 1.12;

      plot.begin(lo, hi, 0, ymax).axes({ title: 'density of log(‖z‖ / ‖x‖) at the output layer', xlabel: 'log(‖z‖ / ‖x‖)', ny: 4 });
      plot.clip(() => {
        plot.bars(hist.edges, hist.h, { color: plot.col.accent, alpha: 0.32 });
        plot.line(xs, gauss, { color: plot.col.theory, width: 2 });
        plot.vline(0, { color: plot.col.muted, width: 1, dash: [3, 3] });
      });

      let mean = 0;
      for (const v of data) mean += v;
      mean /= M;
      let v2 = 0;
      let m2 = 0;
      for (const v of data) {
        v2 += (v - mean) * (v - mean);
        m2 += Math.exp(2 * v);
      }
      v2 /= M - 1;
      m2 /= M;
      const med = quantile(sorted, 0.5);

      $('dl-readout').innerHTML = [
        legend(plot.col.accent, `${M} networks`),
        legend(plot.col.theory, 'N(−ξ/2, ξ/2)'),
        `<span>ξ = L/N = <b>${fmt(xi, 3)}</b></span>`,
        `<span>mean: <b>${fmt(mean, 3)}</b> vs −ξ/2 = ${fmt(mu, 3)}</span>`,
        `<span>variance: <b>${fmt(v2, 3)}</b> vs ξ/2 = ${fmt(xi / 2, 3)}</span>`,
        `<span>E‖z‖²/‖x‖²: <b>${fmt(m2, 2)}</b> (exact 1)</span>`,
        `<span>typical ‖z‖²/‖x‖²: <b>${fmt(Math.exp(2 * med), 3)}</b> vs e<sup>−ξ</sup> = ${fmt(Math.exp(-xi), 3)}</span>`,
      ].join('');
    }

    let queued = false;
    const update = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        simulate();
        draw();
      });
    };
    nIn.addEventListener('input', update);
    lIn.addEventListener('input', update);
    $('dl-resample').addEventListener('click', () => {
      seed++;
      update();
    });

    simulate();
    return draw;
  };

  // ---- Lecture 2: ResNet depth scaling ----
  widgets.resnet = function () {
    const LS = [10, 30, 100, 300, 1000, 3000];
    const N = 32;
    const P = 40;
    const bIn = $('rn-beta');
    const lIn = $('rn-l');
    const plot = new Plot($('rn-paths'));
    let seed = 5;
    let sim = null;
    let params = null;

    function simulate() {
      const beta = parseInt(bIn.value, 10) / 100;
      const L = LS[parseInt(lIn.value, 10)];
      sim = simResNet(N, L, beta, P, gaussian(mulberry32(seed)), 400);
      params = { beta, L };
      $('rn-beta-out').textContent = beta.toFixed(2);
      $('rn-l-out').textContent = String(L);
    }

    function draw() {
      const { beta, L } = params;
      const s = sim.s;
      const lg = Math.log1p(s * s);
      const tx = linspace(0, 1, 200);
      const ty = tx.map((u) => u * L * lg);
      let lo = Math.min(0, ty[ty.length - 1]);
      let hi = Math.max(0, ty[ty.length - 1]);
      for (const p of sim.paths) {
        for (const y of p.ys) {
          if (y < lo) lo = y;
          if (y > hi) hi = y;
        }
      }
      if (hi - lo < 1) {
        const c = (hi + lo) / 2;
        lo = c - 0.5;
        hi = c + 0.5;
      }
      const pad = 0.08 * (hi - lo);
      plot.begin(0, 1, lo - pad, hi + pad).axes({ title: 'log(‖z‖² / ‖x‖²) along the network', xlabel: 'relative depth ℓ/L', ny: 4, nx: 5 });
      plot.clip(() => {
        for (const p of sim.paths) plot.line(p.xs, p.ys, { color: plot.col.accent, width: 1.1, alpha: 0.4 });
        plot.line(tx, ty, { color: plot.col.theory, width: 2.2, dash: [6, 4] });
      });

      const ls2 = L * s * s;
      const regime =
        beta < 0.5 - 1e-9
          ? '<span class="chip explode">exploding: L sₗ² → ∞</span>'
          : beta > 0.5 + 1e-9
            ? '<span class="chip identity">identity: L sₗ² → 0</span>'
            : '<span class="chip critical">critical: Neural SDE</span>';
      $('rn-readout').innerHTML = [
        regime,
        legend(plot.col.accent, `${P} networks, N = ${N}`),
        legend(plot.col.theory, 'log E‖z‖²', true),
        `<span>sₗ = L⁻ᵝ = <b>${fmt(s, 4)}</b></span>`,
        `<span>L sₗ² = <b>${ls2 < 0.01 ? ls2.toExponential(1) : fmt(ls2, 2)}</b></span>`,
      ].join('').replace(/sₗ/g, 's<sub>L</sub>');
    }

    let queued = false;
    const update = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        simulate();
        draw();
      });
    };
    bIn.addEventListener('input', update);
    lIn.addEventListener('input', update);
    $('rn-resample').addEventListener('click', () => {
      seed++;
      update();
    });

    simulate();
    return draw;
  };

  // ---- Lecture 2: Marchenko–Pastur spectrum and ridge risk ----
  widgets.ridge = function () {
    const gIn = $('rr-gamma');
    const sIn = $('rr-snr');
    const pm = new Plot($('rr-mp'));
    const pr = new Plot($('rr-risk'));
    const lams = Array.from({ length: 161 }, (_, i) => Math.pow(10, -2 + (4 * i) / 160));

    function draw() {
      const g = parseFloat(gIn.value);
      const snr = Math.pow(2, parseFloat(sIn.value));
      const r2 = snr;
      const s2 = 1;
      $('rr-gamma-out').textContent = g.toFixed(2);
      $('rr-snr-out').textContent = snr.toFixed(2);

      // spectrum
      const [lm, lp] = mpEdges(g);
      const atom = Math.max(1 - 1 / g, 0);
      // cosine spacing resolves the square-root edges at λ₋ and λ₊
      const ts = Array.from({ length: 400 }, (_, i) => (lm + lp) / 2 - ((lp - lm) / 2) * Math.cos((Math.PI * i) / 399));
      const rho = ts.map((t) => mpDensity(t, g));
      const trim = lm + 0.02 * (lp - lm);
      let ymax = 0;
      ts.forEach((t, i) => {
        if (t >= trim && rho[i] > ymax) ymax = rho[i];
      });
      ymax *= 1.25;
      pm.begin(0, lp * 1.06, 0, ymax).axes({ title: 'Marchenko–Pastur density ρ(t)', xlabel: 'eigenvalue t', ny: 4, nx: 5 });
      pm.clip(() => {
        pm.area(ts, rho, { color: pm.col.accent, alpha: 0.18 });
        pm.line(ts, rho, { color: pm.col.accent, width: 2 });
        pm.vline(lm, { color: pm.col.muted, width: 1, dash: [3, 3] });
        pm.vline(lp, { color: pm.col.muted, width: 1, dash: [3, 3] });
        if (atom > 0) pm.vline(0, { color: pm.col.theory, width: 4, from: 0, to: ymax * 0.82 });
      });
      if (atom > 0) pm.text(0, ymax * 0.86, `atom at 0: 1−1/γ = ${atom.toFixed(2)}`, { color: pm.col.theory, dx: 6 });

      // risk
      const rs = lams.map((l) => ridgeRisk(l, g, r2, s2));
      const lstar = (s2 * g) / r2;
      const star = ridgeRisk(lstar, g, r2, s2);
      const rmax = Math.min(Math.max(...rs.map((r) => r.risk)), 3 * star.risk) * 1.08;
      pr.begin(1e-2, 1e2, 0, rmax, { xlog: true }).axes({ title: 'excess risk R(λ)', xlabel: 'ridge parameter λ (log scale)', ny: 4 });
      pr.clip(() => {
        pr.line(lams, rs.map((r) => r.bias), { color: pr.col.accent, width: 1.6, dash: [5, 3] });
        pr.line(lams, rs.map((r) => r.variance), { color: pr.col.theory, width: 1.6, dash: [2, 3] });
        pr.line(lams, rs.map((r) => r.risk), { color: pr.col.ink, width: 2.2 });
        pr.vline(lstar, { color: pr.col.muted, width: 1, dash: [3, 3] });
        pr.dot(lstar, star.risk, { color: pr.col.ink });
      });
      pr.text(lstar, star.risk, `λ* = ${lstar.toFixed(2)}`, { dx: 7, dy: -9 });

      $('rr-readout').innerHTML = [
        legend(pr.col.ink, 'R(λ)'),
        legend(pr.col.accent, 'bias', true),
        legend(pr.col.theory, 'variance', true),
        `<span>λ₋ = <b>${fmt(lm, 3)}</b>, λ₊ = <b>${fmt(lp, 3)}</b></span>`,
        `<span>λ* = γ/SNR = <b>${fmt(lstar, 3)}</b></span>`,
        `<span>R(λ*) = <b>${fmt(star.risk, 4)}</b> (bias ${fmt(star.bias, 3)}, variance ${fmt(star.variance, 3)})</span>`,
      ].join('');
    }

    gIn.addEventListener('input', draw);
    sIn.addEventListener('input', draw);
    return draw;
  };

  // ======================================================================
  // Boot: build each widget, redraw on resize, theme change and font load
  // ======================================================================

  const drawers = [];
  document.querySelectorAll('[data-widget]').forEach((fig) => {
    const make = widgets[fig.dataset.widget];
    if (!make) return;
    try {
      const draw = make();
      drawers.push(draw);
      draw();
      if ('ResizeObserver' in window) {
        let last = 0;
        new ResizeObserver((entries) => {
          const w = Math.round(entries[0].contentRect.width);
          if (w !== last) {
            last = w;
            draw();
          }
        }).observe(fig);
      }
    } catch (err) {
      console.error('figure', fig.dataset.widget, err);
    }
  });

  const redrawAll = () => drawers.forEach((d) => d());
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
})();
