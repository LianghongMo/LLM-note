#!/usr/bin/env python3
"""Build the static website from the LaTeX lecture notes.

    python tools/build_site.py            # writes index.html, lecture1.html, lecture2.html

Requires pandoc (>= 2.x). The LaTeX sources in notes/ are the single source of
truth; the HTML pages are generated and should not be edited by hand.

Interactive figures: a comment line `% WEBFIG: <name>` in a lecture file marks
where a widget goes on the web page (it is invisible in the PDF). The PDF-only
figure `figures/ridge_gamma3.pdf` is replaced by its interactive version.
"""

import argparse
import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
REPO_URL = "https://github.com/LianghongMo/LLM-note"

LECTURES = [
    {
        "num": 1,
        "tex": "lecture1.tex",
        "page": "lecture1.html",
        "abstract": (
            "Neural networks as parametrized families of functions, and the three "
            "questions the notes are built around: what simple theory emerges at "
            "large scale, what features training discovers, and how a diffusion "
            "model learns and samples an entire distribution."
        ),
        "formula": r"\text{Architecture}+\text{Data}+\text{Optimization}\;\Longrightarrow\;f_{\theta}",
    },
    {
        "num": 2,
        "tex": "lecture2.tex",
        "page": "lecture2.html",
        "abstract": (
            "Three worked scaling limits, each controlled by a single dimensionless "
            "ratio: depth over width in deep linear networks, the residual scale in "
            "deep ResNets, and parameters over data in ridge regression."
        ),
        "formula": r"\xi=L/N,\qquad s_L\sim L^{-1/2},\qquad \gamma=d/n",
    },
]

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans+Condensed:wght@500;600;700"
    "&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400"
    "&display=swap"
)
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"

# ---------------------------------------------------------------------------
# Interactive figures (static markup; behaviour lives in assets/figures.js)
# ---------------------------------------------------------------------------


def slider(fid, label, lo, hi, step, value, out):
    return f"""
      <div class="control">
        <label for="{fid}">{label}</label>
        <output id="{fid}-out" for="{fid}">{out}</output>
        <input type="range" id="{fid}" min="{lo}" max="{hi}" step="{step}" value="{value}">
      </div>"""


WIDGETS = {
    "diffusion": lambda: f"""
<figure class="widget" id="fig-diffusion" data-widget="diffusion">
  <header class="widget-head">
    <p class="kicker">Interactive figure</p>
    <h4 class="widget-title">Noising and denoising a one-dimensional distribution</h4>
  </header>
  <div class="controls">{slider("diff-t", r"noise time \(t\)", 0, 1, 0.005, 0.2, "0.10")}
    <div class="buttons">
      <button type="button" class="btn primary" id="diff-run">Generate from noise</button>
      <button type="button" class="btn" id="diff-reset">Resample data</button>
    </div>
  </div>
  <div class="panels stack">
    <canvas class="plot tall" id="diff-density" role="img" aria-label="Histogram of samples and the density rho_t"></canvas>
    <canvas class="plot short" id="diff-score" role="img" aria-label="Score function s_t(x)"></canvas>
  </div>
  <p class="readout" id="diff-readout" aria-live="polite"></p>
  <figcaption>
    Data distribution \\(\\rho_0\\): a three-component Gaussian mixture. Forward
    process: Ornstein&ndash;Uhlenbeck, \\(dx=-x\\,dt+\\sqrt2\\,dW_t\\), so
    \\(\\alpha_t=e^{{-t}}\\) and \\(\\sigma_t^2=1-e^{{-2t}}\\). Dragging \\(t\\) noises
    the same 5,000 samples through \\(x_t=\\alpha_t x_0+\\sigma_t\\epsilon\\).
    <em>Generate from noise</em> draws fresh samples from \\(\\mathcal N(0,1)\\) and
    integrates the probability-flow ODE \\(dx/dt=-x-s_t(x)\\) backwards from
    \\(t=2.5\\) with the exact score. At every \\(t\\) the histogram follows
    \\(\\rho_t\\) (curve), and at \\(t=0\\) it recovers the data distribution.
  </figcaption>
</figure>""",
    "deeplinear": lambda: f"""
<figure class="widget" id="fig-deeplinear" data-widget="deeplinear">
  <header class="widget-head">
    <p class="kicker">Interactive figure</p>
    <h4 class="widget-title">Log-normal norms in a deep linear network</h4>
  </header>
  <div class="controls">{slider("dl-n", r"width \(N\)", 0, 5, 1, 3, "32")}{slider("dl-l", r"depth \(L\)", 1, 256, 1, 32, "32")}
    <div class="buttons">
      <button type="button" class="btn" id="dl-resample">Resample networks</button>
    </div>
  </div>
  <div class="panels stack">
    <canvas class="plot tall" id="dl-hist" role="img" aria-label="Histogram of the log norm ratio with the Gaussian prediction"></canvas>
  </div>
  <p class="readout" id="dl-readout" aria-live="polite"></p>
  <figcaption>
    Histogram of \\(\\log(\\|z^{{(L)}}\\|/\\|x\\|)\\) over independent random networks
    with \\(W^{{(\\ell)}}_{{ij}}\\sim\\mathcal N(0,1/N)\\). Curve: the double-scaling
    prediction \\(\\mathcal N(-\\xi/2,\\,\\xi/2)\\) with \\(\\xi=L/N\\). For a fresh
    Gaussian matrix \\(Wv\\sim\\mathcal N(0,\\|v\\|^2I/N)\\) exactly, so each layer
    costs \\(N\\) Gaussian draws instead of \\(N^2\\). Scale \\(L\\) and \\(N\\) together
    to stay at fixed \\(\\xi\\); at fixed \\(N\\), growing \\(L\\) drags the typical
    norm to zero while \\(\\mathbb E\\|z^{{(L)}}\\|^2\\) stays equal to \\(\\|x\\|^2\\).
  </figcaption>
</figure>""",
    "resnet": lambda: f"""
<figure class="widget" id="fig-resnet" data-widget="resnet">
  <header class="widget-head">
    <p class="kicker">Interactive figure</p>
    <h4 class="widget-title">Three depth regimes of a random ResNet</h4>
  </header>
  <div class="controls">{slider("rn-beta", r"exponent \(\beta\) in \(s_L=L^{-\beta}\)", 20, 100, 5, 50, "0.50")}{slider("rn-l", r"depth \(L\)", 0, 5, 1, 2, "100")}
    <div class="buttons">
      <button type="button" class="btn" id="rn-resample">Resample networks</button>
    </div>
  </div>
  <div class="panels stack">
    <canvas class="plot tall" id="rn-paths" role="img" aria-label="Log squared norm along depth for random ResNets"></canvas>
  </div>
  <p class="readout" id="rn-readout" aria-live="polite"></p>
  <figcaption>
    Forty random linear ResNets \\(z^{{(\\ell+1)}}=(I+s_LW^{{(\\ell)}})z^{{(\\ell)}}\\)
    of width \\(N=32\\) with iid \\(W^{{(\\ell)}}_{{ij}}\\sim\\mathcal N(0,1/N)\\). Thin
    lines: \\(\\log(\\|z^{{(\\ell)}}\\|^2/\\|x\\|^2)\\) against relative depth
    \\(\\ell/L\\). Dashed: \\(\\log\\mathbb E\\|z^{{(\\ell)}}\\|^2=\\ell\\log(1+s_L^2)\\).
    At \\(\\beta=1/2\\) the picture stops depending on \\(L\\): this is the Neural SDE
    limit. Below it the norms explode with depth; above it the network
    approaches the identity.
  </figcaption>
</figure>""",
    "ridge": lambda: f"""
<figure class="widget" id="fig-ridge" data-widget="ridge">
  <header class="widget-head">
    <p class="kicker">Interactive figure &middot; Figure 2.1 in the PDF</p>
    <h4 class="widget-title">Marchenko&ndash;Pastur spectrum and the ridge risk</h4>
  </header>
  <div class="controls">{slider("rr-gamma", r"aspect ratio \(\gamma=d/n\)", 0.1, 5, 0.05, 3, "3.00")}{slider("rr-snr", r"\(\mathrm{SNR}=r^2/\sigma_\epsilon^2\)", -2, 3, 0.05, 0, "1.00")}
  </div>
  <div class="panels pair">
    <canvas class="plot tall" id="rr-mp" role="img" aria-label="Marchenko-Pastur density"></canvas>
    <canvas class="plot tall" id="rr-risk" role="img" aria-label="Ridge risk as a function of lambda"></canvas>
  </div>
  <p class="readout" id="rr-readout" aria-live="polite"></p>
  <figcaption>
    Left: spectrum of \\(S=X^\\top X/n\\) at aspect ratio \\(\\gamma=d/n\\); for
    \\(\\gamma>1\\) an atom of mass \\(1-1/\\gamma\\) sits at zero. Right: the
    asymptotic excess risk \\(R(\\lambda)\\) split into bias and variance, with
    \\(\\sigma_\\epsilon^2=1\\) and \\(r^2=\\mathrm{{SNR}}\\). The minimum always sits at
    \\(\\lambda_*=\\gamma/\\mathrm{{SNR}}\\). The defaults \\(\\gamma=3\\),
    \\(\\mathrm{{SNR}}=1\\) reproduce the PDF figure: \\(\\lambda_*=3\\),
    \\(R(\\lambda_*)=0.847\\).
  </figcaption>
</figure>""",
}

# ---------------------------------------------------------------------------
# LaTeX -> HTML
# ---------------------------------------------------------------------------


def pandoc(tex: str) -> str:
    tex = re.sub(r"^% WEBFIG: (\w+)\s*$", r"\\begin{webfig}\1\\end{webfig}", tex, flags=re.M)
    res = subprocess.run(
        ["pandoc", "-f", "latex", "-t", "html5", "--mathjax", "--section-divs"],
        input=tex, capture_output=True, text=True, check=True,
    )
    return res.stdout


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def convert(lec: dict) -> dict:
    body = pandoc((NOTES / lec["tex"]).read_text())

    title = re.search(r"<h1>(.*?)</h1>", body, re.S).group(1).strip()
    body = re.sub(r'^<section id="[^"]*" class="level1">\s*<h1>.*?</h1>\s*', "", body, count=1, flags=re.S)
    # pandoc closes the chapter section just before the footnotes (or at the end)
    body = re.sub(r"</section>\s*(?=<section class=\"footnotes\"|$)", "", body, count=1)

    keyideas = re.search(r'<div class="keyideas">(.*?)</div>', body, re.S).group(1)
    body = re.sub(r'<div class="keyideas">.*?</div>\s*', "", body, count=1, flags=re.S)

    # number headings like the PDF (chapter.section.subsection.subsubsection)
    counters = [lec["num"], 0, 0, 0]
    toc = []

    def number(m):
        sid, level, inner = m.group(1), int(m.group(2)), m.group(3).strip()
        if level == 5:  # LaTeX \paragraph: unnumbered run-in heading
            return f'<section id="{sid}" class="level5">\n<h5 class="runin">{inner}</h5>'
        depth = level - 1  # h2 -> 1, h3 -> 2, h4 -> 3
        counters[depth] += 1
        for i in range(depth + 1, 4):
            counters[i] = 0
        num = ".".join(str(c) for c in counters[: depth + 1])
        if level <= 3:
            toc.append((level, sid, num, inner))
        return (
            f'<section id="{sid}" class="level{level}">\n'
            f'<h{level}><span class="secnum">{num}</span> {inner}'
            f'<a class="anchor" href="#{sid}" aria-label="Link to this section">#</a></h{level}>'
        )

    body = re.sub(r'<section id="([^"]+)" class="level(\d)">\s*<h\d>(.*?)</h\d>', number, body, flags=re.S)

    # interactive figures
    body = re.sub(r'<div class="webfig">\s*<p>(\w+)</p>\s*</div>', lambda m: WIDGETS[m.group(1)](), body)
    body, n = re.subn(
        r'<figure>\s*<embed src="figures/ridge_gamma3\.pdf"[^>]*>.*?</figure>',
        lambda m: WIDGETS["ridge"](), body, flags=re.S,
    )
    if lec["num"] == 2:
        assert n == 1, "ridge figure not found"

    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body, flags=re.S)
    body = body.replace('<section class="footnotes" role="doc-endnotes">\n<hr />',
                        '<section class="footnotes" role="doc-endnotes">\n<h2 class="notes-title">Notes</h2>')

    return {**lec, "title": title, "keyideas": keyideas, "body": body, "toc": toc}


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------


def head(title: str, description: str, full: bool) -> str:
    meta = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        if full else ""
    )
    return f"""{meta}<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="assets/site.css">
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script', 'style', 'textarea', 'code', 'canvas'] }}
}};
</script>
<script defer src="{MATHJAX}"></script>
<script defer src="assets/site.js"></script>
{"</head>" + chr(10) + "<body>" if full else ""}
"""


def foot(full: bool) -> str:
    return "</body>\n</html>\n" if full else ""


def topbar(active: str, pdf_href: str) -> str:
    def link(href, label, key):
        cur = ' aria-current="page"' if key == active else ""
        return f'<a href="{href}"{cur}>{label}</a>'

    return f"""<header class="topbar">
  <a class="wordmark" href="index.html">LLM-note</a>
  <nav aria-label="Site">
    {link("lecture1.html", "Lecture 1", "1")}
    {link("lecture2.html", "Lecture 2", "2")}
    <a href="{pdf_href}">PDF</a>
    <a href="{REPO_URL}">Source</a>
  </nav>
</header>"""


def toc_html(toc) -> str:
    out, open_sub = [], False
    for level, sid, num, inner in toc:
        if level == 2:
            if open_sub:
                out.append("</ol></li>")
                open_sub = False
            elif out:
                out.append("</li>")
            out.append(f'<li><a href="#{sid}"><span class="secnum">{num}</span> {inner}</a>')
        else:
            if not open_sub:
                out.append("<ol>")
                open_sub = True
            out.append(f'<li><a href="#{sid}"><span class="secnum">{num}</span> {inner}</a></li>')
    out.append("</ol></li>" if open_sub else "</li>")
    return "<ol>" + "\n".join(out) + "</ol>"


def lecture_page(lec, lectures, pdf_href) -> str:
    n = lec["num"]
    prev_l = next((l for l in lectures if l["num"] == n - 1), None)
    next_l = next((l for l in lectures if l["num"] == n + 1), None)
    pager = []
    if prev_l:
        pager.append(f'<a class="pager-prev" href="{prev_l["page"]}"><span>Previous</span>Lecture {prev_l["num"]}: {prev_l["title"]}</a>')
    if next_l:
        pager.append(f'<a class="pager-next" href="{next_l["page"]}"><span>Next</span>Lecture {next_l["num"]}: {next_l["title"]}</a>')
    toc = toc_html(lec["toc"])
    return (
        head(f'Lecture {n}: {strip_tags(lec["title"])} · LLM-note', lec["abstract"], True)
        + topbar(str(n), pdf_href)
        + f"""
<div class="layout">
  <aside class="toc-rail" aria-label="Contents">
    <p class="kicker">Lecture {n} contents</p>
    <nav class="toc">{toc}</nav>
  </aside>
  <main class="article" id="top">
    <header class="lecture-head">
      <p class="kicker">Lecture {n}</p>
      <h1>{lec["title"]}</h1>
      <p class="lede">{lec["abstract"]}</p>
      <p class="head-formula">\\({lec["formula"]}\\)</p>
    </header>
    <aside class="keyideas" aria-label="Key ideas">
      <p class="kicker">Key ideas</p>
      {lec["keyideas"]}
    </aside>
    <details class="toc-inline">
      <summary>Contents</summary>
      <nav class="toc">{toc}</nav>
    </details>
    <div class="prose">
{lec["body"]}
    </div>
    <nav class="pager" aria-label="Lectures">{"".join(pager)}</nav>
    <footer class="page-foot">
      Generated from <a href="{REPO_URL}/blob/main/notes/{lec["tex"]}">notes/{lec["tex"]}</a>
      &middot; <a href="{pdf_href}">PDF version</a>
    </footer>
  </main>
</div>
<script defer src="assets/figures.js"></script>
"""
        + foot(True)
    )


def index_page(lectures, pdf_href, full=True) -> str:
    cards = []
    for lec in lectures:
        items = "".join(
            f'<li><span class="secnum">{num}</span> {inner}</li>'
            for level, sid, num, inner in lec["toc"] if level == 2
        )
        cards.append(f"""
    <article class="lecture-card">
      <p class="kicker">Lecture {lec["num"]}</p>
      <h3><a href="{lec["page"]}">{lec["title"]}</a></h3>
      <p>{lec["abstract"]}</p>
      <p class="card-formula">\\[{lec["formula"]}\\]</p>
      <ol class="card-sections">{items}</ol>
      <a class="btn primary" href="{lec["page"]}">Read lecture {lec["num"]}</a>
    </article>""")

    return (
        head("LLM-note", "Lecture notes on the theory of neural networks: scaling limits, feature learning, and diffusion models.", full)
        + topbar("home", pdf_href)
        + f"""
<main class="home">
  <section class="hero">
    <h1>LLM-note</h1>
    <p class="lede">Lecture notes on the theory of neural networks: scaling limits,
    feature learning, and diffusion models, read through the lens of statistical
    physics. Large networks are treated the way a physicist treats a many-body
    system: look for the few macroscopic variables and dimensionless ratios that
    survive the limit.</p>
    <p class="meta">2 lectures &middot; 4 interactive figures &middot;
      <a href="{pdf_href}">PDF</a> &middot; <a href="{REPO_URL}">LaTeX source</a></p>
  </section>

  <section class="lectures" aria-label="Lectures">{"".join(cards)}
  </section>

  <section class="ratios">
    <h2>The recurring idea</h2>
    <p>Each example in Lecture 2 is controlled by one ratio of large numbers, not by
    the sizes separately. Fix the ratio, send the sizes to infinity, and a
    simple limiting law appears.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>System</th><th>Ratio held fixed</th><th>Limiting law</th></tr></thead>
        <tbody>
          <tr><td>Deep linear network</td><td>\\(\\xi=L/N\\)</td><td>\\(\\log(\\|z^{{(L)}}\\|/\\|x\\|)\\to\\mathcal N(-\\xi/2,\\,\\xi/2)\\)</td></tr>
          <tr><td>ResNet, iid layers</td><td>\\(L\\,s_L^2\\)</td><td>Neural SDE \\(dz_t=G(z_t)\\,dB_t\\) at \\(s_L\\sim L^{{-1/2}}\\)</td></tr>
          <tr><td>Ridge regression</td><td>\\(\\gamma=d/n\\)</td><td>Marchenko&ndash;Pastur law, \\(\\lambda_*=\\gamma/\\mathrm{{SNR}}\\)</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <footer class="page-foot">
    Generated from the LaTeX sources in <a href="{REPO_URL}/tree/main/notes">notes/</a>
    with <code>python tools/build_site.py</code>.
  </footer>
</main>
"""
        + foot(full)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT, help="output directory (default: repo root)")
    ap.add_argument("--fragment-index", action="store_true",
                    help="write index.html without <html>/<head>/<body> wrappers")
    ap.add_argument("--pdf-href", default="notes/LLM-note.pdf")
    args = ap.parse_args()

    lectures = [convert(l) for l in LECTURES]
    args.out.mkdir(parents=True, exist_ok=True)
    for lec in lectures:
        (args.out / lec["page"]).write_text(lecture_page(lec, lectures, args.pdf_href))
    (args.out / "index.html").write_text(index_page(lectures, args.pdf_href, full=not args.fragment_index))
    print("wrote", ", ".join(["index.html"] + [l["page"] for l in lectures]), "to", args.out)


if __name__ == "__main__":
    main()
