#!/usr/bin/env python3
"""Generate TikZ/PGFPlots figure fragments for the Qwen3.5 MLC paper.

The script intentionally depends only on the Python standard library so the
paper source can be regenerated in a minimal development environment.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def write(name: str, text: str) -> None:
    (OUT / f"{name}.tex").write_text(text.strip() + "\n", encoding="utf-8")


def throughput_parity() -> None:
    write(
        "throughput_parity",
        r"""
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=\linewidth,
    height=0.46\linewidth,
    ymin=0,
    ymax=500,
    ylabel={tokens/s},
    symbolic x coords={MLC gen.,MLC decode,vLLM gen.},
    xtick=data,
    nodes near coords,
    nodes near coords align={vertical},
    bar width=20pt,
    enlarge x limits=0.25,
    grid=major,
    major grid style={draw=black!10},
    every node near coord/.append style={font=\scriptsize},
]
\addplot+[fill=blue!55, draw=blue!70] coordinates {(MLC gen.,414.23)};
\addplot+[fill=green!55!black, draw=green!50!black] coordinates {(MLC decode,446.09)};
\addplot+[fill=red!55, draw=red!70] coordinates {(vLLM gen.,413.30)};
\end{axis}
\end{tikzpicture}
""",
    )


def optimization_timeline() -> None:
    write(
        "optimization_timeline",
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=0.50\linewidth,
    ymin=395,
    ymax=450,
    ylabel={tokens/s},
    symbolic x coords={packed GDN,stable input,cross graph,append+meta,relaxed meta},
    xtick=data,
    x tick label style={rotate=25, anchor=east, font=\scriptsize},
    grid=major,
    major grid style={draw=black!10},
    legend style={at={(0.03,0.97)},anchor=north west,font=\scriptsize},
]
\addplot+[mark=*, thick, blue] coordinates {
    (packed GDN,403.10)
    (stable input,406.68)
    (cross graph,411.02)
    (append+meta,410.27)
    (relaxed meta,414.23)
};
\addlegendentry{generated}
\addplot+[mark=square*, thick, green!50!black] coordinates {
    (packed GDN,433.26)
    (stable input,437.17)
    (cross graph,441.83)
    (append+meta,441.48)
    (relaxed meta,446.09)
};
\addlegendentry{effective decode}
\addplot+[red, dashed, thick] coordinates {
    (packed GDN,413.30)
    (relaxed meta,413.30)
};
\addlegendentry{vLLM generated}
\end{axis}
\end{tikzpicture}
""",
    )


def kernel_breakdown() -> None:
    write(
        "kernel_breakdown",
        r"""
\begin{tikzpicture}
\begin{axis}[
    xbar,
    width=\linewidth,
    height=0.55\linewidth,
    xmin=0,
    xmax=1850,
    xlabel={microseconds/token},
    symbolic y coords={KV cache,FlashInfer,GDN,elementwise,other,GEMV/GEMM},
    ytick=data,
    nodes near coords,
    nodes near coords align={horizontal},
    every node near coord/.append style={font=\scriptsize},
    grid=major,
    major grid style={draw=black!10},
]
\addplot+[fill=blue!55, draw=blue!70] coordinates {
    (7.533,KV cache)
    (51.158,FlashInfer)
    (89.634,GDN)
    (113.838,elementwise)
    (255.254,other)
    (1732.960,GEMV/GEMM)
};
\end{axis}
\end{tikzpicture}
""",
    )


def runtime_visibility() -> None:
    write(
        "runtime_visibility",
        r"""
\begin{tikzpicture}
\begin{axis}[
    width=\linewidth,
    height=0.48\linewidth,
    ymin=0,
    ymax=450,
    ylabel={launches/token},
    symbolic x coords={no graph,CUDA graph},
    xtick=data,
    axis y line*=left,
    grid=major,
    major grid style={draw=black!10},
    legend style={at={(0.02,0.98)},anchor=north west,font=\scriptsize},
]
\addplot+[ybar, bar width=16pt, fill=blue!55, draw=blue!70] coordinates {
    (no graph,415.0)
    (CUDA graph,1.0)
};
\addlegendentry{launches/token}
\end{axis}
\begin{axis}[
    width=\linewidth,
    height=0.48\linewidth,
    ymin=0,
    ymax=2400,
    axis y line*=right,
    axis x line=none,
    ylabel={microseconds/token},
    symbolic x coords={no graph,CUDA graph},
    xtick=data,
    legend style={at={(0.98,0.98)},anchor=north east,font=\scriptsize},
]
\addplot+[mark=square*, thick, orange] coordinates {
    (no graph,2250.376)
    (CUDA graph,0.978)
};
\addlegendentry{visible kernels}
\addplot+[mark=*, thick, green!50!black] coordinates {
    (no graph,0.0)
    (CUDA graph,2189.405)
};
\addlegendentry{graph trace}
\end{axis}
\end{tikzpicture}
""",
    )


def memory_usage() -> None:
    write(
        "memory_usage",
        r"""
\begin{tikzpicture}
\begin{axis}[
    ybar stacked,
    width=\linewidth,
    height=0.36\linewidth,
    ymin=0,
    ymax=2500,
    ylabel={MB},
    symbolic x coords={single GPU},
    xtick=data,
    bar width=32pt,
    legend style={at={(0.5,-0.25)},anchor=north,legend columns=3,font=\scriptsize},
    grid=major,
    major grid style={draw=black!10},
]
\addplot+[fill=blue!55, draw=blue!70] coordinates {(single GPU,1626.943)};
\addplot+[fill=orange!70, draw=orange!80!black] coordinates {(single GPU,655.335)};
\addplot+[fill=green!55!black, draw=green!50!black] coordinates {(single GPU,75.150)};
\legend{parameters,temp buffer,KV cache}
\end{axis}
\end{tikzpicture}
""",
    )


def model_structure() -> None:
    full = {3, 7, 11, 15, 19, 23}
    nodes = []
    for layer in range(24):
        color = "red!60" if layer in full else "green!55!black"
        label = "F" if layer in full else "G"
        nodes.append(
            rf"\node[layer, fill={color}] at ({layer * 0.42},0) {{\tiny {label}{layer}}};"
        )
    write(
        "model_structure",
        r"""
\begin{tikzpicture}[
    layer/.style={rectangle, rounded corners=1pt, minimum width=0.38cm, minimum height=0.36cm, text=white, inner sep=1pt}
]
"""
        + "\n".join(nodes)
        + r"""
\node[anchor=west] at (0,-0.65) {\scriptsize G: Gated DeltaNet / linear attention};
\node[anchor=west] at (5.0,-0.65) {\scriptsize F: full attention};
\end{tikzpicture}
""",
    )


def system_pipeline() -> None:
    write(
        "system_pipeline",
        r"""
\begin{tikzpicture}[
    box/.style={draw, rounded corners=2pt, align=center, minimum height=0.72cm, minimum width=2.3cm, fill=blue!6},
    opt/.style={draw, rounded corners=2pt, align=center, minimum height=0.72cm, minimum width=2.3cm, fill=green!8},
    arrow/.style={-Latex, thick}
]
\node[box] (hf) {HF checkpoint\\tokenizer/config};
\node[box, right=0.55cm of hf] (config) {MLC config\\Qwen3.5 parser};
\node[opt, right=0.55cm of config] (model) {Relax model\\vision + language};
\node[opt, right=0.55cm of model] (passes) {compile passes\\fusion + graph};
\node[box, right=0.55cm of passes] (runtime) {.so + tensor\\cache shards};
\node[box, below=0.6cm of runtime] (serve) {image runner\\MLC engine};
\node[box, left=0.55cm of serve] (compare) {vLLM / HF\\baselines};
\draw[arrow] (hf) -- (config);
\draw[arrow] (config) -- (model);
\draw[arrow] (model) -- (passes);
\draw[arrow] (passes) -- (runtime);
\draw[arrow] (runtime) -- (serve);
\draw[arrow] (serve) -- (compare);
\end{tikzpicture}
""",
    )


def image_path() -> None:
    write(
        "image_path",
        r"""
\begin{tikzpicture}[
    box/.style={draw, rounded corners=2pt, align=center, minimum height=0.68cm, minimum width=2.0cm, fill=purple!6},
    arrow/.style={-Latex, thick}
]
\node[box] (img) {image\\512$\times$512};
\node[box, right=0.35cm of img] (proc) {processor\\grid 1$\times$32$\times$32};
\node[box, right=0.35cm of proc] (vision) {vision tower\\1024 patches};
\node[box, right=0.35cm of vision] (merge) {spatial merge\\256 image tokens};
\node[box, right=0.35cm of merge] (lm) {mrope prefill\\+ decode};
\draw[arrow] (img) -- (proc);
\draw[arrow] (proc) -- (vision);
\draw[arrow] (vision) -- (merge);
\draw[arrow] (merge) -- (lm);
\end{tikzpicture}
""",
    )


def main() -> None:
    throughput_parity()
    optimization_timeline()
    kernel_breakdown()
    runtime_visibility()
    memory_usage()
    model_structure()
    system_pipeline()
    image_path()


if __name__ == "__main__":
    main()
