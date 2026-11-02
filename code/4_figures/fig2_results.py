"""Fig. 2 (results), full width 7.0 in. Reads study_metrics.csv and study_reassuring_header.csv only."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
matplotlib.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
                            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
M = ["Nemotron-3.5-VL-120B", "Gemma-4-31B", "Llama-3.2-90B-V", "Llama-3.2-11B-V", "MedGemma-1.5-4B", "Muse-Glimmer-30B"]
LAB = {"Nemotron-3.5-VL-120B": "Nemotron-3.5-VL", "Gemma-4-31B": "Gemma-4-31B", "Llama-3.2-90B-V": "Llama-3.2-90B",
       "Llama-3.2-11B-V": "Llama-3.2-11B", "MedGemma-1.5-4B": "MedGemma-1.5", "Muse-Glimmer-30B": "Muse-Glimmer"}
COL = dict(zip(M, ["#009E73", "#E69F00", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7"]))   # Okabe-Ito
CONDS = ["Header only", "Image only (aligned)", "Image only (native)", "Image + correct header",
         "Image + conflicting header (aligned)", "Image + conflicting header (native)"]
CL = ["Hdr\nonly", "Img\nal.", "Img\nnat.", "Img +\ntrue", "Img +\nfalse\nal.", "Img +\nfalse\nnat."]

def build(out="fig2_results"):
    t = pd.read_csv("study_metrics.csv"); dom = pd.read_csv("study_reassuring_header.csv")
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.95], wspace=0.55, left=0.065, right=0.985, top=0.74, bottom=0.2)
    plotted = []
    for k, (col, ylab, ttl, hl) in enumerate([
            ("SFR", "Silent failure (%)", "Defective pairs admitted", "false header:\n'pair is complete'"),
            ("FRR", "False refusal (%)", "Admissible pairs refused", "false header:\n'B is truncated'")]):
        ax = fig.add_axes([[0.07, 0.375][k], 0.2, 0.255, 0.54])
        ax.axvspan(3.5, 5.5, color="0.93", lw=0, zorder=0)
        ax.text(4.5, 101, hl, ha="center", va="bottom", fontsize=6.5, color="0.35", linespacing=1.0)
        for mi, m in enumerate(M):
            for ci, c in enumerate(CONDS):
                r = t[(t.model == m) & (t.condition == c)]
                if not len(r): continue
                r = r.iloc[0]; x = ci + (mi - 2.5) * 0.13; p = 100 * r[col]
                ax.errorbar(x, p, yerr=[[p - 100 * r[col + "_lo"]], [100 * r[col + "_hi"] - p]], fmt="o", ms=2.8,
                            color=COL[m], elinewidth=0.7, capsize=0, mew=0.7,
                            mfc="white" if r.n_patients < 30 else COL[m], label=LAB[m] if (k == 0 and ci == 1) else None)
                plotted.append((f"{'ab'[k]}", m, c, col, round(p, 1), int(r.n_patients)))
        ax.set_xticks(range(6)); ax.set_xticklabels(CL); ax.set_xlim(-0.5, 5.5); ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", pad=20, fontweight="bold")
        ax.text(-0.22, 1.2, "ab"[k], transform=ax.transAxes, fontweight="bold", fontsize=9)
    # c: paired header override
    ax = fig.add_axes([0.80, 0.2, 0.185, 0.54])
    rows = [(m, s) for m in M for s in ["native", "aligned"] if len(dom[(dom.model == m) & (dom["style"] == s)])]
    for i, (m, s) in enumerate(rows):
        r = dom[(dom.model == m) & (dom["style"] == s)].iloc[0]
        a, b = 100 * r.SFR_image_only, 100 * r.SFR_reassuring_header
        ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=COL[m], lw=0.9,
                    shrinkA=2, shrinkB=2, mutation_scale=6)) if abs(b - a) > 4 else None
        ax.plot(a, i, "o", ms=3.2, mfc="white", mec=COL[m], mew=0.8, zorder=3)
        ax.plot(b, i, "o", ms=3.2, color=COL[m], zorder=3)
        plotted.append(("c", m, s, "SFR img->hdr", round(a, 1), round(b, 1), int(r.n_patients)))
    ax.set_yticks(range(len(rows)))
    SH = {"Nemotron-3.5-VL-120B": "Nemotron", "Gemma-4-31B": "Gemma-4", "Llama-3.2-90B-V": "Llama-90B",
          "Llama-3.2-11B-V": "Llama-11B", "MedGemma-1.5-4B": "MedGemma", "Muse-Glimmer-30B": "Muse"}
    ax.set_yticklabels([f"{SH[m]} · {s[:3]}" + ("\u2020" if dom[(dom.model == m) & (dom['style'] == s)].n_patients.iloc[0] < 60 else "")
                        for m, s in rows])
    ax.invert_yaxis(); ax.set_xlim(-3, 103); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Silent failure (%)\n\u25CB image only  \u2192  \u25CF + false header")
    ax.set_title("Header override (paired)", loc="left", pad=20, fontweight="bold")
    ax.text(-0.6, 1.2, "c", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.tick_params(axis="y", length=0)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6,
               handletextpad=0.1, columnspacing=0.9)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=600)
    # checks
    rnd = fig.canvas.get_renderer()
    T = [(x, x.get_window_extent(rnd)) for x in fig.findobj(matplotlib.text.Text) if x.get_text().strip() and x.get_visible()]
    ov = [(a.get_text()[:18], b.get_text()[:18]) for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    off = [x.get_text()[:18] for x, bb in T if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1]
    small = sorted({x.get_fontsize() for x, _ in T if x.get_fontsize() < 6.5})
    open("fig2_plotted_values.csv", "w").write("\n".join(",".join(map(str, p)) for p in plotted))
    return ov, off, small, len(plotted)

if __name__ == "__main__":
    print(build())
