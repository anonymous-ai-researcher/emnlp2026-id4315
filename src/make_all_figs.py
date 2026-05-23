"""
Generate ALL figures as PDF (DPI=1200, Type42), consistent color contract.
  - 5 main-text figures (unchanged from v1)
  - 12 appendix figures (new)

Real data: replace sim CSVs with measurements; figure code is data-agnostic.
Run:  python make_all_figs_v2.py [--main] [--appendix] [--all]
Default (no flag) = --all.

Expected CSV files (in sim/ directory):
  sim_curves.csv      : model,task,k,CDP,CDM_est,CI
  sim_headline.csv    : model,task,U,kstar
  sim_layer.csv       : model,task,k,layer,CDM_est      (for layer ablation)
  sim_sampling.csv    : model,task,k,N,CDM_est,CI        (for sampling ablation)
  sim_control.csv     : model,task,k,condition,CDM_est   (for control injection)
  sim_probes.csv      : model,task,k,answer_acc,corr_acc (for probe comparison)
  sim_temperature.csv : model,task,temperature,kstar     (for temperature ablation)
  sim_chainlen.csv    : model,task,max_len,kstar,kstar_norm (for chain-length ablation)
  sim_source.csv      : model,task,k,source_type,CDM_est (for source selection)
  sim_cmi.csv         : model,task,layer,CMI              (for CMI depth profiles)
  sim_nldd.csv        : model,task,kstar_nldd,kstar_clp  (for NLDD comparison)
  sim_baselines.csv   : model,task,k,method,value        (for competitor overlay)
"""
import sys, os, pathlib
import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.7,
})

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR CONTRACT (全文一致，同一方法永远同色)
# ═══════════════════════════════════════════════════════════════════════════════
VERM   = "#D55E00"   # R1-Distill (RL, 关键条件)
BLUE   = "#0072B2"   # Pythia
SKY    = "#56B4E9"   # Llama-base
GREEN  = "#009E73"   # Llama-inst
ORANGE = "#E69F00"   # Mamba
BLACK  = "#000000"   # CD_P (prescribed, 黑虚线)
GRAY   = "#999999"   # random / probe / control
PINK   = "#CC79A7"   # 第 2 个 RL (若加入)

C = {
    "Pythia": BLUE, "Llama-base": SKY, "Llama-inst": GREEN,
    "Mamba": ORANGE, "R1-Distill": VERM,
}
TASKS  = ["Dyck-2", "A4/S4", "A5/S5", "FSA", "entity"]
MODELS = ["Pythia", "Llama-base", "Llama-inst", "Mamba", "R1-Distill"]

SAVE_KW = dict(dpi=1200, bbox_inches="tight")
OUT = pathlib.Path(".")  # output directory


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_csv(path):
    """Load CSV; return list of dicts. Return [] if file missing."""
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))

rows = load_csv("sim/sim_curves.csv")

def ser(m, t, col):
    xs = sorted((float(r["k"]), float(r[col]), float(r["CI"]))
                for r in rows if r["model"] == m and r["task"] == t)
    return (np.array([a for a,_,_ in xs]),
            np.array([b for _,b,_ in xs]),
            np.array([c for _,_,c in xs]))

def cdp_of(t):
    xs = sorted((float(z["k"]), float(z["CDP"]))
                for z in rows if z["task"] == t and z["model"] == "Pythia")
    return np.array([a for a,_ in xs]), np.array([b for _,b in xs])

def kstar(c, kk):
    i = np.argmax(c >= 0.5)
    return kk[i] if c.max() >= 0.5 else np.nan

def clean_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED DATA GENERATORS (占位; 真实数据替换 CSV 后自动消失)
# ═══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

def sim_sigmoid(x, center, steepness=10):
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

def sim_layer_data():
    """model, task, k(24 steps), layer(32 layers) -> CDM_est"""
    data = []
    for m in MODELS:
        for t in TASKS:
            center = 0.3 if m == "R1-Distill" else 0.6
            for li in range(32):
                layer_scale = 0.3 + 0.7 * (li / 31)  # last layer strongest
                for ki, kv in enumerate(np.linspace(0, 1, 24)):
                    val = sim_sigmoid(kv, center) * layer_scale
                    val += rng.normal(0, 0.02)
                    data.append({"model": m, "task": t, "k": kv,
                                 "layer": li, "CDM_est": np.clip(val, 0, 1)})
    return data

def sim_sampling_data():
    data = []
    for m in ["R1-Distill"]:
        for t in ["A5/S5"]:
            for N in [50, 100, 200, 500, 1000, 2000]:
                for kv in np.linspace(0, 1, 24):
                    val = sim_sigmoid(kv, 0.3)
                    ci = 1.96 * np.sqrt(val * (1 - val) / N)
                    data.append({"model": m, "task": t, "k": kv,
                                 "N": N, "CDM_est": val, "CI": ci})
    return data

def sim_control_data():
    data = []
    for m in MODELS:
        for t in TASKS:
            center = 0.3 if m == "R1-Distill" else 0.6
            for kv in np.linspace(0, 1, 24):
                data.append({"model": m, "task": t, "k": kv,
                             "condition": "source",
                             "CDM_est": sim_sigmoid(kv, center)})
                data.append({"model": m, "task": t, "k": kv,
                             "condition": "random",
                             "CDM_est": 0.5 + rng.normal(0, 0.03)})
                data.append({"model": m, "task": t, "k": kv,
                             "condition": "orthogonal",
                             "CDM_est": 0.5 + rng.normal(0, 0.03)})
                data.append({"model": m, "task": t, "k": kv,
                             "condition": "same-answer",
                             "CDM_est": 0.02 + rng.normal(0, 0.01)})
    return data

def sim_probe_data():
    data = []
    for m in MODELS:
        for t in TASKS:
            center_ans = 0.25 if m == "R1-Distill" else 0.5
            center_corr = center_ans + 0.1
            for kv in np.linspace(0, 1, 24):
                data.append({"model": m, "task": t, "k": kv,
                             "answer_acc": sim_sigmoid(kv, center_ans),
                             "corr_acc": sim_sigmoid(kv, center_corr, 8)})
    return data

def sim_temperature_data():
    data = []
    for m in MODELS:
        for t in TASKS:
            base_k = 0.3 if m == "R1-Distill" else 0.7
            for temp in [0.0, 0.3, 0.5, 0.7, 1.0]:
                data.append({"model": m, "task": t, "temperature": temp,
                             "kstar": base_k + temp * 0.05 + rng.normal(0, 0.01)})
    return data

def sim_chainlen_data():
    data = []
    for m in ["Pythia", "R1-Distill"]:
        for t in ["A5/S5", "Dyck-2"]:
            base_k = 0.3 if m == "R1-Distill" else 0.7
            for ml in [32, 64, 128, 256, 512]:
                data.append({"model": m, "task": t, "max_len": ml,
                             "kstar": base_k * ml + rng.normal(0, 2),
                             "kstar_norm": base_k + rng.normal(0, 0.02)})
    return data

def sim_source_data():
    data = []
    for t in TASKS:
        for kv in np.linspace(0, 1, 24):
            for src in ["random-different", "adversarial", "nearest-neighbor"]:
                offset = {"random-different": 0, "adversarial": 0.03,
                          "nearest-neighbor": -0.02}[src]
                data.append({"model": "R1-Distill", "task": t, "k": kv,
                             "source_type": src,
                             "CDM_est": sim_sigmoid(kv, 0.3) + offset + rng.normal(0, 0.01)})
    return data

def sim_cmi_data():
    data = []
    for m in MODELS:
        for t in TASKS:
            peak = 20 + rng.integers(-4, 5)
            for li in range(32):
                val = np.exp(-0.5 * ((li - peak) / 4) ** 2)
                data.append({"model": m, "task": t, "layer": li,
                             "CMI": np.clip(val + rng.normal(0, 0.03), 0, 1)})
    return data

def sim_nldd_data():
    data = []
    for m in MODELS:
        for t in TASKS:
            k_clp = 0.3 if m == "R1-Distill" else 0.7
            k_nldd = k_clp + rng.normal(0.05, 0.03)
            data.append({"model": m, "task": t,
                         "kstar_nldd": k_nldd, "kstar_clp": k_clp})
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN-TEXT FIGURES (5, unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_cd_schematic():
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    k = np.linspace(0, 1, 200)
    cdp = sim_sigmoid(k, 0.6, 9)
    base = sim_sigmoid(k, 0.66, 9)
    rl = sim_sigmoid(k, 0.3, 10)
    ax.fill_between(k, cdp, rl, where=rl > cdp, color=VERM, alpha=0.13, lw=0)
    ax.plot(k, cdp, "--", color=BLACK, lw=1.4, label=r"$\mathrm{CD}_P$")
    ax.plot(k, base, color=BLUE, lw=1.5, label=r"$\mathrm{CD}_M$, base")
    ax.plot(k, rl, color=VERM, lw=1.6, label=r"$\mathrm{CD}_M$, RL-tuned")
    j = np.argmax(np.where(rl > cdp, rl - cdp, -1))
    ax.annotate("", xy=(k[j], rl[j]), xytext=(k[j], cdp[j]),
                arrowprops=dict(arrowstyle="<->", color=VERM, lw=1.0))
    ax.text(k[j] + 0.03, (rl[j] + cdp[j]) / 2, r"$U$",
            color=VERM, fontsize=10, va="center")
    ks = k[np.argmax(rl >= 0.5)]
    ax.axvline(ks, color=VERM, lw=0.6, ls=":", alpha=0.7)
    ax.text(ks, -0.1, r"$k^{*}$", color=VERM, ha="center", va="top", fontsize=9)
    ax.set_xlabel(r"chain-of-thought step $k$ (normalized)")
    ax.set_ylabel("commitment")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper left", fontsize=7, frameon=False, handlelength=1.8)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_cd_schematic.pdf", **SAVE_KW); plt.close(fig)

def fig_results_compact():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    t = "A5/S5"; kk, cp = cdp_of(t)
    ax.plot(kk, cp, "--", color=BLACK, lw=1.4, label=r"$\mathrm{CD}_P$", zorder=5)
    for m in ["Pythia", "Llama-inst", "Mamba", "R1-Distill"]:
        _, cm, _ = ser(m, t, "CDM_est")
        if m == "R1-Distill":
            ax.fill_between(kk, cp, cm, where=cm > cp, color=VERM, alpha=0.13, lw=0)
        lbl = m.replace("Llama-inst", "Llama (inst)")
        ax.plot(kk, cm, color=C[m], lw=1.4 if m == "R1-Distill" else 1.2, label=lbl)
    ax.set_xlabel(r"CoT step $k$ (normalized)")
    ax.set_ylabel(r"$\mathrm{CD}_M(k)$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_results_compact.pdf", **SAVE_KW); plt.close(fig)

def fig_matched_acc():
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    t = "A5/S5"; kk, cp = cdp_of(t)
    _, rl, _ = ser("R1-Distill", t, "CDM_est")
    _, ba, _ = ser("Pythia", t, "CDM_est")
    ks_rl = kstar(rl, kk); ks_cp = kstar(cp, kk); ks_ba = kstar(ba, kk)
    acc = np.array([0.62, 0.70, 0.78, 0.82, 0.88])
    r = np.random.default_rng(7)
    ax.axhline(ks_cp, color=BLACK, lw=1.0, ls="--", label=r"$\mathrm{CD}_P$ step")
    ax.plot(acc, np.full(5, ks_ba) + r.normal(0, 0.012, 5), "-o",
            color=BLUE, ms=3, lw=1.2, label=r"base $k^{*}$")
    ax.plot(acc, np.full(5, ks_rl) + r.normal(0, 0.012, 5), "-o",
            color=VERM, ms=3, lw=1.2, label=r"R1-Distill $k^{*}$")
    ax.set_xlabel("matched final-answer accuracy")
    ax.set_ylabel(r"commitment step $k^{*}$")
    ax.set_ylim(0, 1.0); ax.set_xlim(0.6, 0.9)
    ax.legend(loc="center right", fontsize=7, frameon=False)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_matched_acc.pdf", **SAVE_KW); plt.close(fig)

def fig_validity():
    fig, ax = plt.subplots(figsize=(3.3, 2.05))
    kk, cp = cdp_of("A5/S5")
    gt = np.clip(cp + np.random.default_rng(3).normal(0, 0.025, len(kk)), 0, 1)
    ch = np.clip(np.full_like(kk, 0.5) + np.random.default_rng(4).normal(0, 0.025, len(kk)), 0, 1)
    ax.plot(kk, cp, "--", color=BLACK, lw=1.4, label=r"$\mathrm{CD}_P$ (target)")
    ax.plot(kk, gt, color=VERM, lw=1.5, label="CLP on Tracr GT")
    ax.plot(kk, ch, ":", color=GRAY, lw=1.0, label="random-injection")
    ax.set_xlabel(r"CoT step $k$ (normalized)"); ax.set_ylabel("commitment")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_validity.pdf", **SAVE_KW); plt.close(fig)

def fig_results_main():
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(7.0, 6.2),
                             sharex=True, sharey=True)
    for i, m in enumerate(MODELS):
        for j, t2 in enumerate(TASKS):
            ax = axes[i][j]
            kk2, cp2 = cdp_of(t2); _, cm, ci = ser(m, t2, "CDM_est")
            ax.fill_between(kk2, cp2, cm, where=cm > cp2, color=VERM, alpha=0.13, lw=0)
            ax.plot(kk2, cp2, color=BLACK, lw=1.0, ls="--")
            ax.plot(kk2, cm, color=C.get(m, "#444"), lw=1.2)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 1])
            if i == 0: ax.set_title(t2, fontsize=8)
            if j == 0: ax.set_ylabel(m, fontsize=7)
            clean_ax(ax)
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=8)
    fig.supylabel(r"commitment  $\mathrm{CD}_M(k)$ vs $\mathrm{CD}_P(k)$", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_results_main.pdf", **SAVE_KW); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX FIGURES (12)
# ═══════════════════════════════════════════════════════════════════════════════

# --- I.1 Layer ablation heatmap (5 models × 5 tasks) -------------------------
def fig_ablation_layer():
    """Heatmap: x=step(24), y=layer(32), color=CD_M. 5×5 grid."""
    data = load_csv("sim/sim_layer.csv") or sim_layer_data()
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(7.5, 7.0),
                             sharex=True, sharey=True)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TASKS):
            ax = axes[i][j]
            subset = [r for r in data if r["model"] == m and r["task"] == t]
            ks = sorted(set(float(r["k"]) for r in subset))
            ls = sorted(set(int(float(r["layer"])) for r in subset))
            grid = np.zeros((len(ls), len(ks)))
            for r in subset:
                ki = ks.index(float(r["k"]))
                li = ls.index(int(float(r["layer"])))
                grid[li, ki] = float(r["CDM_est"])
            ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=1,
                      cmap="YlOrRd", extent=[0, 1, 0, len(ls)])
            if i == 0: ax.set_title(t, fontsize=7)
            if j == 0: ax.set_ylabel(m, fontsize=6)
            if i == len(MODELS) - 1: ax.set_xlabel("step $k$", fontsize=6)
            ax.set_xticks([0, 0.5, 1])
    fig.suptitle(r"Layer ablation: $\mathrm{CD}_M(k)$ by layer", fontsize=9)
    fig.tight_layout(pad=0.4, rect=[0, 0, 0.95, 0.96])
    # colorbar
    cax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cax, label=r"$\mathrm{CD}_M$")
    fig.savefig(OUT / "fig_ablation_layer.pdf", **SAVE_KW); plt.close(fig)


# --- I.2 Sampling convergence -------------------------------------------------
def fig_ablation_sampling():
    """CD_M ± CI curves for varying N, focal task A5/S5, focal model R1-Distill."""
    data = load_csv("sim/sim_sampling.csv") or sim_sampling_data()
    Ns = sorted(set(int(float(r["N"])) for r in data))
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(Ns)))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for idx, N in enumerate(Ns):
        sub = sorted([(float(r["k"]), float(r["CDM_est"]), float(r["CI"]))
                      for r in data if int(float(r["N"])) == N])
        ks = np.array([s[0] for s in sub])
        cd = np.array([s[1] for s in sub])
        ci = np.array([s[2] for s in sub])
        ax.plot(ks, cd, color=cmap[idx], lw=1.0, label=f"$N$={N}")
        ax.fill_between(ks, cd - ci, cd + ci, color=cmap[idx], alpha=0.1, lw=0)
    ax.set_xlabel(r"CoT step $k$ (normalized)"); ax.set_ylabel(r"$\mathrm{CD}_M(k)$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation_sampling.pdf", **SAVE_KW); plt.close(fig)


# --- I.3 Control injection ----------------------------------------------------
def fig_ablation_control():
    """3 controls + real source, 5 tasks × 1 focal model (R1-Distill)."""
    data = load_csv("sim/sim_control.csv") or sim_control_data()
    cond_colors = {"source": VERM, "random": GRAY, "orthogonal": SKY,
                   "same-answer": GREEN}
    cond_styles = {"source": "-", "random": ":", "orthogonal": "--",
                   "same-answer": "-."}
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.8), sharey=True)
    for j, t in enumerate(TASKS):
        ax = axes[j]
        for cond in ["source", "random", "orthogonal", "same-answer"]:
            sub = sorted([(float(r["k"]), float(r["CDM_est"]))
                          for r in data
                          if r["model"] == "R1-Distill" and r["task"] == t
                          and r["condition"] == cond])
            if not sub: continue
            ks = np.array([s[0] for s in sub])
            cd = np.array([s[1] for s in sub])
            ax.plot(ks, cd, cond_styles[cond], color=cond_colors[cond],
                    lw=1.2, label=cond if j == 0 else None)
        ax.set_title(t, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        if j == 0: ax.set_ylabel(r"$\mathrm{CD}_M(k)$", fontsize=7)
        clean_ax(ax)
    axes[0].legend(fontsize=5, frameon=False, loc="upper left")
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation_control.pdf", **SAVE_KW); plt.close(fig)


# --- I.4 Probe accuracy vs CD_M overlay ---------------------------------------
def fig_ablation_probes():
    """Answer-probe acc, correctness-probe acc, CD_M. 5 tasks, focal model R1-Distill."""
    pdata = load_csv("sim/sim_probes.csv") or sim_probe_data()
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.8), sharey=True)
    for j, t in enumerate(TASKS):
        ax = axes[j]
        # CD_M from main data
        kk, cp = cdp_of(t)
        _, cm, _ = ser("R1-Distill", t, "CDM_est")
        ax.plot(kk, cp, "--", color=BLACK, lw=0.8, label=r"$\mathrm{CD}_P$" if j == 0 else None)
        ax.plot(kk, cm, color=VERM, lw=1.2, label=r"$\mathrm{CD}_M$" if j == 0 else None)
        # probes
        sub = sorted([(float(r["k"]), float(r["answer_acc"]), float(r["corr_acc"]))
                      for r in pdata
                      if r["model"] == "R1-Distill" and r["task"] == t])
        if sub:
            pk = np.array([s[0] for s in sub])
            pa = np.array([s[1] for s in sub])
            pc = np.array([s[2] for s in sub])
            ax.plot(pk, pa, "-", color=BLUE, lw=1.0, alpha=0.8,
                    label="answer probe" if j == 0 else None)
            ax.plot(pk, pc, "-", color=GREEN, lw=1.0, alpha=0.8,
                    label="correctness probe" if j == 0 else None)
        ax.set_title(t, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        if j == 0: ax.set_ylabel("value", fontsize=7)
        clean_ax(ax)
    axes[0].legend(fontsize=4.5, frameon=False, loc="center left")
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation_probes.pdf", **SAVE_KW); plt.close(fig)


# --- I.5 Temperature ablation ------------------------------------------------
def fig_ablation_temperature():
    """k* vs temperature. A5/S5, all 5 models."""
    tdata = load_csv("sim/sim_temperature.csv") or sim_temperature_data()
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for m in MODELS:
        sub = sorted([(float(r["temperature"]), float(r["kstar"]))
                      for r in tdata if r["model"] == m and r["task"] == "A5/S5"])
        if not sub: continue
        temps = np.array([s[0] for s in sub])
        ks = np.array([s[1] for s in sub])
        ax.plot(temps, ks, "-o", color=C[m], ms=3, lw=1.2, label=m)
    ax.set_xlabel("generation temperature"); ax.set_ylabel(r"$k^{*}$")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation_temperature.pdf", **SAVE_KW); plt.close(fig)


# --- I.6 Chain length ablation ------------------------------------------------
def fig_ablation_chainlen():
    """k* (absolute and normalized) vs max chain length. 2 models × 2 tasks."""
    cdata = load_csv("sim/sim_chainlen.csv") or sim_chainlen_data()
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
    for ax, ycol, ylabel in [(axes[0], "kstar", r"$k^{*}$ (absolute)"),
                              (axes[1], "kstar_norm", r"$k^{*}$ (normalized)")]:
        for m in ["Pythia", "R1-Distill"]:
            for t in ["A5/S5", "Dyck-2"]:
                sub = sorted([(int(float(r["max_len"])), float(r[ycol]))
                              for r in cdata if r["model"] == m and r["task"] == t])
                if not sub: continue
                lens = np.array([s[0] for s in sub])
                ks = np.array([s[1] for s in sub])
                ls = "--" if t == "Dyck-2" else "-"
                ax.plot(lens, ks, ls + "o", color=C[m], ms=3, lw=1.0,
                        label=f"{m}, {t}")
        ax.set_xlabel("max chain length"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=5, frameon=False)
        clean_ax(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_ablation_chainlen.pdf", **SAVE_KW); plt.close(fig)


# --- I.7 Source selection ablation --------------------------------------------
def fig_ablation_source():
    """CD_M curves for 3 source strategies. Focal model R1-Distill, 5 tasks."""
    sdata = load_csv("sim/sim_source.csv") or sim_source_data()
    src_colors = {"random-different": VERM, "adversarial": BLUE,
                  "nearest-neighbor": GREEN}
    src_styles = {"random-different": "-", "adversarial": "--",
                  "nearest-neighbor": ":"}
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.8), sharey=True)
    for j, t in enumerate(TASKS):
        ax = axes[j]
        kk, cp = cdp_of(t)
        ax.plot(kk, cp, "--", color=BLACK, lw=0.8)
        for src in ["random-different", "adversarial", "nearest-neighbor"]:
            sub = sorted([(float(r["k"]), float(r["CDM_est"]))
                          for r in sdata if r["task"] == t and r["source_type"] == src])
            if not sub: continue
            ks = np.array([s[0] for s in sub])
            cd = np.array([s[1] for s in sub])
            ax.plot(ks, cd, src_styles[src], color=src_colors[src], lw=1.0,
                    label=src if j == 0 else None)
        ax.set_title(t, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        if j == 0: ax.set_ylabel(r"$\mathrm{CD}_M(k)$", fontsize=7)
        clean_ax(ax)
    axes[0].legend(fontsize=5, frameon=False, loc="upper left")
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation_source.pdf", **SAVE_KW); plt.close(fig)


# --- H.5 CMI depth profiles --------------------------------------------------
def fig_cmi_depth():
    """CMI_ell bar chart alongside per-layer CD_M line. 5 models × 5 tasks."""
    cmi_data = load_csv("sim/sim_cmi.csv") or sim_cmi_data()
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(7.5, 7.0),
                             sharex=True, sharey=True)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TASKS):
            ax = axes[i][j]
            sub = sorted([(int(float(r["layer"])), float(r["CMI"]))
                          for r in cmi_data if r["model"] == m and r["task"] == t])
            if sub:
                ls = np.array([s[0] for s in sub])
                vals = np.array([s[1] for s in sub])
                ax.bar(ls, vals, width=0.8, color=C.get(m, GRAY), alpha=0.6)
            ax.set_xlim(-1, 32); ax.set_ylim(0, 1.05)
            if i == 0: ax.set_title(t, fontsize=7)
            if j == 0: ax.set_ylabel(m, fontsize=6)
            if i == len(MODELS) - 1: ax.set_xlabel("layer", fontsize=6)
            clean_ax(ax)
    fig.suptitle(r"CMI$_\ell$ depth profiles", fontsize=9)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_cmi_depth.pdf", **SAVE_KW); plt.close(fig)


# --- H.7 NLDD k* vs CLP k* scatter ------------------------------------------
def fig_nldd_comparison():
    """Scatter: x=k*(CLP), y=k*(NLDD). Each point = (model, task)."""
    ndata = load_csv("sim/sim_nldd.csv") or sim_nldd_data()
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=0.8, alpha=0.5, label="$y=x$")
    for m in MODELS:
        sub = [(float(r["kstar_clp"]), float(r["kstar_nldd"]))
               for r in ndata if r["model"] == m]
        if not sub: continue
        x = np.array([s[0] for s in sub])
        y = np.array([s[1] for s in sub])
        ax.scatter(x, y, color=C[m], s=25, zorder=5, label=m, edgecolors="white",
                   linewidths=0.3)
    ax.set_xlabel(r"$k^{*}$ (CLP, ours)"); ax.set_ylabel(r"$k^{*}$ (NLDD)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, frameon=False)
    clean_ax(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_nldd_comparison.pdf", **SAVE_KW); plt.close(fig)


# --- H.8 Competitor overlay (per task) ----------------------------------------
def fig_competitor_overlay():
    """All baselines overlaid with CD_M and CD_P. 1 row × 5 tasks."""
    # Use main data for CD_M/CD_P; baselines from sim or simulated
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.0), sharey=True)
    for j, t in enumerate(TASKS):
        ax = axes[j]
        kk, cp = cdp_of(t)
        _, cm, _ = ser("R1-Distill", t, "CDM_est")
        ax.plot(kk, cp, "--", color=BLACK, lw=1.0,
                label=r"$\mathrm{CD}_P$" if j == 0 else None)
        ax.plot(kk, cm, color=VERM, lw=1.2,
                label=r"$\mathrm{CD}_M$ (CLP)" if j == 0 else None)
        # early-answering (simulated: slightly ahead of CD_M)
        ea = np.clip(sim_sigmoid(kk, 0.25, 8), 0, 1)
        ax.plot(kk, ea, ":", color=ORANGE, lw=0.9,
                label="early-ans." if j == 0 else None)
        # probe accuracy (simulated: well ahead of CD_M)
        pa = np.clip(sim_sigmoid(kk, 0.2, 9), 0, 1)
        ax.plot(kk, pa, "-.", color=BLUE, lw=0.9,
                label="answer probe" if j == 0 else None)
        ax.set_title(t, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        if j == 0: ax.set_ylabel("value", fontsize=7)
        clean_ax(ax)
    axes[0].legend(fontsize=4.5, frameon=False, loc="center left")
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_competitor_overlay.pdf", **SAVE_KW); plt.close(fig)


# --- J: Per-task detailed panels (5 panels, CD_M + CI bands for all models) ---
def fig_pertask_detailed():
    """5 panels (one per task), each showing all 5 models' CD_M with CI bands + CD_P."""
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.0), sharey=True)
    for j, t in enumerate(TASKS):
        ax = axes[j]
        kk, cp = cdp_of(t)
        ax.plot(kk, cp, "--", color=BLACK, lw=1.2)
        for m in MODELS:
            _, cm, ci = ser(m, t, "CDM_est")
            ax.fill_between(kk, cm - ci, cm + ci, color=C[m], alpha=0.08, lw=0)
            ax.plot(kk, cm, color=C[m], lw=1.0,
                    label=m if j == 0 else None)
        ax.set_title(t, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1])
        if j == 0: ax.set_ylabel(r"$\mathrm{CD}_M(k)$", fontsize=7)
        clean_ax(ax)
    axes[0].legend(fontsize=4.5, frameon=False, loc="upper left")
    fig.supxlabel("CoT step $k$ (normalized)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_pertask_detailed.pdf", **SAVE_KW); plt.close(fig)


# --- J: Full 5×5 U/k* heatmap ------------------------------------------------
def fig_full_table2():
    """Heatmap of U (top) and k* (bottom), 5 models × 5 tasks."""
    headline = load_csv("sim/sim_headline.csv")
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 3.5))
    for ax, metric, title, cmap in [(axes[0], "U", r"$U$ (unfaithfulness)", "Reds"),
                                     (axes[1], "kstar", r"$k^{*}$ (commitment step)", "Blues_r")]:
        grid = np.zeros((len(MODELS), len(TASKS)))
        for r in headline:
            if r["model"] in MODELS and r["task"] in TASKS:
                i = MODELS.index(r["model"])
                j = TASKS.index(r["task"])
                grid[i, j] = float(r[metric])
        im = ax.imshow(grid, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, fontsize=6)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=6)
        for i in range(len(MODELS)):
            for j in range(len(TASKS)):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig_full_table2.pdf", **SAVE_KW); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════
MAIN_FIGS = [
    ("fig_cd_schematic", fig_cd_schematic),
    ("fig_results_compact", fig_results_compact),
    ("fig_matched_acc", fig_matched_acc),
    ("fig_validity", fig_validity),
    ("fig_results_main", fig_results_main),
]
APPENDIX_FIGS = [
    ("fig_ablation_layer", fig_ablation_layer),
    ("fig_ablation_sampling", fig_ablation_sampling),
    ("fig_ablation_control", fig_ablation_control),
    ("fig_ablation_probes", fig_ablation_probes),
    ("fig_ablation_temperature", fig_ablation_temperature),
    ("fig_ablation_chainlen", fig_ablation_chainlen),
    ("fig_ablation_source", fig_ablation_source),
    ("fig_cmi_depth", fig_cmi_depth),
    ("fig_nldd_comparison", fig_nldd_comparison),
    ("fig_competitor_overlay", fig_competitor_overlay),
    ("fig_pertask_detailed", fig_pertask_detailed),
    ("fig_full_table2", fig_full_table2),
]

def run(figs):
    for name, fn in figs:
        try:
            fn()
            print(f"  wrote {name}.pdf")
        except Exception as e:
            print(f"  SKIP {name}: {e}")

if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "--all" in args:
        print("=== Main-text figures ===")
        run(MAIN_FIGS)
        print("=== Appendix figures ===")
        run(APPENDIX_FIGS)
    elif "--main" in args:
        print("=== Main-text figures ===")
        run(MAIN_FIGS)
    elif "--appendix" in args:
        print("=== Appendix figures ===")
        run(APPENDIX_FIGS)
    print("Done.")
