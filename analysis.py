import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SPREADSHEET_ID = "1JQ1mNj4BhGHCjHSxbjtCsZDHaGj6NwzKWfn4TxD_S5Q"
CREDENTIALS_FILE = "/Users/pranjaldeo/google_credentials.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
spreadsheet = gc.open_by_key(SPREADSHEET_ID)

MODELS = ["Claude", "GPT", "Gemini", "GLM", "Qwen"]
MODEL_LABELS = {
    "Claude": "claude-opus-4-8",
    "GPT": "gpt-4o",
    "Gemini": "gemini-2.5-flash",
    "GLM": "GLM 5.2",
    "Qwen": "Qwen 3.7",
}
QUESTIONS = [
    "Q1 - Work closely",
    "Q2 - Friend",
    "Q3 - Socialize",
    "Q4 - Neighbor",
    "Q5 - Marry into family",
    "Q6 - Violence likelihood",
]
CRISIS_TYPES = ["Suicidal", "Anxiety/Panic Attack", "Physical Abuse", "Homelessness/Poverty", "Loneliness"]

# Score columns per question (0-indexed): Claude, GPT, Gemini, GLM, Qwen
# Each question block is 10 cols wide; score cols are at offsets 2,4,6,8,10
SCORE_COLS = {
    "Q1 - Work closely":     [2,  4,  6,  8,  10],
    "Q2 - Friend":           [12, 14, 16, 18, 20],
    "Q3 - Socialize":        [22, 24, 26, 28, 30],
    "Q4 - Neighbor":         [32, 34, 36, 38, 40],
    "Q5 - Marry into family":[42, 44, 46, 48, 50],
    "Q6 - Violence likelihood":[52,54, 56, 58, 60],
}

# ── Load Tab 2 ────────────────────────────────────────────────────────────────
print("Loading data from Google Sheets...")
stigma_sheet = spreadsheet.worksheet("Stigma related questions")
stigma_data = stigma_sheet.get_all_values()

prep_sheet = spreadsheet.worksheet("Vignette Prep & Appropriateness")
prep_data = prep_sheet.get_all_values()

# Build stigma dataframe
rows = []
for row in stigma_data[2:]:
    if not row[1].strip():
        continue
    entry = {"Vignette": row[1].strip()}
    for q, cols in SCORE_COLS.items():
        for model, col in zip(MODELS, cols):
            val = row[col].strip() if len(row) > col else ""
            try:
                entry[f"{model}_{q}"] = float(val)
            except ValueError:
                entry[f"{model}_{q}"] = np.nan
    rows.append(entry)

df = pd.DataFrame(rows)

# Add crisis type from Tab 1 (cols 2-6 = Suicidal, Anxiety, Abuse, Homeless, Lonely)
crisis_rows = []
for row in prep_data[2:]:
    if not row[1].strip():
        continue
    crisis = {}
    for i, ctype in enumerate(CRISIS_TYPES):
        val = row[2 + i].strip() if len(row) > 2 + i else ""
        crisis[ctype] = 1 if val.lower() in ["1", "yes", "true", "x"] else 0
    crisis_rows.append(crisis)

crisis_df = pd.DataFrame(crisis_rows)
df = pd.concat([df, crisis_df], axis=1)

print(f"Loaded {len(df)} vignettes\n")

# ── 1. Descriptive Statistics ─────────────────────────────────────────────────
print("=" * 60)
print("1. DESCRIPTIVE STATISTICS — Mean score per model per question")
print("=" * 60)

desc_rows = []
for q in QUESTIONS:
    for model in MODELS:
        col = f"{model}_{q}"
        scores = df[col].dropna()
        desc_rows.append({
            "Question": q,
            "Model": model,
            "Mean": round(scores.mean(), 2),
            "Median": round(scores.median(), 2),
            "Std": round(scores.std(), 2),
            "N": len(scores),
        })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))

# ── 2. Model Comparison (Kruskal-Wallis across models) ───────────────────────
print("\n" + "=" * 60)
print("2. MODEL COMPARISON — Kruskal-Wallis test (do models score differently?)")
print("=" * 60)

for q in QUESTIONS:
    groups = [df[f"{model}_{q}"].dropna().values for model in MODELS]
    stat, p = stats.kruskal(*groups)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{q}: H={stat:.2f}, p={p:.4f} {sig}")

# ── 3. Vignette Type Comparison ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. VIGNETTE TYPE COMPARISON — Mean score by crisis type")
print("=" * 60)

for ctype in CRISIS_TYPES:
    group = df[df[ctype] == 1]
    if len(group) == 0:
        continue
    print(f"\n{ctype} (n={len(group)}):")
    for model in MODELS:
        scores = []
        for q in QUESTIONS:
            scores.extend(group[f"{model}_{q}"].dropna().tolist())
        if scores:
            print(f"  {model}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

# ── 4. Inter-Model Correlation ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. INTER-MODEL CORRELATION — Spearman correlation between models")
print("=" * 60)

all_scores = {}
for model in MODELS:
    scores = []
    for q in QUESTIONS:
        scores.extend(df[f"{model}_{q}"].tolist())
    all_scores[model] = scores

corr_df = pd.DataFrame(all_scores)
corr_matrix = corr_df.corr(method="spearman")
print(corr_matrix.round(2).to_string())

# ── 5. Per-Question Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. PER-QUESTION SUMMARY — Average score across all models")
print("=" * 60)

for q in QUESTIONS:
    all_q_scores = []
    for model in MODELS:
        all_q_scores.extend(df[f"{model}_{q}"].dropna().tolist())
    print(f"{q}: mean={np.mean(all_q_scores):.2f}, std={np.std(all_q_scores):.2f}")

# ── 6. Overall Stigma Score by Model (bar chart with 95% CI) ─────────────────
print("\n" + "=" * 60)
print("6. OVERALL STIGMA SCORE BY MODEL — Mean with 95% CI (all questions pooled)")
print("=" * 60)

model_means, model_cis, model_ns = [], [], []
for model in MODELS:
    scores = []
    for q in QUESTIONS:
        scores.extend(df[f"{model}_{q}"].dropna().tolist())
    scores = np.array(scores)
    n = len(scores)
    mean = scores.mean()
    sem = stats.sem(scores)
    ci95 = sem * stats.t.ppf(0.975, n - 1)
    model_means.append(mean)
    model_cis.append(ci95)
    model_ns.append(n)
    print(f"{model}: mean={mean:.2f}, 95% CI=[{mean - ci95:.2f}, {mean + ci95:.2f}], n={n}")

# Categorical palette (fixed slot order), validated for CVD-safety
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(MODELS))
bars = ax.bar(
    x, model_means, yerr=model_cis, capsize=4,
    color=CATEGORICAL, width=0.6,
    error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capthick": 1.2},
)

for xi, mean, ci in zip(x, model_means, model_cis):
    ax.text(xi, mean + ci + 0.08, f"{mean:.2f}", ha="center", va="bottom",
             fontsize=9, color=INK_PRIMARY)

ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Mean stigma score (1–5)", fontsize=10, color=INK_SECONDARY)
ax.set_title("Overall Stigma Score by Model (95% CI, all questions pooled)",
             fontsize=12, color=INK_PRIMARY, pad=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRIDLINE)
ax.spines["bottom"].set_color(INK_MUTED)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
ax.tick_params(axis="x", length=0)

fig.tight_layout()
OUT_PATH = "/Users/pranjaldeo/stigma_by_model.png"
fig.savefig(OUT_PATH, facecolor=SURFACE)
print(f"\nSaved bar chart to {OUT_PATH}")

# ── 7. Grouped Bar: Mean Score by Question x Model (95% CI) ─────────────────
print("\n" + "=" * 60)
print("7. SCORE BY QUESTION x MODEL — Mean with 95% CI, grouped by question")
print("=" * 60)

Q_LABELS = [q.split(" - ", 1)[1] for q in QUESTIONS]

group_means = {model: [] for model in MODELS}
group_cis = {model: [] for model in MODELS}
for q in QUESTIONS:
    for model in MODELS:
        scores = df[f"{model}_{q}"].dropna().to_numpy()
        n = len(scores)
        mean = scores.mean()
        ci95 = stats.sem(scores) * stats.t.ppf(0.975, n - 1)
        group_means[model].append(mean)
        group_cis[model].append(ci95)
    print(q + ": " + ", ".join(
        f"{model}={group_means[model][-1]:.2f}±{group_cis[model][-1]:.2f}" for model in MODELS
    ))

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

n_models = len(MODELS)
bar_width = 0.8 / n_models
group_x = np.arange(len(QUESTIONS))

for i, model in enumerate(MODELS):
    offsets = group_x - 0.4 + bar_width * (i + 0.5)
    ax.bar(
        offsets, group_means[model], yerr=group_cis[model], capsize=2.5,
        color=CATEGORICAL[i], width=bar_width, label=MODEL_LABELS[model],
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capthick": 1},
    )
    for xi, mean, ci in zip(offsets, group_means[model], group_cis[model]):
        ax.text(xi, mean + ci + 0.08, f"{mean:.2f}", ha="center", va="bottom",
                 fontsize=6.8, color=INK_PRIMARY, rotation=90)

ax.set_ylim(0, 6.3)
ax.set_xticks(group_x)
ax.set_xticklabels(Q_LABELS, fontsize=9.5, color=INK_PRIMARY)
ax.set_ylabel("Mean stigma score (1–5)", fontsize=10, color=INK_SECONDARY)
ax.set_title("Stigma Score by Question and Model (95% CI)",
             fontsize=12, color=INK_PRIMARY, pad=14)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=n_models, fontsize=9, labelcolor=INK_SECONDARY)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRIDLINE)
ax.spines["bottom"].set_color(INK_MUTED)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
ax.tick_params(axis="x", length=0)

fig.tight_layout()
OUT_PATH_Q_MODEL = "/Users/pranjaldeo/stigma_by_question_model.png"
fig.savefig(OUT_PATH_Q_MODEL, facecolor=SURFACE)
print(f"\nSaved grouped bar chart to {OUT_PATH_Q_MODEL}")

# ── 8. Bar: Overall Mean Score by Question, pooled across models (95% CI) ───
print("\n" + "=" * 60)
print("8. OVERALL SCORE BY QUESTION — Mean with 95% CI (all models pooled)")
print("=" * 60)

q_means, q_cis = [], []
for q in QUESTIONS:
    scores = np.array([v for model in MODELS for v in df[f"{model}_{q}"].dropna().tolist()])
    n = len(scores)
    mean = scores.mean()
    ci95 = stats.sem(scores) * stats.t.ppf(0.975, n - 1)
    q_means.append(mean)
    q_cis.append(ci95)
    print(f"{q}: mean={mean:.2f}, 95% CI=[{mean - ci95:.2f}, {mean + ci95:.2f}], n={n}")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(QUESTIONS))
ax.bar(
    x, q_means, yerr=q_cis, capsize=4,
    color=CATEGORICAL[0], width=0.6,
    error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2, "capthick": 1.2},
)

for xi, mean, ci in zip(x, q_means, q_cis):
    ax.text(xi, mean + ci + 0.08, f"{mean:.2f}", ha="center", va="bottom",
             fontsize=9, color=INK_PRIMARY)

ax.set_xticks(x)
ax.set_xticklabels(Q_LABELS, fontsize=9.5, color=INK_PRIMARY)
ax.set_ylabel("Mean stigma score (1–5)", fontsize=10, color=INK_SECONDARY)
ax.set_title("Overall Stigma Score by Question (95% CI, all models pooled)",
             fontsize=12, color=INK_PRIMARY, pad=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRIDLINE)
ax.spines["bottom"].set_color(INK_MUTED)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
ax.tick_params(axis="x", length=0)

fig.tight_layout()
OUT_PATH_Q = "/Users/pranjaldeo/stigma_by_question.png"
fig.savefig(OUT_PATH_Q, facecolor=SURFACE)
print(f"\nSaved bar chart to {OUT_PATH_Q}")

print("\nDone.")
