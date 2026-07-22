# AIMH — AI Mental Health Stigma & Tactic Analysis

Compares how 5 LLMs (Claude, GPT-4o, Gemini, GLM, Qwen) respond to mental-health
crisis vignettes, along two axes:

1. **Stigma** — do models rate the person in a vignette differently across
   relational-closeness questions (work with them? befriend them? marry into
   the family?), and does that vary by crisis type (suicidal ideation, anxiety,
   physical abuse, homelessness, loneliness)?
2. **Empathic tactics** — replicating the methodology from Gueorguieva et al.
   ("AI generates well-liked but templatic empathic responses," arXiv:2604.08479),
   each model's crisis response is tagged for the presence of 12 communicative
   tactics (10 from the paper + 2 crisis-specific additions: Resource Referral,
   Safety Check), then checked for how templated/repetitive each model's
   sequence of tactics is.

## Pipeline

Data lives in a Google Sheet with three tabs: `Vignette Prep & Appropriateness`,
`Stigma related questions`, and `Tactics`. The pipeline runs in three stages:

1. **`src/generation/claude_vignete_input_output_script.py`** — for each new
   vignette, generates a crisis response from all 5 models (Step 1), then asks
   each model to rate the person on 6 stigma-related questions with a
   1–5 score + one-sentence reason (Step 2). Run with `--step1`, `--step2`, or
   no flag for both.
2. **`src/analysis/tactic_analysis.py`** — tags each of the 5 models' crisis
   responses for the 12 tactics using `claude-sonnet-5` as an LLM judge
   (`--tag`), writes the tagged sequences into the `Tactics` sheet
   (`--write-sheet`), and generates prevalence/diversity/template-search charts
   split by crisis type (`--analyze`).
3. **`src/analysis/analysis.py`** — computes descriptive stats, Kruskal-Wallis
   tests, and stigma-score bar charts (by model, by question, by crisis type),
   also split across all crisis types with data.

Charts land in `results/Last Run/`, with a `COMBINED/` subfolder for charts
that show all crisis types side-by-side in one figure.

## Setup

```
pip install -r requirements.txt
```

Required environment variables (API keys for the 4 providers used):

```
ANTHROPIC_API_KEY
OPENAI_API_KEY
GEMINI_API_KEY
FIREWORKS_API_KEY
```

Place a Google service-account key at `google_credentials.json` in the repo
root (gitignored — never commit this file). It needs edit access to the
Google Sheet referenced by `SPREADSHEET_ID` in each script.

## Running

```
python3 src/generation/claude_vignete_input_output_script.py   # generate responses + stigma scores
python3 src/analysis/tactic_analysis.py --tag                  # tag new responses (costs API calls)
python3 src/analysis/tactic_analysis.py --write-sheet          # sync tags to the Tactics sheet
python3 src/analysis/tactic_analysis.py --analyze              # regenerate all tactic charts
python3 src/analysis/analysis.py                                # regenerate all stigma charts
```
