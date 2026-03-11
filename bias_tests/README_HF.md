# HuggingFace-Only Bias Tests

**Created:** 2026-03-02
**Purpose:** Run all 4 bias tests using ONLY HuggingFace API (no OpenRouter/OpenAI/Anthropic needed)

---

## 🎯 Quick Start

### Run bias tests with HuggingFace only:

```bash
cd /Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research

# Make sure HF_TOKEN is set
export HF_TOKEN="hf_your_token_here"
# Or it will load from .env automatically

# Run HuggingFace-only version
python3 AMIA2026_Package/scripts/bias_tests/bias_experiments_hf.py
```

---

## 🔄 What's Different?

| Feature | Original (`bias_experiments.py`) | HuggingFace-Only (`bias_experiments_hf.py`) |
|---------|----------------------------------|---------------------------------------------|
| **APIs Required** | OpenRouter + OpenAI + Anthropic | HuggingFace only |
| **Test 1 Judges** | GPT-4o-mini, Claude Haiku, DeepSeek | DeepSeek-V3, Llama-3.3, Qwen2.5 |
| **Test 2-4** | DeepSeek via OpenRouter | DeepSeek via HuggingFace |
| **Data Source** | Old CSV file | Current cross_judge_results |
| **Setup** | Need 3 API keys | Need 1 API key (HF_TOKEN) |

---

## 🔑 Prerequisites

Only need HuggingFace token (already in your `.env`):

```bash
# Check if token is loaded
echo $HF_TOKEN
# Should show: hf_...

# If not set, export it:
export HF_TOKEN="hf_your_token_here"
```

---

## 📊 What It Tests

### Test 1: Self-Preference Bias
- **Judges:** DeepSeek-V3, Llama-3.3-70B, Qwen2.5-32B
- **Task:** Each judge rates two different extractors
- **Checks:** Do judges prefer their own model family?
- **Sample:** 30 tasks

### Test 2: Temperature Stability
- **Judge:** DeepSeek-V3
- **Task:** Same evaluation repeated 5 times at temps 0.0, 0.5, 1.0
- **Checks:** How much does temperature affect reproducibility?
- **Sample:** 15 tasks
- **Expected:** 100% agreement at temp 0.0, ~90% at temp 1.0

### Test 3: Error Correlation
- **Data:** Uses existing cross_judge_results
- **Judges:** DeepSeek-V3, Llama-3.3, Qwen2.5
- **Checks:** Do judges make independent errors?
- **Metric:** Pearson correlation between judge pairs

### Test 4: Positional Bias
- **Judge:** DeepSeek-V3
- **Task:** Compare two extractions in both orders (A/B, B/A)
- **Checks:** Does presentation order affect judgment?
- **Sample:** 30 tasks
- **Expected:** ~40% inconsistency rate

---

## 🚀 Running via Master Script

The master script automatically chooses HuggingFace-only version if OPENROUTER_API_KEY is missing:

```bash
cd /Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research
./AMIA2026_Package/scripts/run_all_experiments.sh
```

When prompted for Experiment 4:
- If OPENROUTER_API_KEY is set: Uses original version
- If only HF_TOKEN is set: Automatically uses HuggingFace-only version

---

## 📈 Expected Output

```
================================================================================
MedJUDGE BIAS EXPERIMENTS (HuggingFace Only)
Started: 2026-03-02 12:34:56
================================================================================

✅ HuggingFace API configured
   Judges: DeepSeek-V3, Llama-3.3-70B, Qwen2.5-32B

✅ Loaded 60 tasks from cross_judge results

🔬 Running Test 3 (Error Correlation - from existing data)...
================================================================================
TEST 3: ERROR CORRELATION (from cross-judge data)
================================================================================
  Found 60 tasks with judge scores

  Pairwise agreement (within ±1):
    DeepSeek-V3- vs Llama-3.3-70: exact=45.0%, ±1=78.3%, r=0.612 (n=60)
    DeepSeek-V3- vs Qwen2.5-7B  : exact=38.3%, ±1=73.3%, r=0.547 (n=60)
    Llama-3.3-70 vs Qwen2.5-7B  : exact=41.7%, ±1=75.0%, r=0.589 (n=60)

🔬 Running Test 1 (Self-Preference Bias)...
================================================================================
TEST 1: SELF-PREFERENCE BIAS (HuggingFace Only)
================================================================================
  Task 30/30

--- Self-Preference Results ---
  DeepSeek-V3: Ext_A=4.12, Ext_B=4.05, Diff=+0.07
  Llama-3.3-70B: Ext_A=4.08, Ext_B=4.13, Diff=-0.05
  Qwen2.5-32B: Ext_A=3.95, Ext_B=3.98, Diff=-0.03

🔬 Running Test 2 (Temperature Stability)...
================================================================================
TEST 2: TEMPERATURE STABILITY
================================================================================
  Task 15/15

--- Temperature Stability Results ---
  Temp 0.0: Mean variance=0.000, Perfect agreement=100.0%
  Temp 0.5: Mean variance=0.267, Perfect agreement=66.7%
  Temp 1.0: Mean variance=0.389, Perfect agreement=46.7%

🔬 Running Test 4 (Positional Bias)...
================================================================================
TEST 4: POSITIONAL BIAS
================================================================================
  Task 30/30

--- Positional Bias Results ---
  Consistent: 18/30 (60.0%)
  Inconsistent: 12/30 (40.0%)
  Position 1 preference: 52.0% (50% = unbiased)

================================================================================
ALL TESTS COMPLETE: 2026-03-02 14:23:45
Results saved to: ../data/bias_results/
================================================================================
```

---

## 📁 Output Files

Same as original version:

```
AMIA2026_Package/data/bias_results/
├── test1_self_preference.json
├── test2_temperature.json
├── test3_error_correlation.json
└── test4_positional.json
```

---

## 🔍 Advantages of HuggingFace-Only Version

1. ✅ **Simpler setup:** Only need HF_TOKEN
2. ✅ **No external dependencies:** No OpenRouter/OpenAI/Anthropic accounts needed
3. ✅ **Consistent data:** Uses current experiment data (not old CSV)
4. ✅ **Same science:** Tests the same hypotheses with equivalent models
5. ✅ **Faster:** HuggingFace Inference API is well-optimized

---

## ⚡ Runtime Comparison

| Test | Original | HuggingFace-Only | Notes |
|------|----------|------------------|-------|
| Test 1 | ~30 min | ~25 min | 30 tasks × 3 judges × 2 evals |
| Test 2 | ~45 min | ~15 min | 15 tasks × 3 temps × 5 runs |
| Test 3 | <1 min | <1 min | No API calls (uses cached data) |
| Test 4 | ~30 min | ~15 min | 30 tasks × 2 orders |
| **Total** | **~2 hours** | **~1 hour** | HF API is faster |

---

## 🆘 Troubleshooting

### "HF_TOKEN not found"
```bash
# Check .env file
cat .env | grep HF_TOKEN

# If missing, add it:
echo 'HF_TOKEN=hf_your_token_here' >> .env

# Then export:
export HF_TOKEN="hf_your_token_here"
```

### "cross_judge_results not found"
```bash
# Must run Experiment 2 first
python3 AMIA2026_Package/scripts/cross_judge/run_cross_judge_v2.py
```

### Rate limit errors
```bash
# If you hit rate limits, increase sleep time in script
# Edit line: time.sleep(0.3) → time.sleep(1.0)
```

---

## 📖 Documentation

- **Original version:** [bias_experiments.py](bias_experiments.py)
- **HuggingFace version:** [bias_experiments_hf.py](bias_experiments_hf.py)
- **Full guide:** [../README.md](../README.md)

---

**Ready to run bias tests with HuggingFace only!** 🚀
