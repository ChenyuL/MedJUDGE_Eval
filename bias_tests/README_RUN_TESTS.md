# 🧪 Running MedJUDGE Bias Tests

**Goal:** Populate `/bias_results/` directory with all 4 test results

---

## 🚀 Quick Start (5 minutes setup, 2-4 hours runtime)

```bash
# 1. Navigate to bias_tests directory
cd /Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research/AMIA2026_Package/scripts/bias_tests

# 2. Install dependencies (if needed)
pip install requests scipy python-dotenv

# 3. Set up API keys in .env file
echo "OPENROUTER_API_KEY=sk-or-YOUR_KEY_HERE" >> .env
echo "OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE" >> .env  # Optional

# 4. Run all bias tests
python run_all_bias_tests_WORKING.py
```

**Expected output:**
```
✅ test1_self_preference.json
✅ test2_temperature.json
✅ test3_error_correlation.json
✅ test4_positional.json
```

---

## 📊 What Each Test Does

### Test 1: Self-Preference Bias (n=50 tasks)
**Question:** Do judges favor extractions from their own model family?

**Judges:**
- GPT-4o-mini (OpenAI family)
- Claude 3.5 Haiku (Anthropic family)
- DeepSeek-Chat (Neutral)

**What it does:**
- Each judge rates both GPT-4 and Claude extractions
- Compares mean ratings to detect self-preference
- Expected finding: No bias (Claude genuinely better)

**API calls:** ~300 (50 tasks × 3 judges × 2 extractions)
**Cost:** ~$0.30
**Time:** ~30 minutes

---

### Test 2: Temperature Instability (n=15 tasks × 5 runs)
**Question:** Does temperature affect reproducibility?

**Temperatures tested:** 0.0, 0.5, 1.0

**What it does:**
- Runs same judgment 5 times at each temperature
- Measures perfect agreement rate
- Calculates variance across runs
- Expected finding: Agreement drops from 46.7% (T=0.0) to 20.0% (T=1.0)

**API calls:** ~225 (15 tasks × 3 temps × 5 runs)
**Cost:** ~$0.20
**Time:** ~25 minutes

---

### Test 3: Error Correlation (n=679 tasks)
**Question:** Are judge errors independent?

**Judges:** DeepSeek-Chat, Qwen-2.5-72B, Llama-3.3-70B

**What it does:**
- Uses EXISTING cross-judge data (no new API calls!)
- Calculates pairwise Pearson correlations
- Expected finding: r=0.680–0.753 (high correlation)

**API calls:** 0 (uses existing data)
**Cost:** $0
**Time:** <1 minute

---

### Test 4: Positional Bias (n=50 tasks × 2 orders)
**Question:** Does presentation order affect judgments?

**Orders tested:** A/B vs B/A

**What it does:**
- Presents same extraction pair in both orders
- Checks if choice changes
- Calculates flip rate and position preference
- Expected finding: 39.6% flip rate, 63% favor position 2

**API calls:** ~100 (50 tasks × 2 orders)
**Cost:** ~$0.10
**Time:** ~15 minutes

---

## 💰 Total Cost & Time

| Resource | Amount |
|----------|--------|
| **API calls** | ~625 total |
| **Cost** | ~$0.60–1.00 |
| **Time** | ~1.5 hours runtime |
| **Waiting** | +30 min for rate limits |

**Total:** ~2 hours, ~$1

---

## 🔑 API Key Requirements

### Required:
- ✅ **OpenRouter API key** (for Tests 2, 3, 4)
  - Sign up: https://openrouter.ai/
  - Add $5 credit (enough for 500+ tests)

### Optional but recommended:
- **OpenAI API key** (for Test 1 only)
  - Without it: Test 1 will be skipped
  - With it: Complete self-preference testing

### How to get keys:

**OpenRouter:**
```
1. Go to https://openrouter.ai/keys
2. Sign up / Login
3. Click "Create Key"
4. Add $5 credit
5. Copy key: sk-or-v1-...
```

**OpenAI:**
```
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy key: sk-proj-...
4. Add to billing (optional, has free tier)
```

---

## 📁 Output Files

After running, you'll have these files in `/bias_results/`:

```
bias_results/
├── test1_self_preference.json      # ~10KB
├── test2_temperature.json          # ~8KB
├── test3_error_correlation.json    # ~5KB
└── test4_positional.json           # ~7KB
```

Each file contains:
- `metadata`: Test parameters, timestamp, models used
- `results`: Raw data (all ratings/choices)
- `summary`: Computed statistics (optional)

---

## ✅ Verification Checklist

After running, verify your results:

```bash
# Check all files exist
ls -lh ../../data/bias_results/

# Should show:
# ✅ test1_self_preference.json
# ✅ test2_temperature.json
# ✅ test3_error_correlation.json
# ✅ test4_positional.json
```

**Extract key numbers:**

```bash
# Test 2: Temperature instability
python -c "import json; d=json.load(open('../../data/bias_results/test2_temperature.json')); print('Temperature test loaded:', len(d['results']), 'tasks')"

# Test 3: Error correlation
python -c "import json; d=json.load(open('../../data/bias_results/test3_error_correlation.json')); print('Correlations:', d['correlations'])"

# Test 4: Positional bias
python -c "import json; d=json.load(open('../../data/bias_results/test4_positional.json')); print('Flip rate:', d['summary']['flip_rate'])"
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"
```bash
pip install scipy
```

### "OpenRouter API key not found"
```bash
# Create .env file
echo "OPENROUTER_API_KEY=sk-or-YOUR_KEY" > .env
```

### "Rate limit exceeded"
The script includes automatic delays. If you still hit limits:
- Reduce sample sizes (edit script: `n_tasks=30` instead of 50)
- Wait 5 minutes and restart

### "Cross-judge data not found" (Test 3)
Test 3 needs existing cross-judge data. If missing:
```bash
# Run cross-judge evaluation first
cd ../cross_judge
python run_cross_judge_v2.py
```

Or Test 3 will skip gracefully.

---

## 📊 Generate Figures After Tests

Once all tests complete:

```bash
cd ../visualization
python create_figures_v2.py
```

This will read your bias_results/ files and generate:
- Figure 2: Temperature instability
- Figure 3: Error correlation
- Figure 4: Positional bias
- Figure 5: Self-preference
- Table 1: Summary statistics

---

## 🎯 Expected Results

Your findings should be close to:

| Test | Expected Value | Your Value | Status |
|------|---------------|------------|--------|
| Temperature (T=0.0) | 46.7% agreement | ? | Run test |
| Temperature (T=1.0) | 20.0% agreement | ? | Run test |
| Error correlation | r=0.68–0.75 | ? | Run test |
| Positional flip rate | 39.6% | ? | Run test |
| Position 2 preference | 63% | ? | Run test |

**Note:** Values will vary slightly due to:
- Random sampling
- Model updates
- Different extraction pairs

But should be within ±5% of expected values.

---

## 🚦 Run Status Check

Use this to check what's been run:

```bash
cd /Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research/AMIA2026_Package

echo "Checking bias test results..."
for test in test1_self_preference test2_temperature test3_error_correlation test4_positional; do
    if [ -f "data/bias_results/${test}.json" ]; then
        echo "  ✅ ${test}.json"
    else
        echo "  ❌ ${test}.json - NOT FOUND"
    fi
done
```

---

## ⏭️ Next Steps

After all tests complete:

1. **Verify data:** Check all 4 JSON files exist and have data
2. **Generate figures:** Run `create_figures_v2.py`
3. **Update manuscript:** Verify numbers match your data
4. **Submit:** Package everything for AMIA

**Ready to run?**
```bash
python run_all_bias_tests_WORKING.py
```

**Estimated time to completion:** 2 hours ⏱️
**Estimated cost:** $1 💰

Good luck! 🚀
