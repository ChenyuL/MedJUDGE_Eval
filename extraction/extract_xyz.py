#!/usr/bin/env python3
"""
Extract PICO elements (Population, Intervention/Exposure, Comparison, Outcome + Confounders)
from RWE paper FULL-TEXT using multiple LLM models.
Uses first 6,000 characters (abstract + intro + methods sections).
Output: PICO-structured JSON compatible with epidemiology reporting standards.

DATA SOURCE: papers/fulltext/*.txt (30 or 153 full-text RWE papers)

USAGE:
  1. Set EXTRACTION_MODE = '30_papers' or '153_papers' (line 36)
  2. For 30 papers: Uses premium models (Claude, GPT-4.5, Gemini) + open-source
  3. For 153 papers: Uses only open-source models (cost-effective)
  4. Ensure required API keys are in .env file:
     - HF_TOKEN (required for all)
     - OPENAI_API_KEY (for 30-paper mode)
     - ANTHROPIC_API_KEY (for 30-paper mode)
     - GEMINI_API_KEY (for 30-paper mode)
"""

import os, json, time, glob
from datetime import datetime
from huggingface_hub import InferenceClient
import requests
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found. Set it with: export HF_TOKEN='your_token'")
    exit(1)

client = InferenceClient(token=HF_TOKEN)

# Validate API keys based on extraction mode
def validate_api_keys():
    """Check that required API keys are available for selected models."""
    models_to_check = MODELS  # already resolved above

    required_keys = set()
    for _, method, _ in models_to_check:
        if method == 'openai':
            required_keys.add('OPENAI_API_KEY')
        elif method == 'anthropic':
            required_keys.add('ANTHROPIC_API_KEY')
        elif method == 'gemini':
            required_keys.add('GEMINI_API_KEY')

    missing = []
    if 'OPENAI_API_KEY' in required_keys and not OPENAI_API_KEY:
        missing.append('OPENAI_API_KEY')
    if 'ANTHROPIC_API_KEY' in required_keys and not ANTHROPIC_API_KEY:
        missing.append('ANTHROPIC_API_KEY')
    if 'GEMINI_API_KEY' in required_keys and not GEMINI_API_KEY:
        missing.append('GEMINI_API_KEY')

    if missing:
        print(f"ERROR: Missing required API keys for {EXTRACTION_MODE} mode: {', '.join(missing)}")
        print("Add them to your .env file or export them as environment variables")
        exit(1)
RESULTS_DIR = 'pico_extraction'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── Set extraction mode ──────────────────────────────────────────────────────
# 'pico_4models' : AMIA 2026 PICO experiment — 4 open-source extractors (USE THIS)
# '30_papers'    : legacy premium-model test (Claude, GPT, Gemini — requires paid keys)
# '153_papers'   : legacy open-source full run (different model set)
EXTRACTION_MODE = 'pico_4models'

# ── Configuration A: PICO experiment — 4 open-source extractors ──────────────
# Matches run_cross_judge_v2.py ext_models exactly.
# Keys stored: 'DeepSeek-V3', 'Kimi-K2', 'Llama-3.3-70B', 'Gemma-3-27B'
MODELS_PICO = [
    ('DeepSeek-V3',   'hf', 'deepseek-ai/DeepSeek-V3-0324'),
    ('Kimi-K2',       'hf', 'moonshotai/Kimi-K2-Instruct'),
    ('Llama-3.3-70B', 'hf', 'meta-llama/Llama-3.3-70B-Instruct'),
    ('Gemma-3-27B',   'hf', 'google/gemma-3-27b-it'),
]

# ── Configuration B: Premium models for 30-paper test (paid APIs) ────────────
# Run when you have API tokens for Claude / GPT / Gemini.
# Set EXTRACTION_MODE = '30_papers' to activate.
MODELS_30_PAPERS = [
    ('Claude-Sonnet-4.6', 'anthropic', 'claude-sonnet-4-6'),
    ('GPT-4o',            'openai',    'gpt-4o-2024-11-20'),
    ('Gemini-2.0-Flash',  'gemini',    'gemini-2.0-flash'),
    ('DeepSeek-V3',       'hf',        'deepseek-ai/DeepSeek-V3-0324'),
    ('Llama-3.3-70B',     'hf',        'meta-llama/Llama-3.3-70B-Instruct'),
]

# ── Configuration C: Open-source only, 153-paper full run (legacy) ────────────
MODELS_153_PAPERS = [
    ('DeepSeek-V3',   'hf',        'deepseek-ai/DeepSeek-V3-0324'),
    ('Kimi-K2',       'hf',        'moonshotai/Kimi-K2-Instruct'),
    ('Llama-3.3-70B', 'hf',        'meta-llama/Llama-3.3-70B-Instruct'),
    ('Qwen2.5-32B',   'sambanova', 'Qwen/Qwen2.5-32B-Instruct'),
]

# ── Select model list based on mode ──────────────────────────────────────────
if EXTRACTION_MODE == 'pico_4models':
    MODELS = MODELS_PICO
elif EXTRACTION_MODE == '30_papers':
    MODELS = MODELS_30_PAPERS
else:
    MODELS = MODELS_153_PAPERS

# Configuration: how much of the paper to use
PAPER_TRUNCATE_LENGTH = 6000  # First 6,000 characters (abstract + intro + methods)

# GPT-5.2 multi-run configuration (for measuring temperature instability)
# GPT-5.2 lacks temperature control, so we run it multiple times to measure variance
GPT5_NUM_RUNS = 3  # Run GPT-5.2 three times per paper to quantify instability

EXTRACTION_PROMPT = """You are an expert epidemiologist extracting PICO elements from a real-world evidence (RWE) research paper.

Given the paper text below (abstract, introduction, and methods sections), extract the PICO framework elements:

- **P (Population):** Who was studied? Include demographics, disease/condition, setting.
- **I (Intervention/Exposure):** The main exposure, treatment, or intervention being studied.
- **C (Comparison):** The comparison or control group (e.g., placebo, no treatment, alternative drug, unexposed group).
- **O (Outcome):** The primary and secondary outcomes measured.
- **Confounders:** Variables that were adjusted for or controlled in the analysis.

Respond in EXACTLY this JSON format (no other text before or after):

```json
{
  "P": {
    "description": "brief description of study population",
    "demographics": "age, sex, race/ethnicity if reported",
    "setting": "clinical/geographic setting"
  },
  "I": {
    "variables": ["list of exposure/intervention variables"],
    "description": "brief description of the main intervention or exposure"
  },
  "C": {
    "description": "comparison or control group",
    "type": "e.g., placebo, unexposed, active comparator, no treatment"
  },
  "O": {
    "primary_outcome": "the main outcome",
    "secondary_outcomes": ["list of secondary outcomes if any"],
    "measurement": "how outcomes were measured"
  },
  "Confounders": {
    "variables": ["list of confounders/covariates adjusted for"],
    "adjustment_method": "e.g., propensity score matching, multivariable regression, inverse probability weighting"
  },
  "study_design": "type of study (e.g., retrospective cohort, case-control, cross-sectional, RCT)",
  "sample_size": "number of participants if mentioned",
  "causal_question": "one sentence: does I affect O in population P, compared to C, after adjusting for confounders?"
}
```

PAPER TEXT:
{paper_text}
"""

def call_hf(model_id, prompt):
    """Call via HuggingFace InferenceClient."""
    try:
        r = client.chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            model=model_id,
            max_tokens=1000,
            temperature=0.0
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def call_sambanova(model_id, prompt):
    """Call via SambaNova through HF router."""
    try:
        r = requests.post('https://router.huggingface.co/sambanova/v1/chat/completions',
            headers={'Authorization': f'Bearer {HF_TOKEN}', 'Content-Type': 'application/json'},
            json={
                'model': model_id,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1000,
                'temperature': 0.0
            },
            timeout=120)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            # Strip <think> blocks from some models
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            return content
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_openai(model_id, prompt):
    """Call via OpenAI API (handles both GPT-4 and GPT-5 formats).

    RESEARCH NOTE - GPT-5.2 API Limitations:
    - No max_tokens parameter support
    - No temperature parameter support (cannot set temperature=0)

    To measure GPT-5.2's instability from uncontrollable temperature,
    we run it multiple times (GPT5_NUM_RUNS=3) per paper and report:
    - Mean extraction across runs
    - Variance metrics to quantify inconsistency

    All other models use temperature=0 for perfect reproducibility.
    """
    try:
        payload = {
            'model': model_id,
            'messages': [{'role': 'user', 'content': prompt}]
        }

        # GPT-5 models don't support max_tokens or temperature parameters
        if not model_id.startswith('gpt-5'):
            payload['max_tokens'] = 1000
            payload['temperature'] = 0.0  # Perfect reproducibility

        r = requests.post('https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=120)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_anthropic(model_id, prompt):
    """Call via Anthropic API."""
    try:
        r = requests.post('https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json={
                'model': model_id,
                'max_tokens': 1000,
                'temperature': 0.0,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=120)
        if r.status_code == 200:
            return r.json()['content'][0]['text']
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_gemini(model_id, prompt):
    """Call via Google Gemini API."""
    try:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}'
        r = requests.post(url,
            headers={'Content-Type': 'application/json'},
            json={
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'temperature': 0.0,
                    'maxOutputTokens': 1000
                }
            },
            timeout=120)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def extract_json(text):
    """Extract JSON from model response."""
    if not text or text.startswith('ERROR'):
        return None

    # Try to find JSON block
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]

    # Find outermost { ... }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError as e:
            # Try to fix common JSON errors
            json_str = text[start:end+1]
            # Remove trailing commas
            json_str = json_str.replace(',}', '}').replace(',]', ']')
            try:
                return json.loads(json_str)
            except:
                pass
    return None

def to_fhir_evidence_variable(pmid, pico_data, model_name):
    """Convert PICO extraction to FHIR EvidenceVariable-like structure."""
    if not pico_data:
        return None

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "meta": {
            "source": model_name,
            "pmid": pmid,
            "extracted": datetime.now().isoformat(),
            "paper_truncate_length": PAPER_TRUNCATE_LENGTH
        },
        "entry": [
            {
                "resourceType": "EvidenceVariable",
                "id": f"{pmid}-population",
                "name": "Population (P)",
                "description": pico_data.get("P", {}).get("description", ""),
                "demographics": pico_data.get("P", {}).get("demographics", ""),
                "setting": pico_data.get("P", {}).get("setting", "")
            },
            {
                "resourceType": "EvidenceVariable",
                "id": f"{pmid}-intervention",
                "name": "Intervention/Exposure (I)",
                "characteristic": [{"description": v} for v in pico_data.get("I", {}).get("variables", [])],
                "description": pico_data.get("I", {}).get("description", "")
            },
            {
                "resourceType": "EvidenceVariable",
                "id": f"{pmid}-comparison",
                "name": "Comparison (C)",
                "description": pico_data.get("C", {}).get("description", ""),
                "type": pico_data.get("C", {}).get("type", "")
            },
            {
                "resourceType": "EvidenceVariable",
                "id": f"{pmid}-outcome",
                "name": "Outcome (O)",
                "description": pico_data.get("O", {}).get("primary_outcome", ""),
                "secondary_outcomes": pico_data.get("O", {}).get("secondary_outcomes", []),
                "measurement": pico_data.get("O", {}).get("measurement", ""),
                "handling": "continuous"
            },
            {
                "resourceType": "EvidenceVariable",
                "id": f"{pmid}-confounders",
                "name": "Confounders",
                "characteristic": [{"description": v} for v in pico_data.get("Confounders", {}).get("variables", [])],
                "description": pico_data.get("Confounders", {}).get("adjustment_method", "")
            }
        ],
        "study_design": pico_data.get("study_design", ""),
        "sample_size": pico_data.get("sample_size", ""),
        "causal_question": pico_data.get("causal_question", "")
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

# Validate API keys before starting
validate_api_keys()

# Load full-text papers
papers = {}
fulltext_dir = 'papers/fulltext'

if not os.path.exists(fulltext_dir):
    print(f"ERROR: {fulltext_dir}/ directory not found!")
    print("Expected structure: papers/fulltext/*.txt (30 full-text papers)")
    exit(1)

for f in sorted(glob.glob(f'{fulltext_dir}/*.txt')):
    pmid = os.path.basename(f).replace('.txt', '')
    with open(f, encoding='utf-8') as fh:
        full_text = fh.read()
        # Use first 6,000 characters (abstract + intro + methods sections typically)
        papers[pmid] = full_text[:PAPER_TRUNCATE_LENGTH]

if not papers:
    print(f"ERROR: No papers found in {fulltext_dir}/")
    exit(1)

print(f"{'='*80}")
print(f"PICO EXTRACTION PIPELINE (FULL-TEXT)")
print(f"{'='*80}")
print(f"Extraction mode:    {EXTRACTION_MODE}")
print(f"Papers found:       {len(papers)}")
print(f"Models to test:     {len(MODELS)}")
print(f"Text length used:   First {PAPER_TRUNCATE_LENGTH} characters per paper")
print(f"Total extractions:  {len(papers)} × {len(MODELS)} = {len(papers) * len(MODELS)}")
print(f"\nModels selected:")
for model_name, method, model_id in MODELS:
    print(f"  • {model_name:20s} ({method:10s}) {model_id}")
print(f"\nStarted:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

# ── Per-model file helper ────────────────────────────────────────────────────

def save_model_files(all_results, results_dir):
    """Save one JSON file per model: pico_extraction/{ModelName}.json
    Contains {pmid: {raw, parsed, fhir, success, timestamp}} for that model.
    Automatically covers all model names ever seen across runs.
    """
    all_model_names = set()
    for pmid_data in all_results.values():
        all_model_names.update(pmid_data.keys())

    for model_name in sorted(all_model_names):
        model_data = {
            pmid: data[model_name]
            for pmid, data in all_results.items()
            if model_name in data
        }
        safe_name = model_name.replace('/', '-').replace(' ', '_')
        with open(f'{results_dir}/{safe_name}.json', 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, default=str)

    return sorted(all_model_names)

# ── Check for existing progress ───────────────────────────────────────────────
progress_file = f'{RESULTS_DIR}/progress.json'
if os.path.exists(progress_file):
    with open(progress_file, encoding='utf-8') as f:
        all_results = json.load(f)
    completed = sum(1 for p in all_results.values() if len(p) == len(MODELS))
    print(f"📁 Resuming: {completed}/{len(papers)} papers fully completed")
    print()
else:
    all_results = {}
    print(f"📁 Starting fresh extraction\n")

# Extract from each paper with each model
for i, (pmid, paper_text) in enumerate(papers.items()):
    # Check if this paper is already fully processed
    if pmid in all_results and len(all_results[pmid]) == len(MODELS):
        print(f"[{i+1:2d}/{len(papers)}] {pmid} — ✅ Already complete, skipping")
        continue

    print(f"[{i+1:2d}/{len(papers)}] {pmid} ({len(paper_text):,} chars)")

    if pmid not in all_results:
        all_results[pmid] = {}

    prompt = EXTRACTION_PROMPT.replace('{paper_text}', paper_text)

    for model_name, method, model_id in MODELS:
        # Skip if already done
        if model_name in all_results[pmid]:
            existing = all_results[pmid][model_name]
            if isinstance(existing, dict) and existing.get('success'):
                print(f"  → {model_name:20s} ... ⏭  (cached)")
                continue

        # GPT-5.2 special handling: run multiple times to measure instability
        is_gpt5 = (method == 'openai' and model_id.startswith('gpt-5'))
        num_runs = GPT5_NUM_RUNS if is_gpt5 else 1

        if is_gpt5:
            print(f"  → {model_name:20s} ... (running {num_runs}x) ", end='', flush=True)
        else:
            print(f"  → {model_name:20s} ... ", end='', flush=True)

        # Run model (multiple times for GPT-5.2)
        all_runs = []
        for run_idx in range(num_runs):
            # Call model based on method
            if method == 'hf':
                raw = call_hf(model_id, prompt)
            elif method == 'sambanova':
                raw = call_sambanova(model_id, prompt)
            elif method == 'openai':
                raw = call_openai(model_id, prompt)
            elif method == 'anthropic':
                raw = call_anthropic(model_id, prompt)
            elif method == 'gemini':
                raw = call_gemini(model_id, prompt)
            else:
                raw = f"ERROR: Unknown method '{method}'"

            # Parse JSON response
            parsed = extract_json(raw)

            all_runs.append({
                'raw': raw[:3000] if raw else None,
                'parsed': parsed,
                'success': parsed is not None
            })

            if is_gpt5 and run_idx < num_runs - 1:
                time.sleep(1)  # Wait between GPT-5 runs

        # For GPT-5.2: use first successful run as primary, store all runs for variance analysis
        # For other models: just use the single run
        if is_gpt5:
            successful_runs = [r for r in all_runs if r['success']]
            primary_run = successful_runs[0] if successful_runs else all_runs[0]

            fhir = to_fhir_evidence_variable(pmid, primary_run['parsed'], model_name)

            all_results[pmid][model_name] = {
                'raw': primary_run['raw'],
                'parsed': primary_run['parsed'],
                'fhir': fhir,
                'success': primary_run['success'],
                'timestamp': datetime.now().isoformat(),
                'gpt5_all_runs': all_runs,  # Store all runs for variance analysis
                'gpt5_num_runs': num_runs,
                'gpt5_success_rate': sum(1 for r in all_runs if r['success']) / num_runs
            }
        else:
            # Regular model (single run)
            run = all_runs[0]
            fhir = to_fhir_evidence_variable(pmid, run['parsed'], model_name)

            all_results[pmid][model_name] = {
                'raw': run['raw'],
                'parsed': run['parsed'],
                'fhir': fhir,
                'success': run['success'],
                'timestamp': datetime.now().isoformat()
            }

        status = '✅' if all_results[pmid][model_name]['success'] else '❌'
        if is_gpt5:
            success_count = sum(1 for r in all_runs if r['success'])
            print(f"{status} ({success_count}/{num_runs} successful)")
        else:
            print(f"{status}")

        time.sleep(0.5)  # Rate limiting

    # Save progress after each paper
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    save_model_files(all_results, RESULTS_DIR)

    print()

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print("EXTRACTION COMPLETE")
print(f"{'='*80}\n")

print("Success rates by model:")
for model_name, _, _ in MODELS:
    success = sum(1 for p in all_results.values()
                  if p.get(model_name, {}).get('success'))
    total = len(papers)
    pct = (success/total*100) if total > 0 else 0
    print(f"  {model_name:20s}: {success:2d}/{total} ({pct:5.1f}%)")

# Save final comprehensive results
with open(f'{RESULTS_DIR}/all_extractions.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, default=str)

# Save per-model files
saved_models = save_model_files(all_results, RESULTS_DIR)

# Create FHIR EvidenceVariable bundle
fhir_bundle = []
for pmid, models in all_results.items():
    for model_name, data in models.items():
        if data.get('fhir'):
            fhir_bundle.append(data['fhir'])

with open(f'{RESULTS_DIR}/fhir_evidence_variables.json', 'w', encoding='utf-8') as f:
    json.dump({
        "resourceType": "Bundle",
        "type": "collection",
        "total": len(fhir_bundle),
        "entry": fhir_bundle,
        "meta": {
            "created": datetime.now().isoformat(),
            "source": "LLM-based extraction from RWE papers"
        }
    }, f, indent=2)

print(f"\n📁 Results saved to:")
print(f"   • {RESULTS_DIR}/progress.json              (per-model raw + parsed)")
print(f"   • {RESULTS_DIR}/all_extractions.json       (comprehensive)")
print(f"   • {RESULTS_DIR}/fhir_evidence_variables.json (FHIR bundle)")
for m in saved_models:
    safe = m.replace('/', '-').replace(' ', '_')
    print(f"   • {RESULTS_DIR}/{safe}.json")
print(f"\n✅ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")
