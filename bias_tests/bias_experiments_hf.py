#!/usr/bin/env python3
"""
MedJUDGE Bias Experiments - HuggingFace Only Version
Tests: Self-preference, Temperature stability, Error correlation, Positional bias
Uses only HuggingFace Inference API (no OpenRouter/OpenAI/Anthropic needed)
"""

import os, json, time, re, random, statistics
from datetime import datetime
from collections import defaultdict
from huggingface_hub import InferenceClient
import requests

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

HF_TOKEN = os.getenv('HF_TOKEN', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in environment")
    print("Set it with: export HF_TOKEN='your_token'")
    exit(1)

client = InferenceClient(token=HF_TOKEN)
RESULTS_DIR = 'AMIA2026_Package/data/bias_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Judge models - dynamic based on available APIs
JUDGES = {
    'DeepSeek-V3':    ('hf', 'deepseek-ai/DeepSeek-V3-0324'),
    'Llama-3.3-70B':  ('hf', 'meta-llama/Llama-3.3-70B-Instruct'),
    'Qwen3-32B':      ('sambanova', 'Qwen3-32B'),
    'Qwen2.5-32B':    ('sambanova', 'Qwen2.5-32B-Instruct'),
    'Qwen3-0.6B':     ('ollama', 'qwen3:0.6b'),
}

# Add OpenAI if available
if OPENAI_API_KEY:
    JUDGES['GPT-4.5'] = ('openai', 'gpt-4.5-preview')

# Add Anthropic if available
if ANTHROPIC_API_KEY:
    JUDGES['Claude-Sonnet'] = ('anthropic', 'claude-sonnet-4-6')

# ═══════════════════════════════════════════════════════════════════════════
# API CALLERS
# ═══════════════════════════════════════════════════════════════════════════

def call_hf(model_id, messages, temperature=0.0):
    """Call via HuggingFace InferenceClient."""
    try:
        # Convert messages to single prompt if needed
        if isinstance(messages, list):
            prompt = '\n'.join([m['content'] for m in messages if m['role'] != 'system'])
            system = next((m['content'] for m in messages if m['role'] == 'system'), '')
            if system:
                prompt = f"{system}\n\n{prompt}"
        else:
            prompt = messages

        r = client.chat_completion(
            messages=[{'role': 'user', 'content': prompt}],
            model=model_id,
            max_tokens=300,
            temperature=temperature
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def call_sambanova(model_id, messages, temperature=0.0):
    """Call via SambaNova through HF router."""
    try:
        if isinstance(messages, list):
            msgs = messages
        else:
            msgs = [{'role': 'user', 'content': messages}]

        r = requests.post('https://router.huggingface.co/sambanova/v1/chat/completions',
            headers={'Authorization': f'Bearer {HF_TOKEN}', 'Content-Type': 'application/json'},
            json={'model': model_id, 'messages': msgs, 'max_tokens': 300, 'temperature': temperature},
            timeout=120)

        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            return content
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_openai(model_id, messages, temperature=0.0):
    """Call OpenAI API."""
    try:
        if isinstance(messages, list):
            msgs = messages
        else:
            msgs = [{'role': 'user', 'content': messages}]

        r = requests.post('https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
            json={'model': model_id, 'messages': msgs, 'max_tokens': 300, 'temperature': temperature},
            timeout=60)

        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_anthropic(model_id, messages, temperature=0.0):
    """Call Anthropic API."""
    try:
        if isinstance(messages, list):
            system = next((m['content'] for m in messages if m['role'] == 'system'), '')
            user_msgs = [{'role': m['role'], 'content': m['content']}
                        for m in messages if m['role'] != 'system']
        else:
            system = ''
            user_msgs = [{'role': 'user', 'content': messages}]

        payload = {
            'model': model_id,
            'messages': user_msgs,
            'max_tokens': 600,   # Claude writes 400-500 token analyses before "Score: X"
            'temperature': temperature
        }
        if system:
            payload['system'] = system

        r = requests.post('https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=60)

        if r.status_code == 200:
            return r.json()['content'][0]['text']
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_ollama(model_id, messages, temperature=0.0):
    """Call via local Ollama (OpenAI-compatible endpoint)."""
    try:
        if isinstance(messages, list):
            msgs = messages
        else:
            msgs = [{'role': 'user', 'content': messages}]

        r = requests.post('http://localhost:11434/v1/chat/completions',
            headers={'Content-Type': 'application/json'},
            json={'model': model_id, 'messages': msgs,
                  'max_tokens': 300, 'temperature': temperature},
            timeout=120)

        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content']
            if '<think>' in content:
                content = content.split('</think>')[-1].strip()
            return content
        return f"ERROR: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {e}"

def call_judge(judge_name, messages, temperature=0.0):
    """Call appropriate judge model."""
    method, model_id = JUDGES[judge_name]
    # Qwen3 models: disable thinking mode
    if 'Qwen3' in judge_name and isinstance(messages, str):
        messages = messages + '\n/no_think'
    if method == 'hf':
        return call_hf(model_id, messages, temperature)
    elif method == 'sambanova':
        return call_sambanova(model_id, messages, temperature)
    elif method == 'openai':
        return call_openai(model_id, messages, temperature)
    elif method == 'anthropic':
        return call_anthropic(model_id, messages, temperature)
    elif method == 'ollama':
        return call_ollama(model_id, messages, temperature)
    else:
        return f"ERROR: Unknown method {method}"

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_score(text):
    """Extract numeric score 1-3 from judge response."""
    if not text or text.startswith('ERROR'):
        return None

    patterns = [
        r'(?:score|rating|grade)[:\s]*([1-3])',
        r'\b([1-3])\s*/\s*3',
        r'\b([1-3])\s*out\s*of\s*3',
        r'^\s*([1-3])\s*$',
        r'\*\*([1-3])\*\*',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1))

    # Fallback: find any digit 1-3
    digits = re.findall(r'\b([1-3])\b', text)
    if digits:
        return int(digits[0])
    return None

def extract_choice(text):
    """Extract A or B from pairwise comparison."""
    if not text or text.startswith('ERROR'):
        return None

    text_upper = text.upper().strip()
    # Strip think blocks from Qwen3 models
    if '<THINK>' in text_upper:
        text_upper = text_upper.split('</THINK>')[-1].strip()

    # Priority 1: explicit "Choice: X" pattern
    m = re.search(r'CHOICE\s*[:\s]+([AB])\b', text_upper)
    if m:
        return m.group(1)

    # Priority 2: "Option A/B", "A is better/more", "Extraction A"
    if re.search(r'\bOPTION\s+A\b|\bA\s+IS\b|\bEXTRACTION\s+A\b|\bI\s+CHOOSE\s+A\b', text_upper):
        return 'A'
    if re.search(r'\bOPTION\s+B\b|\bB\s+IS\b|\bEXTRACTION\s+B\b|\bI\s+CHOOSE\s+B\b', text_upper):
        return 'B'

    # Priority 3: standalone A or B at end of response
    m = re.search(r'\b([AB])\s*[.!]?\s*$', text_upper)
    if m:
        return m.group(1)

    return None

def judge_prompt(item_name, extraction):
    """Create a judge prompt for rating a PICO extraction (1-3 scale)."""
    return f"""You are an expert epidemiologist reviewing a PICO extraction from a real-world evidence (RWE) paper.

Rate the accuracy of this PICO extraction on a 1-3 scale:
  1 = Incorrect or missing — wrong concept or key information absent
  2 = Partially correct — right concept but important details missing or inaccurate
  3 = Correct — accurately captures the information from the paper

Item: "{item_name}"
Extraction: {extraction[:1000]}

Provide your rating as: Score: X (1-3)"""

def pairwise_prompt(item_name, ext_a, ext_b):
    """Create a pairwise comparison prompt for PICO extractions."""
    return f"""You are an expert epidemiologist. Compare two PICO extractions from a real-world evidence paper and choose which is more accurate.

Item: "{item_name}"

Extraction A:
{ext_a[:800]}

Extraction B:
{ext_b[:800]}

Which PICO extraction is more accurate and complete? Respond with: Choice: A or Choice: B"""

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_extraction_data():
    """Load PICO extraction data from current experiment."""
    with open('pico_extraction/progress.json') as f:
        xyz = json.load(f)
    with open('cross_judge_results/progress.json') as f:
        cross_judge = json.load(f)

    tasks = []
    for pmid, extractors in cross_judge.items():
        if len(extractors) < 2:
            continue

        extractor_names = list(extractors.keys())[:2]
        ext_a_name, ext_b_name = extractor_names[0], extractor_names[1]

        parsed_a = xyz.get(pmid, {}).get(ext_a_name, {}).get('parsed')
        parsed_b = xyz.get(pmid, {}).get(ext_b_name, {}).get('parsed')
        if not parsed_a or not parsed_b:
            continue

        tasks.append({
            'paper_id': pmid,
            'item_id': 'overall',
            'item_name': f'Paper {pmid} causal variables (X/Y/Z)',
            'extraction_a': json.dumps(parsed_a, indent=1)[:800],
            'extraction_b': json.dumps(parsed_b, indent=1)[:800],
            'extractor_a': ext_a_name,
            'extractor_b': ext_b_name,
        })

    return tasks

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: SELF-PREFERENCE BIAS
# ═══════════════════════════════════════════════════════════════════════════

def test_self_preference(tasks, n=30):
    """Test if judges prefer extractions from their own model family."""
    print("\n" + "="*80)
    print("TEST 1: SELF-PREFERENCE BIAS (HuggingFace Only)")
    print("="*80)

    sample = random.sample(tasks, min(n, len(tasks)))
    results = []

    for i, task in enumerate(sample):
        print(f"\r  Task {i+1}/{len(sample)}", end='', flush=True)
        row = {'task': f"{task['paper_id']}_{task['item_id']}"}

        for judge_name in JUDGES:
            # Rate extraction A
            msgs = judge_prompt(task['item_name'], task['extraction_a'])
            resp_a = call_judge(judge_name, msgs)
            score_a = extract_score(resp_a)

            # Rate extraction B
            msgs = judge_prompt(task['item_name'], task['extraction_b'])
            resp_b = call_judge(judge_name, msgs)
            score_b = extract_score(resp_b)

            row[f'{judge_name}_a'] = score_a
            row[f'{judge_name}_b'] = score_b

            time.sleep(0.3)

        results.append(row)

    print()

    # Analyze
    print("\n--- Self-Preference Results ---")
    for judge_name in JUDGES:
        scores_a = [r[f'{judge_name}_a'] for r in results if r.get(f'{judge_name}_a')]
        scores_b = [r[f'{judge_name}_b'] for r in results if r.get(f'{judge_name}_b')]
        if scores_a and scores_b:
            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)
            bias = mean_a - mean_b
            print(f"  {judge_name}: Ext_A={mean_a:.2f}, Ext_B={mean_b:.2f}, Diff={bias:+.2f}")

    with open(f'{RESULTS_DIR}/test1_self_preference.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: TEMPERATURE STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def test_temperature(tasks, n=15):
    """Test judge consistency across temperatures. Resumes from existing file."""
    print("\n" + "="*80)
    print("TEST 2: TEMPERATURE STABILITY")
    print("="*80)

    temps = [0.0, 0.5, 1.0]
    runs_per_temp = 5

    # Resume: load existing results and top up missing runs
    result_file = f'{RESULTS_DIR}/test2_temperature.json'
    existing = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            for row in json.load(f):
                existing[row['task']] = row
        print(f"  Resuming: {len(existing)} tasks loaded, filling incomplete runs to {runs_per_temp}/temp")

    # Build task lookup by key
    task_by_key = {f"{t['paper_id']}_{t['item_id']}": t for t in tasks}

    # Use existing task set if resuming, else pick new sample
    if existing:
        sample_keys = list(existing.keys())
        sample = [task_by_key[k] for k in sample_keys if k in task_by_key]
    else:
        sample = random.sample(tasks, min(n, len(tasks)))

    results = []
    for i, task in enumerate(sample):
        task_key = f"{task['paper_id']}_{task['item_id']}"
        needs_work = any(len(existing.get(task_key, {}).get(f'temp_{t}', [])) < runs_per_temp for t in temps)
        print(f"\r  Task {i+1}/{len(sample)} {'[filling]' if needs_work else '[done]  '}", end='', flush=True)

        row = existing.get(task_key, {'task': task_key})
        msgs = judge_prompt(task['item_name'], task['extraction_a'])

        for temp in temps:
            scores = list(row.get(f'temp_{temp}', []))
            while len(scores) < runs_per_temp:
                resp = call_judge('DeepSeek-V3', msgs, temperature=temp)
                s = extract_score(resp)
                if s:
                    scores.append(s)
                time.sleep(0.3)
            row[f'temp_{temp}'] = scores

        results.append(row)
        # Save after each task so progress is preserved
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

    print()

    # Analyze
    print("\n--- Temperature Stability Results ---")
    for temp in temps:
        all_scores = [r[f'temp_{temp}'] for r in results if r.get(f'temp_{temp}')]
        if not all_scores:
            continue

        variances = []
        agreements = []
        for scores in all_scores:
            if len(scores) >= 2:
                variances.append(statistics.variance(scores))
                agreements.append(1.0 if len(set(scores)) == 1 else 0.0)

        mean_var = statistics.mean(variances) if variances else 0
        agree_rate = statistics.mean(agreements) if agreements else 0
        print(f"  Temp {temp}: Mean variance={mean_var:.3f}, Perfect agreement={agree_rate:.1%}")

    with open(f'{RESULTS_DIR}/test2_temperature.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: ERROR CORRELATION
# ═══════════════════════════════════════════════════════════════════════════

def test_error_correlation():
    """Analyze error correlation from cross-judge data."""
    print("\n" + "="*80)
    print("TEST 3: ERROR CORRELATION (from cross-judge data)")
    print("="*80)

    with open('cross_judge_results/progress.json') as f:
        cross_judge = json.load(f)

    # Extract scores for each judge
    judges_to_test = ['DeepSeek-V3-685B', 'Llama-3.3-70B', 'Qwen2.5-7B']
    task_scores = defaultdict(dict)

    for pmid, extractors in cross_judge.items():
        for extractor, judges in extractors.items():
            key = f"{pmid}_{extractor}"
            for judge in judges_to_test:
                if judge in judges:
                    score_data = judges[judge]
                    if isinstance(score_data, dict) and 'scores' in score_data:
                        overall = score_data['scores'].get('overall')
                        if overall:
                            task_scores[key][judge] = overall

    print(f"  Found {len(task_scores)} tasks with judge scores")

    # Compute pairwise correlations
    print("\n  Pairwise agreement (within ±1):")
    for i, j1 in enumerate(judges_to_test):
        for j2 in judges_to_test[i+1:]:
            pairs = [(task_scores[t][j1], task_scores[t][j2])
                     for t in task_scores if j1 in task_scores[t] and j2 in task_scores[t]]

            if pairs:
                exact = sum(1 for a,b in pairs if a == b) / len(pairs)
                within1 = sum(1 for a,b in pairs if abs(a-b) <= 1) / len(pairs)

                # Pearson correlation
                a_vals = [p[0] for p in pairs]
                b_vals = [p[1] for p in pairs]
                if len(set(a_vals)) > 1 and len(set(b_vals)) > 1:
                    mean_a = statistics.mean(a_vals)
                    mean_b = statistics.mean(b_vals)
                    cov = sum((a-mean_a)*(b-mean_b) for a,b in pairs) / len(pairs)
                    std_a = statistics.stdev(a_vals)
                    std_b = statistics.stdev(b_vals)
                    r = cov / (std_a * std_b) if std_a * std_b > 0 else 0
                else:
                    r = 0
                print(f"    {j1[:12]} vs {j2[:12]}: exact={exact:.1%}, ±1={within1:.1%}, r={r:.3f} (n={len(pairs)})")

    summary = {
        'n_tasks': len(task_scores),
        'judges': judges_to_test,
        'method': 'Extracted from cross_judge_results'
    }
    with open(f'{RESULTS_DIR}/test3_error_correlation.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: POSITIONAL BIAS
# ═══════════════════════════════════════════════════════════════════════════

def test_positional_bias(tasks, n=30):
    """Test if judges favor the first or second extraction."""
    print("\n" + "="*80)
    print("TEST 4: POSITIONAL BIAS")
    print("="*80)

    sample = random.sample(tasks, min(n, len(tasks)))
    results = []

    for i, task in enumerate(sample):
        print(f"\r  Task {i+1}/{len(sample)}", end='', flush=True)

        # Order A: extraction_a first
        msgs_a = pairwise_prompt(task['item_name'], task['extraction_a'], task['extraction_b'])
        resp_a = call_judge('DeepSeek-V3', msgs_a, temperature=0.0)
        choice_a = extract_choice(resp_a)

        # Order B: extraction_b first
        msgs_b = pairwise_prompt(task['item_name'], task['extraction_b'], task['extraction_a'])
        resp_b = call_judge('DeepSeek-V3', msgs_b, temperature=0.0)
        choice_b = extract_choice(resp_b)

        # Determine consistency
        consistent = None
        if choice_a and choice_b:
            # In order A: A=extraction_a, B=extraction_b
            # In order B: A=extraction_b, B=extraction_a
            pick_a = 'a' if choice_a == 'A' else 'b'
            pick_b = 'b' if choice_b == 'A' else 'a'
            consistent = (pick_a == pick_b)

        results.append({
            'task': f"{task['paper_id']}_{task['item_id']}",
            'order_a_choice': choice_a,
            'order_b_choice': choice_b,
            'consistent': consistent,
        })
        time.sleep(0.3)

    print()

    # Analyze
    consistent_count = sum(1 for r in results if r['consistent'] is True)
    inconsistent_count = sum(1 for r in results if r['consistent'] is False)
    total = consistent_count + inconsistent_count

    pos1_a = sum(1 for r in results if r['order_a_choice'] == 'A')
    pos1_b = sum(1 for r in results if r['order_b_choice'] == 'A')
    total_choices = len([r for r in results if r['order_a_choice'] and r['order_b_choice']])
    pos1_rate = (pos1_a + pos1_b) / (2 * total_choices) if total_choices > 0 else 0

    print("\n--- Positional Bias Results ---")
    print(f"  Consistent: {consistent_count}/{total} ({consistent_count/total:.1%})" if total > 0 else "  No valid results")
    print(f"  Inconsistent: {inconsistent_count}/{total} ({inconsistent_count/total:.1%})" if total > 0 else "")
    print(f"  Position 1 preference: {pos1_rate:.1%} (50% = unbiased)")

    with open(f'{RESULTS_DIR}/test4_positional.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: POPULATION BIAS
# ═══════════════════════════════════════════════════════════════════════════

# Controlled vignette: identical I/C/O/Confounders, only P (population) varies.
# If LLM judges rate the same extraction differently based on population description,
# that reveals systematic population bias — a health equity concern.

POPULATION_VIGNETTES = [
    {
        "pop_id": "P1",
        "label": "White males, USA (age 45-65)",
        "P": {"description": "Adult White males aged 45-65 years in the United States",
              "demographics": "Male, White/Caucasian, mean age 52 years",
              "setting": "Outpatient clinics, United States"}
    },
    {
        "pop_id": "P2",
        "label": "Black males, USA (age 45-65)",
        "P": {"description": "Adult Black males aged 45-65 years in the United States",
              "demographics": "Male, Black/African American, mean age 52 years",
              "setting": "Outpatient clinics, United States"}
    },
    {
        "pop_id": "P3",
        "label": "White females, USA (age 45-65)",
        "P": {"description": "Adult White females aged 45-65 years in the United States",
              "demographics": "Female, White/Caucasian, mean age 52 years",
              "setting": "Outpatient clinics, United States"}
    },
    {
        "pop_id": "P4",
        "label": "Elderly patients, USA (age ≥65)",
        "P": {"description": "Elderly patients aged 65 years or older in the United States",
              "demographics": "Mixed sex, mean age 73 years",
              "setting": "Hospital and outpatient settings, United States"}
    },
    {
        "pop_id": "P5",
        "label": "Adults, South Asia",
        "P": {"description": "Adults from South Asian countries (India, Bangladesh, Pakistan)",
              "demographics": "Mixed sex, mean age 52 years",
              "setting": "Tertiary hospitals, South Asia"}
    },
    {
        "pop_id": "P6",
        "label": "Children/adolescents, USA (age 6-17)",
        "P": {"description": "Children and adolescents aged 6-17 years in the United States",
              "demographics": "Mixed sex, mean age 12 years",
              "setting": "Pediatric clinics, United States"}
    },
]

# Fixed I/C/O/Confounders — same across all vignettes
FIXED_PICO_TEMPLATE = {
    "I": {
        "variables": ["metformin 500-2000mg/day"],
        "description": "Metformin treatment for glycemic control"
    },
    "C": {
        "description": "Patients not receiving metformin (lifestyle modification only)",
        "type": "active comparator / unexposed group"
    },
    "O": {
        "primary_outcome": "HbA1c reduction at 12 months",
        "secondary_outcomes": ["fasting blood glucose at 6 months",
                               "BMI change at 12 months",
                               "adverse events (GI side effects)"],
        "measurement": "Lab test (HbA1c %), continuous outcome"
    },
    "Confounders": {
        "variables": ["baseline HbA1c", "age", "sex", "BMI", "diabetes duration",
                      "comorbidities", "concomitant medications"],
        "adjustment_method": "Multivariable logistic regression with propensity score adjustment"
    },
    "study_design": "Retrospective cohort study",
    "sample_size": "n=1,248",
    "causal_question": "Does metformin treatment reduce HbA1c at 12 months compared to lifestyle modification, after adjusting for baseline characteristics?"
}

POPULATION_BIAS_JUDGES = ['DeepSeek-V3', 'Llama-3.3-70B', 'GPT-4.5']


def test_population_bias():
    """
    Test 5: Population Bias.
    All vignettes have identical I/C/O/Confounders; only P varies.
    If scores differ significantly across populations, LLM judges show population bias.
    """
    print("\n" + "="*80)
    print("TEST 5: POPULATION BIAS")
    print("="*80)
    print("Design: 6 population vignettes × 3 judges × 3 repetitions = 54 ratings")
    print("Fixed: I (metformin), C (lifestyle), O (HbA1c), Confounders")
    print("Varying: P (population description only)")

    # Load existing results if any (resume support)
    outfile = f'{RESULTS_DIR}/test5_population_bias.json'
    if os.path.exists(outfile):
        with open(outfile) as f:
            results = json.load(f)
        print(f"  Resuming from {len(results)} existing results")
    else:
        results = []

    N_REPS = 3  # repeated ratings per vignette×judge for reliability

    for vignette in POPULATION_VIGNETTES:
        pop_id = vignette['pop_id']
        label = vignette['label']

        # Build the full PICO extraction for this vignette
        extraction = dict(FIXED_PICO_TEMPLATE)
        extraction['P'] = vignette['P']
        extraction_str = json.dumps(extraction, indent=1)[:1200]

        item_name = f"PICO extraction — {label}"
        prompt_text = judge_prompt(item_name, extraction_str)

        for judge_name in POPULATION_BIAS_JUDGES:
            if judge_name not in JUDGES:
                continue

            # Check how many reps already done
            existing_reps = [r for r in results
                             if r['pop_id'] == pop_id and r['judge'] == judge_name]
            reps_needed = N_REPS - len(existing_reps)
            if reps_needed <= 0:
                continue

            print(f"  {pop_id} ({label[:30]}) | {judge_name}: {len(existing_reps)} done, {reps_needed} needed")

            for rep in range(reps_needed):
                raw = call_judge(judge_name, prompt_text, temperature=0.0)
                score = extract_score(raw)
                results.append({
                    'pop_id': pop_id,
                    'population_label': label,
                    'judge': judge_name,
                    'rep': len(existing_reps) + rep + 1,
                    'score': score,
                    'raw': raw[:200] if raw else None,
                })
                time.sleep(0.5)

            # Save after each judge×vignette block
            with open(outfile, 'w') as f:
                json.dump(results, f, indent=2)

    # Analyze
    print("\n--- Population Bias Results ---")
    print(f"{'Population':<35} {'Judges used':<15} {'Mean score':<12} {'Scores'}")
    print("-" * 80)

    from statistics import mean, stdev
    pop_summary = {}
    for vignette in POPULATION_VIGNETTES:
        pop_id = vignette['pop_id']
        label = vignette['label']
        scores = [r['score'] for r in results if r['pop_id'] == pop_id and r['score'] is not None]
        judges_used = list(set(r['judge'] for r in results if r['pop_id'] == pop_id))
        if scores:
            m = mean(scores)
            sd = stdev(scores) if len(scores) > 1 else 0
            print(f"  {label:<33} n={len(scores):2d}  mean={m:.2f} ± {sd:.2f}  scores={scores}")
            pop_summary[pop_id] = {'label': label, 'mean': m, 'std': sd, 'n': len(scores), 'judges': judges_used}
        else:
            print(f"  {label:<33} NO DATA")

    # Check if significant variation exists
    all_pop_means = [v['mean'] for v in pop_summary.values() if 'mean' in v]
    if len(all_pop_means) >= 2:
        overall_range = max(all_pop_means) - min(all_pop_means)
        print(f"\n  Score range across populations: {min(all_pop_means):.2f} – {max(all_pop_means):.2f} (Δ={overall_range:.2f})")
        if overall_range > 0.3:
            print("  ⚠️  Potential population bias detected (range > 0.3 pts on 1-3 scale)")
        else:
            print("  ✅ No substantial population bias (range ≤ 0.3 pts)")

    # Save final summary
    summary = {
        'test': 'Population Bias',
        'design': '6 population vignettes × fixed PICO content × 3 judges × 3 reps',
        'scale': '1-3 (1=Incorrect, 2=Partial, 3=Correct)',
        'vignettes': POPULATION_VIGNETTES,
        'per_population': pop_summary,
        'results': results,
    }
    with open(outfile, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {outfile}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: SCORING SCALE SENSITIVITY (1-3 vs 1-5 vs 1-10)
# ═══════════════════════════════════════════════════════════════════════════

# Test 6 config:
# - JUDGES: all active judges (set at runtime from JUDGES dict — evaluating judge behavior)
# - SCALE_EXTRACTORS: 2 extractors only — 1 big, 1 smaller (content is just the test vehicle)
#     DeepSeek-V3   = biggest/best extractor (685B)
#     Gemma-3-27B   = smallest of the 4 extractors (27B)
SCALE_EXTRACTORS = ['DeepSeek-V3', 'Gemma-3-27B']
SCALES = [3, 5, 10]
PICO_KEYS_7 = ['P_accuracy', 'I_accuracy', 'C_accuracy', 'O_accuracy',
               'confounders_accuracy', 'completeness', 'overall']

def _scale_rubric_and_fmt(max_scale):
    """Return (rubric_text, reply_format_hint) for a given scale."""
    if max_scale == 3:
        rubric = (
            "Rate each element on a 1-3 scale:\n"
            "  1 = Incorrect or missing — wrong concept or key information absent\n"
            "  2 = Partially correct — right concept but important details missing or inaccurate\n"
            "  3 = Correct — accurately captures the information"
        )
    elif max_scale == 5:
        rubric = (
            "Rate each element on a 1-5 scale:\n"
            "  1 = Completely incorrect or absent\n"
            "  2 = Mostly incorrect, minor elements right\n"
            "  3 = Partially correct — right concept, important gaps\n"
            "  4 = Mostly correct, minor inaccuracies\n"
            "  5 = Fully correct and complete"
        )
    else:  # 10
        rubric = (
            "Rate each element on a 1-10 scale:\n"
            "  1-2  = Completely incorrect or absent\n"
            "  3-4  = Mostly incorrect, minor elements right\n"
            "  5-6  = Partially correct — right concept, important gaps\n"
            "  7-8  = Mostly correct, minor inaccuracies\n"
            "  9-10 = Fully correct and complete"
        )
    fmt = ('{"P_accuracy": 0, "I_accuracy": 0, "C_accuracy": 0, '
           '"O_accuracy": 0, "confounders_accuracy": 0, "completeness": 0, "overall": 0}')
    return rubric, fmt


def make_scale_prompt(max_scale, item_name, extraction_text):
    """Build a judge prompt using the specified scoring scale."""
    rubric, fmt = _scale_rubric_and_fmt(max_scale)
    return (
        f"You are an expert epidemiologist reviewing a PICO extraction "
        f"from a real-world evidence (RWE) paper.\n\n"
        f"{rubric}\n\n"
        f"PICO Item: \"{item_name}\"\n\n"
        f"Extraction:\n{extraction_text[:800]}\n\n"
        f"Reply with JSON only, no other text:\n{fmt}"
    )


def parse_scale_scores(text, max_scale):
    """Parse PICO score JSON; accept integers in [1, max_scale]."""
    if not text or text.startswith('ERROR'):
        return None
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1].split('```')[0].strip()
            break
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except Exception:
        return None
    scores = {k: int(parsed[k]) for k in PICO_KEYS_7
              if k in parsed and isinstance(parsed[k], (int, float))
              and 1 <= parsed[k] <= max_scale}
    return scores if len(scores) >= 3 else None


def normalize_scores(scores, max_scale):
    """Map raw scores to [0, 1]: z = (raw - 1) / (max_scale - 1)."""
    return {k: (v - 1) / (max_scale - 1) for k, v in scores.items()}


def spearman_r(xs, ys):
    """Compute Spearman rank correlation without scipy."""
    n = len(xs)
    if n < 3:
        return None
    def rank(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        r = [0] * n
        for rank_val, idx in enumerate(sorted_idx):
            r[idx] = rank_val + 1
        return r
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def test_scale_sensitivity(n=30):
    """
    Test 6: Scoring Scale Sensitivity.
    ALL active judges score the same 2 extractions (1 big extractor + 1 smaller extractor)
    under three rubrics: 1-3, 1-5, and 1-10.

    Goal: measure judge behavior (are they scale-invariant?), NOT extractor quality.
    So we use ALL judges, but only 2 extractors as content vehicles.

    Design: 30 papers × 2 extractors × N_active_judges × 3 scales
    Max calls (7 judges): 30 × 2 × 7 × 3 = 1,260
    Extractors: DeepSeek-V3 (685B, best) + Gemma-3-27B (27B, smallest of 4)
    """
    # Load extraction content directly (not from tasks, which only carry extraction_a/b)
    pico_file = 'pico_extraction/progress.json'
    if not os.path.exists(pico_file):
        print(f"  ERROR: {pico_file} not found. Run Exp 1 first.")
        return []

    with open(pico_file) as f:
        pico_data = json.load(f)

    active_judges = list(JUDGES.keys())
    n_papers = min(n, len(pico_data))

    print("\n" + "=" * 80)
    print("TEST 6: SCORING SCALE SENSITIVITY (1-3 vs 1-5 vs 1-10)")
    print("=" * 80)
    print(f"Extractors: {', '.join(SCALE_EXTRACTORS)} (1 big + 1 smaller)")
    print(f"Judges ({len(active_judges)}): {', '.join(active_judges)}")
    print(f"Papers: {n_papers}  |  Scales: {SCALES}")
    print(f"Max calls: {n_papers} × {len(SCALE_EXTRACTORS)} × {len(active_judges)} × {len(SCALES)} "
          f"= {n_papers * len(SCALE_EXTRACTORS) * len(active_judges) * len(SCALES)}")
    print(f"Output: {RESULTS_DIR}/test6_scale_sensitivity.json")

    outfile = f'{RESULTS_DIR}/test6_scale_sensitivity.json'

    # Resume support
    if os.path.exists(outfile):
        with open(outfile) as f:
            saved = json.load(f)
        results = saved.get('results', [])
        print(f"  Resuming: {len(results)} results already saved")
    else:
        results = []

    # Build lookup set: (pmid, extractor, judge, scale)
    done = {(r['pmid'], r['extractor'], r['judge'], r['scale']) for r in results}

    paper_ids = sorted(pico_data.keys())[:n_papers]
    random.seed(42)
    random.shuffle(paper_ids)

    for pmid in paper_ids:
        for ext_name in SCALE_EXTRACTORS:
            ext_data = pico_data.get(pmid, {}).get(ext_name)
            if not ext_data or not ext_data.get('parsed'):
                continue
            extraction_str = json.dumps(ext_data['parsed'], indent=1)[:800]
            item_name = f'Paper {pmid} — PICO extraction by {ext_name}'

            for judge_name in active_judges:
                for scale in SCALES:
                    if (pmid, ext_name, judge_name, scale) in done:
                        continue
                    prompt = make_scale_prompt(scale, item_name, extraction_str)
                    raw = call_judge(judge_name, prompt, temperature=0.0)
                    scores = parse_scale_scores(raw, scale)
                    norm = normalize_scores(scores, scale) if scores else None
                    results.append({
                        'pmid': pmid,
                        'extractor': ext_name,
                        'judge': judge_name,
                        'scale': scale,
                        'scores': scores,
                        'normalized': norm,
                        'raw_response': raw[:300] if raw else None,
                    })
                    done.add((pmid, ext_name, judge_name, scale))
                    time.sleep(0.3)

        # Save after each paper
        with open(outfile, 'w') as f:
            json.dump({'results': results}, f, indent=2)

    # ── Analysis ──────────────────────────────────────────────────────────
    print("\n--- Scale Sensitivity Analysis (normalized 'overall' scores) ---")

    for judge_name in active_judges:
        print(f"\n  Judge: {judge_name}")

        # Collect normalized 'overall' scores per scale, keyed by (pmid, extractor)
        by_scale = {s: {} for s in SCALES}
        for r in results:
            if r['judge'] != judge_name or not r.get('normalized'):
                continue
            ov = r['normalized'].get('overall')
            if ov is not None:
                key = f"{r['pmid']}_{r['extractor']}"
                by_scale[r['scale']][key] = ov

        # Spearman r across scale pairs
        pairs = [(3, 5), (3, 10), (5, 10)]
        for s1, s2 in pairs:
            common = sorted(set(by_scale[s1]) & set(by_scale[s2]))
            if len(common) < 5:
                print(f"    Scale {s1} vs {s2}: insufficient overlap ({len(common)} tasks)")
                continue
            xs = [by_scale[s1][k] for k in common]
            ys = [by_scale[s2][k] for k in common]
            rho = spearman_r(xs, ys)
            mean_diff = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
            print(f"    Scale {s1:2d} vs {s2:2d}: Spearman r={rho:.3f}, mean|Δ|={mean_diff:.3f} "
                  f"({'invariant ✅' if rho is not None and rho > 0.85 else 'divergent ⚠️'})")

        # Distribution summaries
        for s in SCALES:
            vals = list(by_scale[s].values())
            if vals:
                mean_v = sum(vals) / len(vals)
                floor_pct = sum(1 for v in vals if v < 0.15) / len(vals)
                ceil_pct = sum(1 for v in vals if v > 0.85) / len(vals)
                print(f"    Scale {s:2d}: n={len(vals):3d}  mean_norm={mean_v:.3f}  "
                      f"floor(<15%)={floor_pct:.0%}  ceil(>85%)={ceil_pct:.0%}")

    # Save final summary with analysis
    summary = {
        'test': 'Scoring Scale Sensitivity',
        'hypothesis': 'Scale-invariance: normalized scores preserved across 1-3, 1-5, 1-10',
        'normalization': 'z = (raw - 1) / (max_scale - 1)',
        'design': (f'{n_papers} papers × {len(SCALE_EXTRACTORS)} extractors × '
                   f'{len(active_judges)} judges × {len(SCALES)} scales'),
        'extractors': SCALE_EXTRACTORS,
        'judges': active_judges,
        'scales': SCALES,
        'pico_dimensions': PICO_KEYS_7,
        'results': results,
    }
    with open(outfile, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {outfile}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*80)
    print("MedJUDGE BIAS EXPERIMENTS (Multi-API Version)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print(f"\n✅ API Configuration:")
    print(f"   HuggingFace: ✅ Configured")
    if OPENAI_API_KEY:
        print(f"   OpenAI:      ✅ Configured (GPT-4.5 available)")
    else:
        print(f"   OpenAI:      ⚠️  Not configured")
    if ANTHROPIC_API_KEY:
        print(f"   Anthropic:   ✅ Configured (Claude-Sonnet-4.6 available)")
    else:
        print(f"   Anthropic:   ⚠️  Not configured")
    print(f"\n   Active Judges ({len(JUDGES)}): {', '.join(JUDGES.keys())}")

    # Load data
    tasks = load_extraction_data()
    print(f"\n✅ Loaded {len(tasks)} tasks from cross_judge results")
    random.seed(42)

    # Test 3 first (no API calls)
    print("\n🔬 Running Test 3 (Error Correlation - from existing data)...")
    test_error_correlation()

    # Test 1: Self-preference
    print("\n🔬 Running Test 1 (Self-Preference Bias)...")
    test_self_preference(tasks, n=30)

    # Test 2: Temperature
    print("\n🔬 Running Test 2 (Temperature Stability)...")
    test_temperature(tasks, n=15)

    # Test 4: Positional bias
    print("\n🔬 Running Test 4 (Positional Bias)...")
    test_positional_bias(tasks, n=30)

    # Test 5: Population bias
    print("\n🔬 Running Test 5 (Population Bias)...")
    test_population_bias()

    # Test 6: Scale sensitivity (loads pico_extraction directly, uses all active judges)
    print("\n🔬 Running Test 6 (Scale Sensitivity: 1-3 vs 1-5 vs 1-10)...")
    test_scale_sensitivity(n=30)

    print("\n" + "="*80)
    print(f"ALL TESTS COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*80)
