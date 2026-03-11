#!/usr/bin/env python3
"""Cross-Judge: All models judge all extractors. All via HuggingFace API."""
import os, json, time, random, re, requests
from datetime import datetime
from statistics import mean, stdev
from collections import defaultdict
from huggingface_hub import InferenceClient

TOKEN = os.getenv('HF_TOKEN', '')
client = InferenceClient(token=TOKEN)
RESULTS_DIR = 'cross_judge_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    # Big models (≥27B parameters)
    'DeepSeek-V3-685B':   ('hf', 'deepseek-ai/DeepSeek-V3-0324'),
    'Kimi-K2-1T':         ('hf', 'moonshotai/Kimi-K2-Instruct'),
    'Llama-3.3-70B':      ('hf', 'meta-llama/Llama-3.3-70B-Instruct'),
    'Gemma-3-27B':        ('hf', 'google/gemma-3-27b-it'),
    'Qwen3-32B':          ('sambanova', 'Qwen3-32B'),
    # Small models (<10B parameters)
    'Llama-3.1-8B':       ('sambanova', 'Meta-Llama-3.1-8B-Instruct'),
    'Llama-3.2-3B':       ('together', 'meta-llama/Llama-3.2-3B-Instruct-Turbo'),
    'Qwen2.5-7B':         ('together', 'Qwen/Qwen2.5-7B-Instruct-Turbo'),
    'Qwen3-0.6B':         ('ollama', 'qwen3:0.6b'),
}

BIG = ['DeepSeek-V3-685B', 'Kimi-K2-1T', 'Llama-3.3-70B', 'Gemma-3-27B', 'Qwen3-32B']
SMALL = ['Llama-3.1-8B', 'Llama-3.2-3B', 'Qwen2.5-7B', 'Qwen3-0.6B']
ALL_JUDGES = BIG + SMALL

def call(name, prompt):
    method, model_id = MODELS[name]
    # Disable thinking mode for Qwen3 models to get clean JSON output
    if 'Qwen3' in name:
        prompt = prompt + '\n/no_think'
    for attempt in range(2):
        try:
            if method == 'ollama':
                r = requests.post(
                    'http://localhost:11434/v1/chat/completions',
                    headers={'Content-Type': 'application/json'},
                    json={'model': model_id,
                          'messages': [{'role': 'user', 'content': prompt}],
                          'max_tokens': 400, 'temperature': 0.1},
                    timeout=120)
                if r.status_code == 200:
                    c = r.json()['choices'][0]['message']['content']
                    if '<think>' in c:
                        c = c.split('</think>')[-1].strip()
                    return c
                time.sleep(2)
            elif method == 'hf':
                r = client.chat_completion(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=model_id, max_tokens=400, temperature=0.1)
                c = r.choices[0].message.content
                if '<think>' in c:
                    c = c.split('</think>')[-1].strip()
                return c
            else:
                r = requests.post(
                    f'https://router.huggingface.co/{method}/v1/chat/completions',
                    headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'},
                    json={'model': model_id,
                          'messages': [{'role': 'user', 'content': prompt}],
                          'max_tokens': 400, 'temperature': 0.1},
                    timeout=60)
                if r.status_code == 200:
                    c = r.json()['choices'][0]['message']['content']
                    if '<think>' in c:
                        c = c.split('</think>')[-1].strip()
                    return c
                time.sleep(2)
        except Exception as e:
            if attempt == 0:
                time.sleep(3)
            else:
                return f'ERROR: {e}'
    return None

PICO_JUDGE_PROMPT = """\
You are an expert epidemiologist reviewing a PICO extraction from a real-world evidence (RWE) paper.

Rate each element on a 1-3 scale:
  1 = Incorrect or missing — wrong concept or key information absent
  2 = Partially correct — right concept but important details missing or inaccurate
  3 = Correct — accurately captures the information from the paper

PAPER:
{paper_text}

EXTRACTION (by {extractor}):
{extraction_json}

Reply with JSON only, no other text:
{{"P_accuracy": 0, "I_accuracy": 0, "C_accuracy": 0, "O_accuracy": 0, "confounders_accuracy": 0, "completeness": 0, "overall": 0}}"""

PICO_KEYS = ['P_accuracy', 'I_accuracy', 'C_accuracy', 'O_accuracy', 'confounders_accuracy', 'completeness', 'overall']

def parse_json(text):
    if not text or text.startswith('ERROR'):
        return None
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1].split('```')[0]
            break
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end+1])
            # Accept if it has at least 3 of the expected PICO keys
            if sum(1 for k in PICO_KEYS if k in parsed) >= 3:
                return parsed
        except:
            pass
    # Regex fallback for PICO keys
    scores = {}
    for key in PICO_KEYS:
        m = re.search(rf'{key}["\s:]+([1-3])', text)
        if m:
            scores[key] = int(m.group(1))
    return scores if len(scores) >= 3 else None

# Load data
with open('pico_extraction/progress.json') as f:
    extractions = json.load(f)
papers = {}
for fn in sorted(os.listdir('papers/fulltext')):
    pmid = fn.replace('.txt', '')
    with open(f'papers/fulltext/{fn}') as fh:
        papers[pmid] = fh.read()

random.seed(42)
sample = list(papers.keys())  # Use all available papers (30 for Phase 1)
ext_models = ['DeepSeek-V3', 'Kimi-K2', 'Llama-3.3-70B', 'Gemma-3-27B']

# Load progress
progress_file = f'{RESULTS_DIR}/progress.json'
results = {}
if os.path.exists(progress_file):
    with open(progress_file) as f:
        results = json.load(f)

total_expected = len(sample) * len(ext_models) * len(ALL_JUDGES)
print(f'CROSS-JUDGE (all via HF API)')
print(f'{len(sample)} papers x {len(ext_models)} extractors x {len(ALL_JUDGES)} judges = {total_expected}')
print(f'Started: {datetime.now():%H:%M:%S}')


for i, pmid in enumerate(sample):
    text = papers[pmid][:3000]
    if pmid not in results:
        results[pmid] = {}
    
    print(f'\n[{i+1}/{len(sample)}] {pmid}')
    
    for ext_name in ext_models:
        ext_data = extractions.get(pmid, {}).get(ext_name, {}).get('parsed')
        if not ext_data:
            continue
        if ext_name not in results[pmid]:
            results[pmid][ext_name] = {}
        
        ext_json = json.dumps(ext_data, indent=1)[:1200]

        prompt = PICO_JUDGE_PROMPT.format(
            paper_text=text[:3000],
            extractor=ext_name,
            extraction_json=ext_json
        )
        
        for judge_name in ALL_JUDGES:
            if judge_name in results[pmid][ext_name]:
                existing = results[pmid][ext_name][judge_name]
                if isinstance(existing, dict) and existing.get('success'):
                    continue
            
            raw = call(judge_name, prompt)
            scores = parse_json(raw)
            results[pmid][ext_name][judge_name] = {
                'scores': scores,
                'success': scores is not None
            }
            tag = 'B' if judge_name in BIG else 'S'
            status = 'OK' if scores else 'FAIL'
            print(f'  [{tag}] {ext_name[:10]:10s}<-{judge_name[:12]:12s} {status}')
            print(f'    RAW: {raw or "None"}')
            if scores:
                print(f'    SCORES: {scores}')
            time.sleep(0.3)
    
    # Save after each paper (makedirs ensures dir exists even if deleted mid-run)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*80}')
print('ANALYSIS')
print(f'{"="*80}')

big_s, small_s = [], []
judge_scores = defaultdict(list)
dim_big = defaultdict(list)
dim_small = defaultdict(list)

for pmid, exts in results.items():
    for ext, judges in exts.items():
        for j, data in judges.items():
            if not isinstance(data, dict):
                continue
            sc = data.get('scores')
            if not sc or not isinstance(sc, dict):
                continue
            o = sc.get('overall')
            if o:
                judge_scores[j].append(o)
                if j in BIG:
                    big_s.append(o)
                elif j in SMALL:
                    small_s.append(o)
            for dim in PICO_KEYS:
                v = sc.get(dim)
                if v:
                    if j in BIG:
                        dim_big[dim].append(v)
                    elif j in SMALL:
                        dim_small[dim].append(v)

print(f'\n1. BIG vs SMALL MODELS')
if big_s:
    print(f'  Big (27-685B):  mean={mean(big_s):.2f} std={stdev(big_s):.2f} n={len(big_s)}')
if small_s:
    print(f'  Small (3-8B):   mean={mean(small_s):.2f} std={stdev(small_s):.2f} n={len(small_s)}')
if big_s and small_s:
    print(f'  Gap: {mean(big_s)-mean(small_s):+.2f}')

print(f'\n2. PER JUDGE MEAN SCORES')
for j in ALL_JUDGES:
    tag = 'BIG' if j in BIG else 'SML'
    v = judge_scores[j]
    if v:
        print(f'  [{tag}] {j:20s}: mean={mean(v):.2f} std={stdev(v):.2f} n={len(v)}')

print(f'\n3. BIG vs SMALL BY PICO DIMENSION')
for dim in PICO_KEYS:
    b, s = dim_big[dim], dim_small[dim]
    if b and s:
        print(f'  {dim:25s}: Big={mean(b):.2f} Small={mean(s):.2f} gap={mean(b)-mean(s):+.2f}')

print(f'\n4. INTER-JUDGE AGREEMENT')
for j1 in ALL_JUDGES:
    for j2 in ALL_JUDGES:
        if j1 >= j2:
            continue
        pairs = []
        for pmid, exts in results.items():
            for ext, judges in exts.items():
                s1 = None
                s2 = None
                d1 = judges.get(j1)
                d2 = judges.get(j2)
                if isinstance(d1, dict) and d1.get('scores'):
                    s1 = d1['scores'].get('overall')
                if isinstance(d2, dict) and d2.get('scores'):
                    s2 = d2['scores'].get('overall')
                if s1 and s2:
                    pairs.append((s1, s2))
        if len(pairs) >= 5:
            exact = sum(1 for a, b in pairs if a == b) / len(pairs)
            w1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
            t1 = 'B' if j1 in BIG else 'S'
            t2 = 'B' if j2 in BIG else 'S'
            print(f'  [{t1}]{j1[:12]:12s} vs [{t2}]{j2[:12]:12s}: exact={exact:.0%} within1={w1:.0%} n={len(pairs)}')

print(f'\n5. SELF-PREFERENCE CHECK')
for j in ALL_JUDGES:
    family = j.split('-')[0]
    own, other = [], []
    for pmid, exts in results.items():
        for ext, judges in exts.items():
            d = judges.get(j)
            if not isinstance(d, dict) or not d.get('scores'):
                continue
            s = d['scores'].get('overall')
            if s:
                if family.lower() in ext.lower():
                    own.append(s)
                else:
                    other.append(s)
    if own and other:
        print(f'  {j:20s}: own={mean(own):.2f}(n={len(own)}) other={mean(other):.2f}(n={len(other)}) bias={mean(own)-mean(other):+.2f}')
    elif other:
        print(f'  {j:20s}: no self-eval, other={mean(other):.2f}(n={len(other)})')

# Save analysis
analysis = {
    'big_mean': mean(big_s) if big_s else 0,
    'small_mean': mean(small_s) if small_s else 0,
    'gap': (mean(big_s) - mean(small_s)) if big_s and small_s else 0,
    'n_big': len(big_s),
    'n_small': len(small_s),
    'per_judge': {j: {'mean': mean(v), 'std': stdev(v), 'n': len(v)}
                  for j, v in judge_scores.items() if v},
    'timestamp': datetime.now().isoformat()
}
with open(f'{RESULTS_DIR}/analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'\nSaved to {RESULTS_DIR}/')
print(f'Done: {datetime.now():%H:%M:%S}')
