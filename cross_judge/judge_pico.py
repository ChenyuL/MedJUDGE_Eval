#!/usr/bin/env python3
"""
Cross-Judge Evaluation of PICO Extractions
30 papers × 4 extractors × 7 judges = 840 judgments
+ IRR analysis
"""
import os, json, time, random, re, requests, sys
from datetime import datetime
from collections import defaultdict
import numpy as np
from huggingface_hub import InferenceClient

# Token
TOKEN = None
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                TOKEN = line.strip().split('=', 1)[1]
if not TOKEN:
    TOKEN = os.getenv('HF_TOKEN', '')

client = InferenceClient(token=TOKEN)

# Output
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'pico_judge_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    'DeepSeek-V3-685B': ('hf', 'deepseek-ai/DeepSeek-V3-0324'),
    'Kimi-K2-1T':       ('hf', 'moonshotai/Kimi-K2-Instruct'),
    'Llama-3.3-70B':    ('hf', 'meta-llama/Llama-3.3-70B-Instruct'),
    'Gemma-3-27B':      ('hf', 'google/gemma-3-27b-it'),
    'Llama-3.1-8B':     ('sambanova', 'Meta-Llama-3.1-8B-Instruct'),
    'Llama-3.2-3B':     ('together', 'meta-llama/Llama-3.2-3B-Instruct-Turbo'),
    'Qwen2.5-7B':       ('together', 'Qwen/Qwen2.5-7B-Instruct-Turbo'),
}
BIG = ['DeepSeek-V3-685B', 'Kimi-K2-1T', 'Llama-3.3-70B', 'Gemma-3-27B']
SMALL = ['Llama-3.1-8B', 'Llama-3.2-3B', 'Qwen2.5-7B']
ALL_JUDGES = BIG + SMALL
EXT_MODELS = ['DeepSeek-V3', 'Kimi-K2', 'Llama-3.3-70B', 'Gemma-3-27B']

def call(name, prompt):
    method, model_id = MODELS[name]
    for attempt in range(2):
        try:
            if method == 'hf':
                r = client.chat_completion(
                    messages=[{'role': 'user', 'content': prompt}],
                    model=model_id, max_tokens=400, temperature=0.1)
                return r.choices[0].message.content
            else:
                r = requests.post(
                    f'https://router.huggingface.co/{method}/v1/chat/completions',
                    headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'},
                    json={'model': model_id, 'messages': [{'role': 'user', 'content': prompt}],
                          'max_tokens': 400, 'temperature': 0.1},
                    timeout=60)
                if r.status_code == 200:
                    c = r.json()['choices'][0]['message']['content']
                    if '<think>' in c: c = c.split('</think>')[-1].strip()
                    return c
                time.sleep(2)
        except Exception as e:
            if attempt == 0: time.sleep(3)
            else: return f'ERROR: {e}'
    return None

def parse_json(text):
    if not text or text.startswith('ERROR'): return None
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1].split('```')[0]
            break
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
    scores = {}
    for key in ['P_accuracy', 'I_accuracy', 'C_accuracy', 'O_accuracy', 'overall']:
        m = re.search(rf'{key}["\s:]+(\d)', text)
        if m: scores[key] = int(m.group(1))
    return scores if len(scores) >= 3 else None

# ── IRR functions ──
def fleiss_kappa(mat):
    cats = [1, 2, 3, 4, 5]
    counts = np.zeros((mat.shape[0], len(cats)))
    for i in range(mat.shape[0]):
        ratings = mat[i][~np.isnan(mat[i])].astype(int)
        for r in ratings:
            if r in cats: counts[i, cats.index(r)] += 1
    n_per = counts.sum(axis=1)
    mask = n_per >= 2
    counts, n_per = counts[mask], n_per[mask]
    N = counts.shape[0]
    if N == 0: return None
    Pi = np.array([(np.sum(counts[i]**2) - n_per[i]) / (n_per[i]*(n_per[i]-1)) if n_per[i]>1 else 0 for i in range(N)])
    Pbar = np.mean(Pi)
    pj = counts.sum(axis=0) / counts.sum()
    Pe = np.sum(pj**2)
    if Pe == 1: return 1.0
    return (Pbar - Pe) / (1 - Pe)

def icc_2_1(mat):
    valid = ~np.any(np.isnan(mat), axis=1)
    m = mat[valid]
    if m.shape[0] < 5: return None, m.shape[0]
    n, k = m.shape
    gm = m.mean()
    rm, cm = m.mean(axis=1), m.mean(axis=0)
    SSR = k * np.sum((rm - gm)**2)
    SSC = n * np.sum((cm - gm)**2)
    SST = np.sum((m - gm)**2)
    SSE = SST - SSR - SSC
    MSR, MSC, MSE = SSR/(n-1), SSC/(k-1), SSE/((n-1)*(k-1))
    denom = MSR + (k-1)*MSE + k*(MSC-MSE)/n
    if denom == 0: return None, n
    return (MSR - MSE) / denom, n

def kripp_alpha(mat):
    pairs_sq = []
    for i in range(mat.shape[0]):
        vals = mat[i][~np.isnan(mat[i])]
        for a in range(len(vals)):
            for b in range(a+1, len(vals)):
                pairs_sq.append((vals[a]-vals[b])**2)
    if not pairs_sq: return None
    Do = np.mean(pairs_sq)
    all_vals = mat[~np.isnan(mat)]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_vals), size=min(3000, len(all_vals)), replace=False)
    s = all_vals[idx]
    total = [(s[a]-s[b])**2 for a in range(len(s)) for b in range(a+1, len(s))]
    De = np.mean(total)
    if De == 0: return 1.0
    return 1 - Do/De

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

# Load PICO extractions
pico_path = os.path.join(DATA_DIR, 'pico_extraction', 'all_extractions.json')
with open(pico_path) as f:
    extractions = json.load(f)

# Load paper texts
papers = {}
fulltext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'papers', 'fulltext')
if not os.path.exists(fulltext_dir):
    fulltext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'papers', 'pdfs')
for fn in sorted(os.listdir(fulltext_dir)):
    pmid = fn.replace('.txt', '').replace('.pdf', '')
    if pmid in extractions:
        with open(os.path.join(fulltext_dir, fn)) as fh:
            papers[pmid] = fh.read()

print(f'{"="*70}')
print(f'PICO CROSS-JUDGE EVALUATION')
print(f'Papers: {len(extractions)} | Extractors: {len(EXT_MODELS)} | Judges: {len(ALL_JUDGES)}')
print(f'Total: {len(extractions)} x {len(EXT_MODELS)} x {len(ALL_JUDGES)} = {len(extractions)*len(EXT_MODELS)*len(ALL_JUDGES)}')
print(f'Started: {datetime.now():%H:%M:%S}')
print(f'{"="*70}')

# Progress
progress_file = os.path.join(RESULTS_DIR, 'progress.json')
results = {}
if os.path.exists(progress_file):
    with open(progress_file) as f:
        results = json.load(f)

pmids = sorted(extractions.keys())
for i, pmid in enumerate(pmids):
    if pmid not in results:
        results[pmid] = {}

    text = papers.get(pmid, '')[:3000]

    needs_work = False
    for ext_name in EXT_MODELS:
        if not extractions[pmid].get(ext_name, {}).get('success'):
            continue
        for j_name in ALL_JUDGES:
            existing = results.get(pmid, {}).get(ext_name, {}).get(j_name, {})
            if not (isinstance(existing, dict) and existing.get('success')):
                needs_work = True
                break
        if needs_work: break

    if not needs_work:
        continue

    print(f'[{i+1}/{len(pmids)}] {pmid}', end=' ', flush=True)

    for ext_name in EXT_MODELS:
        ext_data = extractions[pmid].get(ext_name, {}).get('parsed')
        if not ext_data:
            continue
        if ext_name not in results[pmid]:
            results[pmid][ext_name] = {}

        ext_json = json.dumps(ext_data, indent=1)[:1200]

        prompt = (
            "Rate this PICO extraction from a clinical research paper. Score 1-5 on each dimension. Reply JSON only.\n\n"
            "PAPER:\n" + text + "\n\n"
            "EXTRACTION (by " + ext_name + "):\n" + ext_json + "\n\n"
            "Rate 1-5:\n"
            "- P_accuracy: Population correctly identified?\n"
            "- I_accuracy: Intervention/exposure correctly identified?\n"
            "- C_accuracy: Comparator correctly identified?\n"
            "- O_accuracy: Outcomes correctly identified?\n"
            "- overall: Overall extraction quality?\n\n"
            '{"P_accuracy": 0, "I_accuracy": 0, "C_accuracy": 0, "O_accuracy": 0, "overall": 0}'
        )

        for j_name in ALL_JUDGES:
            if j_name in results[pmid][ext_name]:
                existing = results[pmid][ext_name][j_name]
                if isinstance(existing, dict) and existing.get('success'):
                    continue

            raw = call(j_name, prompt)
            scores = parse_json(raw)
            results[pmid][ext_name][j_name] = {'scores': scores, 'success': scores is not None}
            print('.' if scores else 'x', end='', flush=True)
            time.sleep(0.3)

    print()
    if (i + 1) % 3 == 0:
        with open(progress_file, 'w') as f:
            json.dump(results, f, indent=1, default=str)

with open(progress_file, 'w') as f:
    json.dump(results, f, indent=1, default=str)

# ══════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print('IRR ANALYSIS')
print(f'{"="*70}')

items = []
for pmid in pmids:
    for ext in EXT_MODELS:
        if extractions[pmid].get(ext, {}).get('success'):
            items.append((pmid, ext))

matrix = np.full((len(items), len(ALL_JUDGES)), np.nan)
for i, (pmid, ext) in enumerate(items):
    for j, judge in enumerate(ALL_JUDGES):
        d = results.get(pmid, {}).get(ext, {}).get(judge, {})
        if isinstance(d, dict) and d.get('scores'):
            s = d['scores'].get('overall')
            if s: matrix[i, j] = s

big_vals = matrix[:, :len(BIG)][~np.isnan(matrix[:, :len(BIG)])].tolist()
small_vals = matrix[:, len(BIG):][~np.isnan(matrix[:, len(BIG):])].tolist()

print(f'\n1. BIG vs SMALL')
if big_vals: print(f'   Big (27-685B):  mean={np.mean(big_vals):.2f} std={np.std(big_vals):.2f} n={len(big_vals)}')
if small_vals: print(f'   Small (3-8B):   mean={np.mean(small_vals):.2f} std={np.std(small_vals):.2f} n={len(small_vals)}')
if big_vals and small_vals: print(f'   Gap: {np.mean(big_vals)-np.mean(small_vals):+.2f}')

print(f'\n2. PER JUDGE')
for j_idx, j_name in enumerate(ALL_JUDGES):
    vals = matrix[:, j_idx][~np.isnan(matrix[:, j_idx])]
    tag = 'BIG' if j_name in BIG else 'SML'
    if len(vals) > 0:
        print(f'   [{tag}] {j_name:20s}: mean={np.mean(vals):.2f} std={np.std(vals):.2f} n={len(vals)}')

print(f'\n3. IRR METRICS')
fk_all = fleiss_kappa(matrix)
fk_big = fleiss_kappa(matrix[:, :len(BIG)])
fk_small = fleiss_kappa(matrix[:, len(BIG):])
print(f"   Fleiss' kappa:  all={fk_all:.3f}  big={fk_big:.3f}  small={fk_small:.3f}")

icc_all, n1 = icc_2_1(matrix)
icc_big, n2 = icc_2_1(matrix[:, :len(BIG)])
icc_small, n3 = icc_2_1(matrix[:, len(BIG):])
if icc_all is not None:
    print(f'   ICC(2,1):       all={icc_all:.3f}  big={icc_big:.3f}  small={icc_small:.3f}')

ka_all = kripp_alpha(matrix)
ka_big = kripp_alpha(matrix[:, :len(BIG)])
ka_small = kripp_alpha(matrix[:, len(BIG):])
print(f"   Krippendorff:   all={ka_all:.3f}  big={ka_big:.3f}  small={ka_small:.3f}")

# Per dimension
print(f'\n4. ICC BY DIMENSION')
for dim in ['P_accuracy', 'I_accuracy', 'C_accuracy', 'O_accuracy', 'overall']:
    dm = np.full((len(items), len(ALL_JUDGES)), np.nan)
    for i, (pmid, ext) in enumerate(items):
        for j, judge in enumerate(ALL_JUDGES):
            d = results.get(pmid, {}).get(ext, {}).get(judge, {})
            if isinstance(d, dict) and d.get('scores'):
                s = d['scores'].get(dim)
                if s: dm[i, j] = s
    icc, n = icc_2_1(dm)
    icc_b, _ = icc_2_1(dm[:, :len(BIG)])
    if icc is not None:
        print(f'   {dim:15s}: all={icc:.3f}  big={icc_b:.3f if icc_b else "N/A"}')

# Save analysis
analysis = {
    'n_papers': len(pmids), 'n_items': len(items),
    'n_judgments': int(np.sum(~np.isnan(matrix))),
    'big_mean': float(np.mean(big_vals)) if big_vals else None,
    'small_mean': float(np.mean(small_vals)) if small_vals else None,
    'gap': float(np.mean(big_vals) - np.mean(small_vals)) if big_vals and small_vals else None,
    'fleiss_all': fk_all, 'fleiss_big': fk_big, 'fleiss_small': fk_small,
    'icc_all': icc_all, 'icc_big': icc_big, 'icc_small': icc_small,
    'kripp_all': ka_all, 'kripp_big': ka_big, 'kripp_small': ka_small,
    'per_judge': {},
    'timestamp': datetime.now().isoformat()
}
for j_idx, j_name in enumerate(ALL_JUDGES):
    vals = matrix[:, j_idx][~np.isnan(matrix[:, j_idx])]
    if len(vals) > 0:
        analysis['per_judge'][j_name] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': int(len(vals))}

with open(os.path.join(RESULTS_DIR, 'analysis.json'), 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'\nSaved to {RESULTS_DIR}/')
print(f'Done: {datetime.now():%H:%M:%S}')
