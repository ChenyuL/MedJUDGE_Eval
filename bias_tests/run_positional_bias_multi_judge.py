#!/usr/bin/env python3
"""
Positional Bias Test — Multi-Judge, All-4-Extractor Edition
============================================================
Two complementary sub-tests:

  Test A — Pairwise (2-way):
    For each paper: judge sees (DeepSeek-V3 vs Kimi-K2) in order AB, then BA.
    consistent=True → picks same content regardless of position.

  Test B — 4-way Ranking:
    For each paper: judge sees all 4 extractions in order [0,1,2,3],
    then in a shuffled order. Judge picks the BEST each time.
    consistent=True → judge picks the same underlying content both times.

Judges: DeepSeek-V3 (OpenRouter), Claude-Sonnet-4.6 (Anthropic), GPT-4.5 (OpenAI), Gemini-2.5-Pro
Judges: all 8 primary OSS judges + 3 proprietary (Claude-Sonnet-4.6, GPT-4.5, Gemini-2.5-Pro)
"""

import os, json, time, re, random, requests
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[3] / '.env')
except Exception:
    pass

import anthropic
from openai import OpenAI
from huggingface_hub import InferenceClient

ANTHROPIC_KEY  = os.getenv('ANTHROPIC_API_KEY', '')
OPENAI_KEY     = os.getenv('OPENAI_API_KEY', '')
GEMINI_KEY     = os.getenv('GEMINI_API_KEY', '')
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', '')
HF_TOKEN       = os.getenv('HF_TOKEN', '')

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None
openai_client = OpenAI(api_key=OPENAI_KEY)                 if OPENAI_KEY    else None
hf_client     = InferenceClient(token=HF_TOKEN)            if HF_TOKEN      else None

# ── Judge registry ────────────────────────────────────────────────────────────
# backend options: hf | together | sambanova | openrouter | anthropic | openai | gemini
JUDGES = {
    # ── Primary OSS judges (same as cross-judge study) ──
    'DeepSeek-V3-685B':  ('hf',          'deepseek-ai/DeepSeek-V3-0324'),
    'Kimi-K2-1T':        ('hf',          'moonshotai/Kimi-K2-Instruct'),
    'Llama-3.3-70B':     ('hf',          'meta-llama/Llama-3.3-70B-Instruct'),
    'Gemma-3-27B':       ('hf',          'google/gemma-3-27b-it'),
    'Qwen3-32B':         ('hf',          'Qwen/Qwen3-32B'),
    'Llama-3.1-8B':      ('sambanova',   'Meta-Llama-3.1-8B-Instruct'),
    # 'Llama-3.2-3B': removed — Together returns HTTP 400 for all Llama-3.2-3B model IDs
    'Qwen2.5-7B':        ('together',    'Qwen/Qwen2.5-7B-Instruct-Turbo'),
    # ── Proprietary judges ──
    'Claude-Sonnet-4.6': ('anthropic',   'claude-sonnet-4-6'),
    'GPT-4.5':           ('openai',      'gpt-4.5-preview'),
    # 'Gemini-2.5-Pro':  ('gemini',      'gemini-2.5-pro-preview-03-25'),  # re-enable when quota available
}

EXTRACTORS = ['DeepSeek-V3', 'Kimi-K2', 'Llama-3.3-70B', 'Gemma-3-27B']

RWE_ROOT  = Path(__file__).parents[3]
DATA_DIR  = RWE_ROOT / 'AMIA2026_Package' / 'data'
PICO_FILE = DATA_DIR / 'pico_extraction' / 'all_extractions.json'
OUT_FILE  = DATA_DIR / 'bias_results' / 'test4_positional_multi_judge.json'

SYSTEM_2WAY = (
    "You are a clinical research expert. "
    "Decide which PICO extraction is more accurate and complete. "
    "Reply with ONLY one letter: A or B."
)
SYSTEM_4WAY = (
    "You are a clinical research expert. "
    "Pick the BEST PICO extraction from the list. "
    "Reply with ONLY one letter: A, B, C, or D."
)


def fmt_pico(d: dict) -> str:
    return (f"  Population:   {d.get('P', {}).get('description', 'N/A')[:120]}\n"
            f"  Intervention: {d.get('I', {}).get('description', 'N/A')[:120]}\n"
            f"  Comparator:   {d.get('C', {}).get('description', 'N/A')[:100]}\n"
            f"  Outcome:      {d.get('O', {}).get('description', 'N/A')[:120]}")


def make_prompt_2way(ea: dict, eb: dict) -> str:
    return (f"=== Extraction A ===\n{fmt_pico(ea)}\n\n"
            f"=== Extraction B ===\n{fmt_pico(eb)}\n\n"
            "Which is more accurate and complete? Reply A or B only.")


def make_prompt_4way(ordered: list[tuple[str, dict]]) -> str:
    labels = ['A', 'B', 'C', 'D']
    parts = [f"=== Extraction {labels[i]} ===\n{fmt_pico(d)}"
             for i, (_, d) in enumerate(ordered)]
    return ('\n\n'.join(parts) + '\n\nWhich extraction is best? Reply A, B, C, or D only.')


# ── API callers ───────────────────────────────────────────────────────────────
def call_openrouter(model_id: str, system: str, prompt: str) -> str | None:
    if not OPENROUTER_KEY:
        return None
    for attempt in range(3):
        try:
            r = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={'Authorization': f'Bearer {OPENROUTER_KEY}',
                         'Content-Type': 'application/json'},
                json={'model': model_id,
                      'messages': [{'role': 'system', 'content': system},
                                   {'role': 'user',   'content': prompt}],
                      'max_tokens': 5, 'temperature': 0.0},
                timeout=30)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content'].strip()
            print(f'      OR HTTP {r.status_code} (attempt {attempt+1})')
            time.sleep(3)
        except Exception as e:
            print(f'      OR error (attempt {attempt+1}): {str(e)[:80]}')
            time.sleep(3)
    return None


def call_anthropic(model_id: str, system: str, prompt: str) -> str | None:
    if not claude_client:
        return None
    try:
        msg = claude_client.messages.create(
            model=model_id, max_tokens=5, system=system,
            messages=[{'role': 'user', 'content': prompt}])
        return msg.content[0].text.strip()
    except Exception as e:
        print(f'      Anthropic error: {str(e)[:80]}')
        return None


def call_openai(model_id: str, system: str, prompt: str) -> str | None:
    if not openai_client:
        return None
    try:
        r = openai_client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user',   'content': prompt}],
            max_completion_tokens=5)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f'      OpenAI error: {str(e)[:80]}')
        return None


def call_gemini(model_id: str, system: str, prompt: str) -> str | None:
    if not GEMINI_KEY:
        return None
    url = (f'https://generativelanguage.googleapis.com/v1beta/models/'
           f'{model_id}:generateContent?key={GEMINI_KEY}')
    body = {
        'contents': [{'parts': [{'text': system + '\n\n' + prompt}]}],
        'generationConfig': {'maxOutputTokens': 5, 'temperature': 0.0},
    }
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, timeout=30)
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f'      Gemini 429 — waiting {wait}s', end='', flush=True)
                time.sleep(wait)
                continue
            print(f'      Gemini HTTP {r.status_code} (attempt {attempt+1})')
            time.sleep(5)
        except Exception as e:
            print(f'      Gemini error: {str(e)[:80]}')
            time.sleep(5)
    return None


def call_hf(model_id: str, system: str, prompt: str) -> str | None:
    if not hf_client:
        return None
    # Use 500 tokens for thinking models (e.g. Qwen3-32B generates <think>...</think> first)
    is_thinking = 'Qwen3' in model_id
    max_tok = 500 if is_thinking else 5
    for attempt in range(3):
        try:
            r = hf_client.chat_completion(
                messages=[{'role': 'user', 'content': system + '\n\n' + prompt}],
                model=model_id, max_tokens=max_tok, temperature=0.0)
            txt = r.choices[0].message.content.strip()
            if '<think>' in txt:
                txt = txt.split('</think>')[-1].strip()
            return txt
        except Exception as e:
            print(f'      HF error (attempt {attempt+1}): {str(e)[:80]}')
            time.sleep(5 * (attempt + 1))
    return None


def call_router(backend: str, model_id: str, system: str, prompt: str) -> str | None:
    """Together / SambaNova via HF router."""
    if not HF_TOKEN:
        return None
    url = f'https://router.huggingface.co/{backend}/v1/chat/completions'
    for attempt in range(3):
        try:
            r = requests.post(
                url,
                headers={'Authorization': f'Bearer {HF_TOKEN}',
                         'Content-Type': 'application/json'},
                json={'model': model_id,
                      'messages': [{'role': 'user', 'content': system + '\n\n' + prompt}],
                      'max_tokens': 5, 'temperature': 0.0},
                timeout=30)
            if r.status_code == 200:
                txt = r.json()['choices'][0]['message']['content']
                if '<think>' in txt:
                    txt = txt.split('</think>')[-1].strip()
                return txt
            print(f'      {backend} HTTP {r.status_code} (attempt {attempt+1})')
            time.sleep(5)
        except Exception as e:
            print(f'      {backend} error: {str(e)[:80]}')
            time.sleep(5)
    return None


def call_judge(name: str, system: str, prompt: str) -> str | None:
    api, model_id = JUDGES[name]
    if api == 'hf':
        raw = call_hf(model_id, system, prompt)
    elif api in ('together', 'sambanova'):
        raw = call_router(api, model_id, system, prompt)
    elif api == 'openrouter':
        raw = call_openrouter(model_id, system, prompt)
    elif api == 'anthropic':
        raw = call_anthropic(model_id, system, prompt)
    elif api == 'openai':
        raw = call_openai(model_id, system, prompt)
    elif api == 'gemini':
        raw = call_gemini(model_id, system, prompt)
    else:
        raw = None
    if raw:
        m = re.search(r'\b([ABCD])\b', raw.upper())
        return m.group(1) if m else None
    return None


# ── Load data ─────────────────────────────────────────────────────────────────
print('=' * 70)
print('POSITIONAL BIAS — MULTI-JUDGE × ALL-4-EXTRACTORS')
print('Test A: 2-way pairwise (DeepSeek-V3 vs Kimi-K2)')
print('Test B: 4-way ranking  (all 4 extractors, shuffled)')
print('=' * 70)

with open(PICO_FILE) as f:
    extractions = json.load(f)

results = []
results_by_pmid = {}
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        results = json.load(f)
    results_by_pmid = {r['pmid']: r for r in results}
    print(f'  Resuming — {len(results_by_pmid)} papers in file.')

random.seed(42)
pmids = sorted(extractions.keys())
all_pmids = [p for p in pmids if all(extractions[p].get(e, {}).get('parsed') for e in EXTRACTORS)]

print(f'  Judges:     {list(JUDGES.keys())}')
print(f'  Papers:     {len(all_pmids)} total\n')

for i, pmid in enumerate(all_pmids):
    parsed = {e: extractions[pmid][e]['parsed'] for e in EXTRACTORS}

    # Restore or create row; preserve existing orders for reproducibility
    if pmid in results_by_pmid:
        row = results_by_pmid[pmid]
        # Ensure test_b sub-keys exist (older format compat)
        if 'test_a' not in row:
            row['test_a'] = {}
        if 'test_b' not in row:
            row['test_b'] = {'judges': {}}
        if 'judges' not in row['test_b']:
            row['test_b']['judges'] = {}
        # Restore orders if present; else regenerate deterministically
        order1 = row['test_b'].get('order1', list(EXTRACTORS))
        order2 = row['test_b'].get('order2')
    else:
        order1 = list(EXTRACTORS)
        order2 = list(EXTRACTORS)
        random.shuffle(order2)
        while order2 == order1:
            random.shuffle(order2)
        row = {
            'pmid': pmid,
            'test_a': {},
            'test_b': {'order1': order1, 'order2': order2, 'judges': {}},
        }
        results_by_pmid[pmid] = row

    # Identify which judges still need to run (or retry if all calls previously failed)
    def needs_retry(j):
        if j not in row['test_a']:
            return True
        ta = row['test_a'][j]
        if ta.get('order_ab_choice') is None and ta.get('order_ba_choice') is None:
            return True  # both test-A calls failed last time
        if j not in row['test_b']['judges']:
            return True
        tb = row['test_b']['judges'][j]
        if tb.get('order1_pick_label') is None and tb.get('order2_pick_label') is None:
            return True  # both test-B calls failed last time
        return False

    pending = [j for j in JUDGES if needs_retry(j)]
    if not pending:
        print(f'[{i+1:02d}/{len(all_pmids)}] {pmid}  (all judges done, skip)')
        continue

    print(f'[{i+1:02d}/{len(all_pmids)}] {pmid}', end='  ', flush=True)

    for j_name in pending:
        # ── Test A: 2-way pairwise ──
        ta_prev = row['test_a'].get(j_name, {})
        if j_name not in row['test_a'] or (
                ta_prev.get('order_ab_choice') is None and ta_prev.get('order_ba_choice') is None):
            ea, eb = parsed[EXTRACTORS[0]], parsed[EXTRACTORS[1]]
            c_ab = call_judge(j_name, SYSTEM_2WAY, make_prompt_2way(ea, eb))
            time.sleep(0.3)
            c_ba = call_judge(j_name, SYSTEM_2WAY, make_prompt_2way(eb, ea))
            time.sleep(0.3)
            c_ba_norm = {'A': 'B', 'B': 'A'}.get(c_ba) if c_ba else None
            consistent_a = (c_ab == c_ba_norm) if (c_ab and c_ba_norm) else None
            row['test_a'][j_name] = {
                'order_ab_choice': c_ab, 'order_ba_choice': c_ba,
                'preferred': c_ab, 'consistent': consistent_a,
            }
        else:
            consistent_a = row['test_a'][j_name].get('consistent')

        # ── Test B: 4-way ranking ──
        tb_prev = row['test_b']['judges'].get(j_name, {})
        if j_name not in row['test_b']['judges'] or (
                tb_prev.get('order1_pick_label') is None and tb_prev.get('order2_pick_label') is None):
            pairs1 = [(e, parsed[e]) for e in order1]
            pairs2 = [(e, parsed[e]) for e in order2]
            label_map1 = {lab: ext for lab, (ext, _) in zip('ABCD', pairs1)}
            label_map2 = {lab: ext for lab, (ext, _) in zip('ABCD', pairs2)}
            pick1_raw = call_judge(j_name, SYSTEM_4WAY, make_prompt_4way(pairs1))
            time.sleep(0.3)
            pick2_raw = call_judge(j_name, SYSTEM_4WAY, make_prompt_4way(pairs2))
            time.sleep(0.3)
            pick1_ext = label_map1.get(pick1_raw) if pick1_raw else None
            pick2_ext = label_map2.get(pick2_raw) if pick2_raw else None
            consistent_b = (pick1_ext == pick2_ext) if (pick1_ext and pick2_ext) else None
            row['test_b']['judges'][j_name] = {
                'order1_pick_label': pick1_raw, 'order1_pick_extractor': pick1_ext,
                'order2_pick_label': pick2_raw, 'order2_pick_extractor': pick2_ext,
                'consistent': consistent_b,
            }
        else:
            consistent_b = row['test_b']['judges'][j_name].get('consistent')

        icon  = '✓' if consistent_a else ('✗' if consistent_a is False else '?')
        icon4 = '✓' if consistent_b else ('✗' if consistent_b is False else '?')
        print(f'{j_name}:2w{icon}/4w{icon4}', end='  ', flush=True)

    print()
    # Rebuild results list from dict after each paper
    results = list(results_by_pmid.values())
    if (i + 1) % 5 == 0 or (i + 1) == len(all_pmids):
        with open(OUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [saved {len(results)} results]')

results = list(results_by_pmid.values())
with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f'\n{"=" * 70}')
print('ANALYSIS SUMMARY')
print(f'{"=" * 70}')
all_judge_names = sorted({j for r in results for j in r.get('test_a', {})})
print(f'\n--- Test A: 2-way pairwise ({EXTRACTORS[0]} vs {EXTRACTORS[1]}) ---')
print(f'{"Judge":<22}  {"Consist":>8}  {"Biased":>8}  {"No-dec":>7}  {"Rate%":>7}  {"PreferA%":>9}')
print('-' * 68)
for j in all_judge_names:
    c = sum(1 for r in results if r['test_a'].get(j, {}).get('consistent') is True)
    b = sum(1 for r in results if r['test_a'].get(j, {}).get('consistent') is False)
    n = sum(1 for r in results if r['test_a'].get(j, {}).get('consistent') is None)
    t = c + b
    rate = c/t if t else float('nan')
    pa = sum(1 for r in results if r['test_a'].get(j, {}).get('preferred') == 'A')
    pct = pa/len(results) if results else float('nan')
    print(f'{j:<22}  {c:>8}  {b:>8}  {n:>7}  {rate:>7.1%}  {pct:>9.1%}')

print(f'\n--- Test B: 4-way ranking (all 4 extractors, order-shuffled) ---')
all_judge_names_b = sorted({j for r in results for j in r.get('test_b', {}).get('judges', {})})
print(f'{"Judge":<22}  {"Consist":>8}  {"Biased":>8}  {"No-dec":>7}  {"Rate%":>7}')
print('-' * 58)
for j in all_judge_names_b:
    c = sum(1 for r in results if r['test_b']['judges'].get(j, {}).get('consistent') is True)
    b = sum(1 for r in results if r['test_b']['judges'].get(j, {}).get('consistent') is False)
    n = sum(1 for r in results if r['test_b']['judges'].get(j, {}).get('consistent') is None)
    t = c + b
    rate = c/t if t else float('nan')
    print(f'{j:<22}  {c:>8}  {b:>8}  {n:>7}  {rate:>7.1%}')

print(f'\nSaved → {OUT_FILE.relative_to(RWE_ROOT)}')
print(f'Done:    {datetime.now():%H:%M:%S}')
