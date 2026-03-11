#!/usr/bin/env python3
"""
MedJUDGE — Private Model Cross-Judge Evaluation
================================================
Runs Claude-Sonnet-4.6, GPT-4.5, and Gemini-2.5-Pro as judges on all 30 papers × 4 extractors,
using the same PICO_JUDGE_PROMPT as run_cross_judge_v2.py.
Results are appended to cross_judge_results/progress.json.

Usage:
    cd /Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research
    python AMIA2026_Package/scripts/cross_judge/run_private_cross_judge.py

Requires in .env:
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=AIza...
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os, json, time, re, requests
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR         = Path('/Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research')
PICO_FILE        = BASE_DIR / 'pico_extraction' / 'progress.json'
CROSS_JUDGE_FILE = BASE_DIR / 'cross_judge_results' / 'progress.json'
PAPERS_DIR       = BASE_DIR / 'papers' / 'fulltext'

OPENAI_API_KEY    = os.getenv('OPENAI_API_KEY', '')
GEMINI_API_KEY    = os.getenv('GEMINI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# ── Private judge definitions (label → (backend, model_id)) ──────────────────
# Labels match those already used in bias tests for consistency.
PRIVATE_JUDGES = {}
if ANTHROPIC_API_KEY:
    PRIVATE_JUDGES['Claude-Sonnet-4.6'] = ('anthropic', 'claude-sonnet-4-6')
if OPENAI_API_KEY:
    PRIVATE_JUDGES['GPT-4.5']           = ('openai',    'gpt-4.5-preview')
# Gemini: skipped — API quota exhausted on free tier
# if GEMINI_API_KEY:
#     PRIVATE_JUDGES['Gemini-3.1-Pro'] = ('gemini', 'gemini-3.1-pro-preview')

EXT_MODELS = ['DeepSeek-V3', 'Kimi-K2', 'Llama-3.3-70B', 'Gemma-3-27B']

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

PICO_KEYS = ['P_accuracy', 'I_accuracy', 'C_accuracy', 'O_accuracy',
             'confounders_accuracy', 'completeness', 'overall']

# ── API Callers ───────────────────────────────────────────────────────────────

def call_anthropic(model_id, prompt):
    for attempt in range(3):
        try:
            resp = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={'x-api-key': ANTHROPIC_API_KEY,
                         'anthropic-version': '2023-06-01',
                         'Content-Type': 'application/json'},
                json={'model': model_id,
                      'messages': [{'role': 'user', 'content': prompt}],
                      'max_tokens': 400},
                timeout=90,
            )
            data = resp.json()
            if 'content' in data:
                return data['content'][0]['text']
            print(f"    [Anthropic] error: {data.get('error', {}).get('message', data)}")
        except Exception as exc:
            print(f"    [Anthropic] attempt {attempt+1} failed: {exc}")
        time.sleep(5 * (attempt + 1))
    return None


def call_openai(model_id, prompt):
    for attempt in range(3):
        try:
            resp = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}',
                         'Content-Type': 'application/json'},
                json={'model': model_id,
                      'messages': [{'role': 'user', 'content': prompt}],
                      'max_tokens': 400, 'temperature': 0.0},
                timeout=90,
            )
            data = resp.json()
            if 'choices' in data:
                return data['choices'][0]['message']['content']
            print(f"    [OpenAI] error: {data.get('error', {}).get('message', data)}")
        except Exception as exc:
            print(f"    [OpenAI] attempt {attempt+1} failed: {exc}")
        time.sleep(5 * (attempt + 1))
    return None


def call_gemini(model_id, prompt):
    url = (f'https://generativelanguage.googleapis.com/v1beta/models/{model_id}'
           f':generateContent?key={GEMINI_API_KEY}')
    payload = {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.0, 'maxOutputTokens': 400},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            data = resp.json()
            if 'candidates' in data:
                return data['candidates'][0]['content']['parts'][0]['text']
            print(f"    [Gemini] error: {data.get('error', {}).get('message', data)}")
        except Exception as exc:
            print(f"    [Gemini] attempt {attempt+1} failed: {exc}")
        time.sleep(5 * (attempt + 1))
    return None


def call(judge_name, prompt):
    backend, model_id = PRIVATE_JUDGES[judge_name]
    if backend == 'anthropic':
        return call_anthropic(model_id, prompt)
    elif backend == 'openai':
        return call_openai(model_id, prompt)
    elif backend == 'gemini':
        return call_gemini(model_id, prompt)
    return None


# ── JSON Parser ───────────────────────────────────────────────────────────────

def parse_json(text):
    if not text or text.startswith('ERROR'):
        return None
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1].split('```')[0]
            break
    start = text.find('{')
    end   = text.rfind('}')
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end+1])
            if sum(1 for k in PICO_KEYS if k in parsed) >= 3:
                return parsed
        except Exception:
            pass
    # Regex fallback
    scores = {}
    for key in PICO_KEYS:
        m = re.search(rf'{key}["\s:]+([1-3])', text)
        if m:
            scores[key] = int(m.group(1))
    return scores if len(scores) >= 3 else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('MedJUDGE — PRIVATE MODEL CROSS-JUDGE EVALUATION')
    print(f'Started: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('=' * 70)

    if not PRIVATE_JUDGES:
        print('ERROR: No API keys found. Set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.')
        return

    print(f'Active judges ({len(PRIVATE_JUDGES)}): {", ".join(PRIVATE_JUDGES)}')

    # Load data
    with open(PICO_FILE) as f:
        extractions = json.load(f)

    papers = {}
    for fn in sorted(PAPERS_DIR.iterdir()):
        if fn.suffix == '.txt':
            papers[fn.stem] = fn.read_text()

    # Load existing progress
    with open(CROSS_JUDGE_FILE) as f:
        results = json.load(f)

    pmids = sorted(results.keys())  # same 30 papers
    total = len(pmids) * len(EXT_MODELS) * len(PRIVATE_JUDGES)
    print(f'\n{len(pmids)} papers × {len(EXT_MODELS)} extractors × {len(PRIVATE_JUDGES)} judges = {total} calls')

    done = skipped = failed = 0

    for i, pmid in enumerate(pmids):
        paper_text = papers.get(pmid, '')[:3000]
        print(f'\n[{i+1}/{len(pmids)}] {pmid}')

        if pmid not in results:
            results[pmid] = {}

        for ext_name in EXT_MODELS:
            ext_data = extractions.get(pmid, {}).get(ext_name, {}).get('parsed')
            if not ext_data:
                continue
            if ext_name not in results[pmid]:
                results[pmid][ext_name] = {}

            ext_json = json.dumps(ext_data, indent=1)[:1200]
            prompt = PICO_JUDGE_PROMPT.format(
                paper_text=paper_text,
                extractor=ext_name,
                extraction_json=ext_json,
            )

            for judge_name in PRIVATE_JUDGES:
                existing = results[pmid][ext_name].get(judge_name)
                if isinstance(existing, dict) and existing.get('success'):
                    skipped += 1
                    continue

                raw    = call(judge_name, prompt)
                scores = parse_json(raw)
                results[pmid][ext_name][judge_name] = {
                    'scores': scores,
                    'success': scores is not None,
                }
                status = 'OK' if scores else 'FAIL'
                tag    = 'P'  # Private
                print(f'  [{tag}] {ext_name[:10]:10s} <- {judge_name:15s} {status}')
                if scores:
                    done += 1
                else:
                    failed += 1
                    print(f'    RAW: {(raw or "None")[:200]}')
                time.sleep(0.5)

        # Save after each paper
        with open(CROSS_JUDGE_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\n{"="*70}')
    print(f'Done: {done} OK | {skipped} skipped | {failed} failed')
    print(f'Saved to: {CROSS_JUDGE_FILE}')
    print(f'Finished: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print('\nNext: run Figure I notebook cell to regenerate heatmap with Private group.')


if __name__ == '__main__':
    main()
