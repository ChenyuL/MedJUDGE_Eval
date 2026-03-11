#!/usr/bin/env python3
"""
MedJUDGE Bias Tests - FINAL VERSION
Properly configured for your actual data structure
"""

import os
import json
import time
import random
import statistics
from datetime import datetime
from pathlib import Path
import requests
import re

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# API Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
HF_TOKEN = os.getenv('HF_TOKEN', os.getenv('HUGGINGFACE_TOKEN', ''))

# Paths - Your actual data locations
BASE_DIR = Path('/Users/chenyuli/Desktop/MedOS/BMI-Research/rwe-research')
CROSS_JUDGE_DIR = BASE_DIR / 'cross_judge_results'
BIAS_RESULTS_DIR = BASE_DIR / 'AMIA2026_Package' / 'data' / 'bias_results'

# Create results directory
BIAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Judge models for bias tests
BIAS_JUDGE_MODELS = {
    'deepseek': 'deepseek/deepseek-chat',
    'qwen': 'qwen/qwen-2.5-72b-instruct',
    'llama': 'meta-llama/llama-3.3-70b-instruct'
}

SELF_PREF_JUDGES = {
    'gpt-4.5': {'api': 'openai', 'model': 'gpt-4.5-preview'},
    'claude-haiku': {'api': 'openrouter', 'model': 'anthropic/claude-3.5-haiku'},
    'deepseek': {'api': 'openrouter', 'model': 'deepseek/deepseek-chat'}
}

# Model name mappings (from your data to display names)
JUDGE_NAME_MAP = {
    'DeepSeek-V3-685B': 'deepseek',
    'Qwen2.5-7B': 'qwen',
    'Llama-3.3-70B': 'llama',
    'Llama-3.1-8B': 'llama',
    'Llama-3.2-3B': 'llama'
}

print("="*80)
print("MedJUDGE BIAS TESTS - FINAL VERSION")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nPaths:")
print(f"  Data: {CROSS_JUDGE_DIR}")
print(f"  Output: {BIAS_RESULTS_DIR}")

# Check API keys
print("\nAPI Keys:")
has_openrouter = bool(OPENROUTER_API_KEY)
has_openai = bool(OPENAI_API_KEY)
has_hf = bool(HF_TOKEN)

print(f"  {'✅' if has_openrouter else '❌'} OpenRouter")
print(f"  {'✅' if has_openai else '❌'} OpenAI")
print(f"  {'✅' if has_hf else '❌'} HuggingFace")

if not has_openrouter:
    print("\n❌ ERROR: OpenRouter API key required!")
    print("   Add to .env: OPENROUTER_API_KEY=sk-or-...")
    exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def call_openrouter(model, messages, temperature=0.0, max_retries=3):
    """Call OpenRouter API with retries"""
    for attempt in range(max_retries):
        try:
            resp = requests.post('https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': 300
                },
                timeout=60
            )
            data = resp.json()
            if 'choices' in data:
                return data['choices'][0]['message']['content']
            if 'error' in data:
                print(f"\n    API error: {data['error'].get('message','unknown')}")
                time.sleep(5)
        except Exception as e:
            print(f"\n    Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return None

def call_openai(messages, temperature=0.0):
    """Call OpenAI API"""
    if not OPENAI_API_KEY:
        return None
    try:
        resp = requests.post('https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'gpt-4.5-preview',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': 300
            },
            timeout=60
        )
        data = resp.json()
        if 'choices' in data:
            return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"\n    OpenAI error: {e}")
    return None

def extract_rating(text):
    """Extract 1-5 rating from response"""
    if not text:
        return None

    patterns = [
        r'(?:rating|score)[:\s]*(\d)',
        r'\b([1-5])\s*/\s*5',
        r'^\s*([1-5])\s*$',
        r'\*\*([1-5])\*\*'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 5:
                return rating

    digits = re.findall(r'\b([1-5])\b', text)
    if digits:
        return int(digits[0])
    return None

# ══════════════════════════════════════════════════════════════════════════════
# LOAD YOUR ACTUAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_actual_extraction_data():
    """Load your actual cross-judge data"""
    print("\n" + "="*80)
    print("LOADING YOUR DATA")
    print("="*80)

    progress_file = CROSS_JUDGE_DIR / 'progress.json'

    if not progress_file.exists():
        print(f"❌ Data file not found: {progress_file}")
        exit(1)

    print(f"✅ Found: {progress_file}")

    with open(progress_file) as f:
        data = json.load(f)

    print(f"✅ Loaded data with {len(data)} PMIDs")

    # Your data structure:
    # data[pmid][extractor_model][judge_model]['scores']['overall']

    # Extract extraction pairs for Tests 1, 2, 4
    pairs = []
    extractors = []

    for pmid, pmid_data in list(data.items()):
        extractor_models = list(pmid_data.keys())

        # Get pairs of extractions from same paper
        if len(extractor_models) >= 2:
            ext_a = extractor_models[0]
            ext_b = extractor_models[1]

            pairs.append({
                'id': f"{pmid}_pair",
                'pmid': pmid,
                'extractor_a': ext_a,
                'extractor_b': ext_b,
                'extraction_a': f"Extraction from {ext_a} (PMID {pmid})",
                'extraction_b': f"Extraction from {ext_b} (PMID {pmid})",
                'paper_abstract': f"Paper PMID: {pmid}"
            })

        # Also collect individual extractions
        for extractor_model in extractor_models:
            extractors.append({
                'pmid': pmid,
                'extractor': extractor_model,
                'judges': pmid_data[extractor_model],
                'text': f"Extraction from {extractor_model} for PMID {pmid}"
            })

    print(f"✅ Extracted {len(pairs)} extraction pairs")
    print(f"✅ Extracted {len(extractors)} individual extractions")

    return pairs, extractors, data

# Load data
EXTRACTION_PAIRS, INDIVIDUAL_EXTRACTIONS, RAW_DATA = load_actual_extraction_data()

if len(EXTRACTION_PAIRS) < 10:
    print(f"\n⚠️  Warning: Only {len(EXTRACTION_PAIRS)} pairs available")
    print("   Some tests may use limited samples")

print(f"\n✅ Data ready!\n")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: SELF-PREFERENCE BIAS
# ══════════════════════════════════════════════════════════════════════════════

def test1_self_preference(n_tasks=50):
    """Test if judges prefer their own model family"""
    print("\n" + "="*80)
    print("TEST 1: SELF-PREFERENCE BIAS")
    print("="*80)

    if not OPENAI_API_KEY:
        print("⚠️  SKIPPING: OpenAI API key needed for GPT-4.5 judge")
        return None

    n_samples = min(n_tasks, len(EXTRACTION_PAIRS))
    print(f"Sample size: {n_samples} pairs")
    print("Judges: GPT-4.5, Claude-3.5-Haiku, DeepSeek-Chat")

    sample = random.sample(EXTRACTION_PAIRS, n_samples)
    results = []

    for i, task in enumerate(sample):
        print(f"\r  Progress: {i+1}/{len(sample)}", end='', flush=True)

        row = {'task_id': task['id'], 'pmid': task['pmid']}

        def make_prompt(extraction):
            return [
                {'role': 'system', 'content': 'Rate extraction quality 1-5. Respond: Rating: X'},
                {'role': 'user', 'content': f'Rate this extraction:\n\n{extraction}\n\nRating:'}
            ]

        # Each judge rates both extractions
        for judge_name, judge_config in SELF_PREF_JUDGES.items():
            # Rate extraction A
            if judge_config['api'] == 'openai':
                response_a = call_openai(make_prompt(task['extraction_a']))
            else:
                response_a = call_openrouter(judge_config['model'], make_prompt(task['extraction_a']))
            rating_a = extract_rating(response_a)

            # Rate extraction B
            if judge_config['api'] == 'openai':
                response_b = call_openai(make_prompt(task['extraction_b']))
            else:
                response_b = call_openrouter(judge_config['model'], make_prompt(task['extraction_b']))
            rating_b = extract_rating(response_b)

            row[f'{judge_name}_rating_a'] = rating_a
            row[f'{judge_name}_rating_b'] = rating_b

            time.sleep(0.5)

        results.append(row)

    print()

    # Analyze
    print("\n  Analysis:")
    for judge_name in SELF_PREF_JUDGES.keys():
        ratings_a = [r[f'{judge_name}_rating_a'] for r in results if r.get(f'{judge_name}_rating_a')]
        ratings_b = [r[f'{judge_name}_rating_b'] for r in results if r.get(f'{judge_name}_rating_b')]

        if ratings_a and ratings_b:
            mean_a = statistics.mean(ratings_a)
            mean_b = statistics.mean(ratings_b)
            diff = mean_b - mean_a
            print(f"    {judge_name}: A={mean_a:.2f}, B={mean_b:.2f}, Diff={diff:+.2f}")

    # Save
    output_file = BIAS_RESULTS_DIR / 'test1_self_preference.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test': 'self_preference',
                'n_tasks': len(results),
                'timestamp': datetime.now().isoformat(),
                'judges': list(SELF_PREF_JUDGES.keys())
            },
            'results': results
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file.name}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: TEMPERATURE INSTABILITY
# ══════════════════════════════════════════════════════════════════════════════

def test2_temperature(n_tasks=15, n_runs=5):
    """Test how temperature affects reproducibility"""
    print("\n" + "="*80)
    print("TEST 2: TEMPERATURE INSTABILITY")
    print("="*80)

    n_samples = min(n_tasks, len(INDIVIDUAL_EXTRACTIONS))
    print(f"Sample size: {n_samples} extractions × {n_runs} runs")
    print("Temperatures: 0.0, 0.5, 1.0")

    sample = random.sample(INDIVIDUAL_EXTRACTIONS, n_samples)
    results = []

    temps = [0.0, 0.5, 1.0]
    judge_model = BIAS_JUDGE_MODELS['deepseek']

    for i, task in enumerate(sample):
        print(f"\r  Progress: {i+1}/{len(sample)}", end='', flush=True)

        row = {'task_id': f"{task['pmid']}_{task['extractor']}", 'pmid': task['pmid']}

        prompt = [
            {'role': 'system', 'content': 'Rate extraction quality 1-5. Respond: Rating: X'},
            {'role': 'user', 'content': f'Rate this extraction:\n\n{task["text"]}\n\nRating:'}
        ]

        # Test each temperature
        for temp in temps:
            ratings = []
            for run in range(n_runs):
                response = call_openrouter(judge_model, prompt, temperature=temp)
                rating = extract_rating(response)
                if rating:
                    ratings.append(rating)
                time.sleep(0.3)

            row[f'temp_{temp}'] = ratings

        results.append(row)

    print()

    # Analyze
    print("\n  Analysis:")
    for temp in temps:
        all_scores = [r[f'temp_{temp}'] for r in results if r.get(f'temp_{temp}')]
        if all_scores:
            agreements = [1.0 if len(set(scores)) == 1 else 0.0 for scores in all_scores if len(scores) > 0]
            agreement_rate = statistics.mean(agreements) if agreements else 0

            variances = [statistics.variance(scores) if len(scores) > 1 else 0 for scores in all_scores]
            mean_var = statistics.mean(variances) if variances else 0

            print(f"    T={temp}: Agreement={agreement_rate:.1%}, Variance={mean_var:.3f}")

    # Save
    output_file = BIAS_RESULTS_DIR / 'test2_temperature.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test': 'temperature_instability',
                'n_tasks': len(results),
                'n_runs': n_runs,
                'temperatures': temps,
                'timestamp': datetime.now().isoformat(),
                'judge_model': judge_model
            },
            'results': results
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file.name}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: ERROR CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def test3_error_correlation():
    """Calculate correlation between judge errors"""
    print("\n" + "="*80)
    print("TEST 3: ERROR CORRELATION")
    print("="*80)
    print("Using your existing cross-judge data...")

    # Extract all ratings by judge from your data structure
    # RAW_DATA[pmid][extractor][judge]['scores']['overall']

    judge_ratings = {}

    for pmid, pmid_data in RAW_DATA.items():
        for extractor, extractor_data in pmid_data.items():
            for judge, judge_data in extractor_data.items():
                if judge_data.get('success') and 'scores' in judge_data:
                    overall = judge_data['scores'].get('overall')
                    if overall is not None:
                        # Map full judge name to short name
                        short_name = JUDGE_NAME_MAP.get(judge, judge)
                        if short_name not in judge_ratings:
                            judge_ratings[short_name] = []
                        judge_ratings[short_name].append(float(overall))

    # Report what we found
    print("\n  Ratings collected:")
    for judge, ratings in sorted(judge_ratings.items()):
        print(f"    {judge}: {len(ratings)} ratings")

    # Make all equal length (use minimum)
    min_len = min(len(ratings) for ratings in judge_ratings.values())
    for judge in judge_ratings:
        judge_ratings[judge] = judge_ratings[judge][:min_len]

    print(f"\n  Using {min_len} ratings per judge for correlation")

    # Calculate pairwise correlations
    try:
        from scipy.stats import pearsonr
    except ImportError:
        print("  Installing scipy...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy', '-q'])
        from scipy.stats import pearsonr

    correlations = {}
    judges = sorted(judge_ratings.keys())

    print("\n  Pairwise correlations:")
    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            if len(judge_ratings[j1]) > 1 and len(judge_ratings[j2]) > 1:
                r, p = pearsonr(judge_ratings[j1], judge_ratings[j2])
                correlations[f'{j1}_vs_{j2}'] = {
                    'r': float(r),
                    'p': float(p),
                    'n': min_len
                }
                print(f"    {j1:12s} vs {j2:12s}: r={r:.3f} (p={p:.6f})")

    # Save
    output_file = BIAS_RESULTS_DIR / 'test3_error_correlation.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test': 'error_correlation',
                'n_ratings': min_len,
                'timestamp': datetime.now().isoformat(),
                'judges': judges
            },
            'correlations': correlations,
            'sample_ratings': {j: judge_ratings[j][:10] for j in judges}
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file.name}")
    return correlations

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: POSITIONAL BIAS
# ══════════════════════════════════════════════════════════════════════════════

def test4_positional(n_tasks=50):
    """Test if presentation order affects judgment"""
    print("\n" + "="*80)
    print("TEST 4: POSITIONAL BIAS")
    print("="*80)

    n_samples = min(n_tasks, len(EXTRACTION_PAIRS))
    print(f"Sample size: {n_samples} pairs × 2 orders")

    sample = random.sample(EXTRACTION_PAIRS, n_samples)
    results = []

    judge_model = BIAS_JUDGE_MODELS['deepseek']

    for i, task in enumerate(sample):
        print(f"\r  Progress: {i+1}/{len(sample)}", end='', flush=True)

        # Order 1: A then B
        prompt_ab = [
            {'role': 'system', 'content': 'Compare extractions. Choose A or B. Respond: Choice: A or Choice: B'},
            {'role': 'user', 'content': f'Extraction A:\n{task["extraction_a"]}\n\nExtraction B:\n{task["extraction_b"]}\n\nWhich is better? Choice:'}
        ]

        response_ab = call_openrouter(judge_model, prompt_ab, temperature=0.0)
        choice_ab = 'A' if response_ab and 'A' in response_ab.upper().split('CHOICE')[-1][:5] else 'B'

        time.sleep(0.5)

        # Order 2: B then A (reversed)
        prompt_ba = [
            {'role': 'system', 'content': 'Compare extractions. Choose A or B. Respond: Choice: A or Choice: B'},
            {'role': 'user', 'content': f'Extraction A:\n{task["extraction_b"]}\n\nExtraction B:\n{task["extraction_a"]}\n\nWhich is better? Choice:'}
        ]

        response_ba = call_openrouter(judge_model, prompt_ba, temperature=0.0)
        choice_ba = 'A' if response_ba and 'A' in response_ba.upper().split('CHOICE')[-1][:5] else 'B'

        # Check consistency
        consistent = (choice_ab == 'A' and choice_ba == 'B') or (choice_ab == 'B' and choice_ba == 'A')

        results.append({
            'task_id': task['id'],
            'order_ab_choice': choice_ab,
            'order_ba_choice': choice_ba,
            'consistent': consistent
        })

        time.sleep(0.5)

    print()

    # Analyze
    print("\n  Analysis:")
    consistent_count = sum(1 for r in results if r['consistent'])
    flip_count = len(results) - consistent_count

    pos1_pref = sum(1 for r in results if r['order_ab_choice'] == 'A')
    pos2_pref = sum(1 for r in results if r['order_ab_choice'] == 'B')

    flip_rate = flip_count / len(results) if results else 0

    print(f"    Consistent: {consistent_count}/{len(results)} ({consistent_count/len(results):.1%})")
    print(f"    Flip rate: {flip_rate:.1%}")
    print(f"    Position 1 preference: {pos1_pref/len(results):.1%}")
    print(f"    Position 2 preference: {pos2_pref/len(results):.1%}")

    # Save
    output_file = BIAS_RESULTS_DIR / 'test4_positional.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'test': 'positional_bias',
                'n_tasks': len(results),
                'timestamp': datetime.now().isoformat(),
                'judge_model': judge_model
            },
            'results': results,
            'summary': {
                'consistent_count': consistent_count,
                'flip_count': flip_count,
                'flip_rate': flip_rate,
                'position1_rate': pos1_pref / len(results) if results else 0,
                'position2_rate': pos2_pref / len(results) if results else 0
            }
        }, f, indent=2)

    print(f"\n  ✅ Saved: {output_file.name}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("RUNNING ALL BIAS TESTS")
    print("="*80)

    random.seed(42)

    try:
        # Test 3 first (no API calls, uses existing data)
        test3_error_correlation()

        # Test 2 (moderate API calls)
        test2_temperature(n_tasks=15, n_runs=5)

        # Test 4 (moderate API calls)
        test4_positional(n_tasks=50)

        # Test 1 last (needs both APIs)
        if OPENAI_API_KEY:
            test1_self_preference(n_tasks=50)
        else:
            print("\n⚠️  Skipping Test 1 (needs OpenAI API key)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("BIAS TESTS COMPLETE")
    print("="*80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults: {BIAS_RESULTS_DIR}/")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(BIAS_RESULTS_DIR.glob('*.json')):
        size_kb = f.stat().st_size / 1024
        print(f"  ✅ {f.name} ({size_kb:.1f} KB)")

    print("\n📊 Next steps:")
    print("  1. Review results in bias_results/")
    print("  2. Generate figures: python ../visualization/create_figures_v2.py")
    print("  3. Submit to AMIA!\n")
