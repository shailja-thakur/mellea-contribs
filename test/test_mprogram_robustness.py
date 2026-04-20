"""
Robustness test for a Mellea m-program using problem variations.

Your m-program may pass a unit test and still be fragile. For example:

  - It answers correctly when numbers are given as digits ("22 people")
    but fails when phrased as words ("twenty-two people")         → format sensitivity
  - It handles a direct question but breaks when the same problem
    is framed as a story or hypothetical                          → phrasing sensitivity
  - It gets the right answer on the example but fails when
    constraints are reordered or combined differently             → structural sensitivity

A human writing unit tests would vary the inputs but always preserve the prompt
structure. Variation testing finds something harder to catch: the model passing
on the full structured prompt but breaking when the same question arrives with
different surface form — a dependency a developer would never think to test for.

    python test_mprogram_robustness.py --input-file test/data/sample_5problems.json
    python test_mprogram_robustness.py --input-file test/data/sample_5problems.json --model mistral:7b --num-variations 3
    python test_mprogram_robustness.py --input-file test/data/sample_5problems.json --gen-model qwen3:8b --quick
"""
import sys, os, io, json, contextlib, argparse, logging
from typing import Any
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.components.unit_test_eval import TestBasedEval
from mellea_contribs.tools.variation_engine import test_with_variations, analyze_robustness

G, R, Y, D, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"


def suppress_noise():
    for name in ['BenchDrift', 'benchdrift', 'mellea_contribs', 'mellea',
                 'httpx', 'httpcore', 'urllib3', 'requests', 'fancy_logger']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        from mellea.helpers.fancy_logger import FancyLogger
        fl = FancyLogger.get_logger()
        fl.setLevel(logging.CRITICAL)
        fl.handlers = []
        fl.propagate = False
    except Exception:
        pass
    os.environ['TQDM_DISABLE'] = '1'
    os.environ['MELLEA_LOG_LEVEL'] = 'CRITICAL'


def load_samples(path: str) -> list[dict]:
    """Load unit test JSON → list of {input_id, name, input, target}."""
    tests = TestBasedEval.from_json_file(path)
    samples = []
    for test in tests:
        for i, input_text in enumerate(test.inputs):
            target_list = test.targets[i] if i < len(test.targets) else []
            target = target_list[0] if target_list else ""
            if not input_text or not target:
                continue
            samples.append({
                'input_id': test.input_ids[i] if i < len(test.input_ids) else f"{test.test_id}.{i}",
                'name': test.name,
                'input': input_text,
                'target': target,
            })
    return samples


def test_m_program_robustness(input_file: str, cli_overrides: dict = None):
    suppress_noise()

    samples = load_samples(input_file)
    if not samples:
        print(f"{R}No valid samples in {input_file}{X}")
        return

    config_path = Path(__file__).parent.parent / 'config' / 'variation_config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    config = {k: v for k, v in cfg.items() if k != '_advanced'}
    if '_advanced' in cfg and isinstance(cfg['_advanced'], dict):
        config['_advanced'] = cfg['_advanced']
    if cli_overrides:
        config.update(cli_overrides)

    target_model = config.pop('target_model', config.pop('model', config.pop('backend_model', 'granite3.3:8b')))
    variation_model = config.get('variation_model', config.get('gen_model', 'qwen2.5:3b'))
    judge_model = config.get('judge_model', 'mistral:7b')
    num_variations = config.get('num_variations', 10)

    # keep gen_model key aligned for variation_engine internals
    config['gen_model'] = variation_model

    print(f"\n{B}Testing M-Program with Problem Variations{X}")
    print(f"{'─' * 70}")
    print(f"  Target model     : {B}{target_model}{X}")
    print(f"  Variation model  : {variation_model}  |  judge: {judge_model}")
    print(f"  Variations       : {num_variations}")
    print(f"  Input file       : {input_file}  ({len(samples)} samples)")
    print(f"{'─' * 70}")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            session = start_session(
                backend_name="ollama", model_id=target_model,
                model_options={ModelOption.TEMPERATURE: 0.1})
    except Exception as e:
        print(f"{R}Cannot connect to Ollama or start session: {e}{X}")
        print(f"\nMake sure Ollama is running:")
        print(f"  ollama serve")
        print(f"\nThen install the model:")
        print(f"  ollama pull {target_model}")
        return

    all_results = []

    for i, sample in enumerate(samples):
        print(f"\n{B}[{i+1}/{len(samples)}] {sample['name']} — {sample['input_id']}{X}")
        print(f"  {D}problem : {sample['input']}{X}")
        print(f"  {D}expected: {sample['target']}{X}")
        print(f"{'─' * 70}")

        call_count = [0]
        variation_counter = [0]

        def program(question: str) -> Any:
            call_count[0] += 1
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                response = session.instruct(question)
            return response.value if hasattr(response, 'value') else response

        _out = sys.stdout

        def on_progress(current, total, status, entry):
            if status == "baseline":
                c = G if entry.get('correct') else R
                _out.write(f"  Baseline  {c}{'PASS' if entry.get('correct') else 'FAIL'}{X}\n")
                _out.flush()
            elif status in ("PASS", "FAIL"):
                variation_counter[0] += 1
                c = G if status == "PASS" else R
                vtype = entry.get('variation_type', '')
                _out.write(f"  Variation {variation_counter[0]}/{total}  {D}[{vtype}]{X}  {c}{status}{X}\n")
                _out.flush()

        try:
            variations = test_with_variations(
                problem=sample['input'],
                expected_answer=sample['target'],
                program=program,
                mellea_session=session,
                config_overrides=config.copy(),
                progress_callback=on_progress,
            )
        except Exception as e:
            print(f"{R}Failed: {e}{X}")
            continue

        if not variations or len(variations) <= 1:
            print(f"{R}No variations generated for this problem.{X}")
            print(f"\nPossible causes:")
            print(f"  1. Variation model not installed: ollama pull {variation_model}")
            print(f"  2. Problem may be too short — try a more detailed problem")
            print(f"  3. Try --quick to skip enrichment and generate faster\n")
            continue

        report = analyze_robustness(variations)
        pc = G if report['pass_rate'] >= 0.7 else (Y if report['pass_rate'] >= 0.4 else R)
        print(f"{'─' * 70}")
        print(f"  Pass rate: {pc}{B}{report['pass_rate']:.0%}{X}"
              f"  ({report['passed']}/{report['total']})"
              f"  |  baseline: {'PASS' if report['baseline_correct'] else 'FAIL'}")
        failed_types = [t for t, r in report['by_variation_type'].items() if r < 1.0]
        if failed_types:
            print(f"  {Y}Your m-program is sensitive to: {', '.join(failed_types)}{X}")

        all_results.append({
            'input_id': sample['input_id'],
            'name': sample['name'],
            'summary': report,
            'variations': variations,
        })
        session.reset()

    # --- Aggregate ---
    if len(all_results) > 1:
        total_v = sum(r['summary']['total'] for r in all_results)
        total_p = sum(r['summary']['passed'] for r in all_results)
        overall = total_p / total_v if total_v else 0.0
        pc = G if overall >= 0.7 else (Y if overall >= 0.4 else R)
        print(f"\n{'═' * 70}")
        print(f"{B}Overall: {pc}{overall:.0%}{X}  ({total_p}/{total_v} variations)  |  "
              f"baseline: {sum(1 for r in all_results if r['summary']['baseline_correct'])}/{len(all_results)}")
        print(f"{'═' * 70}\n")

    # --- Save ---
    out_dir = Path(__file__).parent.parent / 'logs'
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"robustness_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump({
            "timestamp": ts,
            "target_model": target_model,
            "variation_model": variation_model,
            "judge_model": judge_model,
            "input_file": input_file,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Results saved: {D}{out_file}{X}\n")


def parse_args():
    p = argparse.ArgumentParser(description='Test your m-program robustness with problem variations')
    p.add_argument('--input-file', type=str,
                   default=str(Path(__file__).parent / 'data' / 'sample_5problems.json'),
                   help='Unit test JSON file (default: test/data/sample_5problems.json)')
    p.add_argument('--target-model', type=str, default=None,
                   help='Ollama model for the m-program under test (default: granite3.3:8b)')
    p.add_argument('--variation-model', type=str, default=None,
                   help='Model for generating problem variations — supports client/model e.g. groq/llama-3.3-70b-versatile (default: qwen2.5:3b)')
    p.add_argument('--judge-model', type=str, default=None,
                   help='Model for answer evaluation (default: ministral-3:3b)')
    p.add_argument('--num-variations', type=int, default=None,
                   help='Number of variations to generate per problem (default: 10)')
    p.add_argument('--variation-types', type=str, default=None,
                   help='Types of variations, comma-separated or "all" (default: linguistic,referential,pragmatic)')
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: skip LLM feature enrichment (faster)')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {}
    if args.target_model:               overrides['target_model'] = args.target_model
    if args.variation_model:            overrides['variation_model'] = args.variation_model
    if args.judge_model:                overrides['judge_model'] = args.judge_model
    if args.num_variations is not None: overrides['num_variations'] = args.num_variations
    if args.variation_types:            overrides['variation_types'] = args.variation_types
    if args.quick:                      overrides.setdefault('_advanced', {})['quick_mode'] = True
    test_m_program_robustness(args.input_file, overrides or None)
