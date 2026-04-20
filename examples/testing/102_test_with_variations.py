"""Test your m-program using a Mellea unit test JSON file.

Loads problems from a unit test JSON, generates variations of each, and
reports pass rate per problem and overall. This mirrors what developers
already do with unit tests — just adds robustness coverage automatically.

Run:
    python examples/testing/102_test_with_variations.py
    python examples/testing/102_test_with_variations.py --input-file path/to/tests.json
"""
import sys, os, io, json, contextlib, argparse, logging
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mellea import start_session
from mellea.backends import ModelOption
from mellea.stdlib.components.unit_test_eval import TestBasedEval
from mellea_contribs.tools.variation_engine import test_with_variations, analyze_robustness

G, R, Y, D, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"

DEFAULT_INPUT = str(Path(__file__).parent.parent.parent / 'test' / 'data' / 'sample_5problems.json')


def suppress_noise():
    for name in ['BenchDrift', 'benchdrift', 'mellea_contribs', 'mellea',
                 'httpx', 'httpcore', 'urllib3', 'requests', 'fancy_logger']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        from mellea.helpers.fancy_logger import FancyLogger
        fl = FancyLogger.get_logger()
        fl.setLevel(logging.CRITICAL); fl.handlers = []; fl.propagate = False
    except Exception:
        pass
    os.environ['TQDM_DISABLE'] = '1'
    os.environ['MELLEA_LOG_LEVEL'] = 'CRITICAL'


def main(input_file: str, target_model: str = "granite3.3:8b",
         variation_model: str = "mistral:7b", judge_model: str = "llama3.1:8b",
         num_variations: int = 5):

    suppress_noise()
    tests = TestBasedEval.from_json_file(input_file)
    samples = []
    for test in tests:
        for i, input_text in enumerate(test.inputs):
            target_list = test.targets[i] if i < len(test.targets) else []
            target = target_list[0] if target_list else ""
            if input_text and target:
                samples.append({'name': test.name, 'input': input_text, 'target': target})

    if not samples:
        print(f"{R}No valid samples in {input_file}{X}"); return

    print(f"\n{B}Testing M-Program with Problem Variations{X}")
    print(f"{'─' * 70}")
    print(f"  Target model     : {B}{target_model}{X}")
    print(f"  Variation model  : {variation_model}  |  judge: {judge_model}  |  variations: {num_variations}")
    print(f"  Input file       : {input_file}  ({len(samples)} problems)")
    print(f"{'─' * 70}")

    session = start_session(backend_name="ollama", model_id=target_model,
                            model_options={ModelOption.TEMPERATURE: 0.1})

    def my_program(question: str) -> str:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = session.instruct(question)
        return r.value if hasattr(r, 'value') else str(r)

    config = {"gen_model": variation_model, "judge_model": judge_model,
              "num_variations": num_variations, "no_enrich": True}
    all_results = []

    for i, sample in enumerate(samples):
        print(f"\n{B}[{i+1}/{len(samples)}] {sample['name']}{X}")
        print(f"  {D}problem : {sample['input']}{X}")
        print(f"  {D}expected: {sample['target']}{X}")
        print(f"{'─' * 70}")

        variation_counter = [0]
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

        variations = test_with_variations(
            problem=sample['input'], expected_answer=sample['target'],
            program=my_program, mellea_session=session,
            config_overrides=config.copy(),
            progress_callback=on_progress)

        if not variations or len(variations) <= 1:
            print(f"  {R}No variations generated — skipping{X}"); continue

        report = analyze_robustness(variations)
        pc = G if report['pass_rate'] >= 0.7 else (Y if report['pass_rate'] >= 0.4 else R)
        print(f"{'─' * 70}")
        print(f"  Pass rate: {pc}{B}{report['pass_rate']:.0%}{X}"
              f"  ({report['passed']}/{report['total']})"
              f"  |  baseline: {'PASS' if report['baseline_correct'] else 'FAIL'}")
        failed_types = [t for t, r in report['by_variation_type'].items() if r < 1.0]
        if failed_types:
            print(f"  {Y}Your m-program is sensitive to: {', '.join(failed_types)}{X}")
        all_results.append({'name': sample['name'], 'summary': report})
        session.reset()

    if len(all_results) > 1:
        total_v = sum(r['summary']['total'] for r in all_results)
        total_p = sum(r['summary']['passed'] for r in all_results)
        overall = total_p / total_v if total_v else 0.0
        pc = G if overall >= 0.7 else (Y if overall >= 0.4 else R)
        print(f"\n{'═' * 70}")
        print(f"{B}Overall: {pc}{overall:.0%}{X}  ({total_p}/{total_v} variations)")
        print(f"{'═' * 70}\n")

    out_dir = Path(__file__).parent.parent.parent / 'logs'; out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"robustness_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump({"timestamp": ts, "input_file": input_file, "results": all_results},
                  f, indent=2, default=str)
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input-file', default=DEFAULT_INPUT)
    p.add_argument('--target-model', default='granite3.3:8b', help='Model for the m-program under test')
    p.add_argument('--variation-model', default='mistral:7b', help='Model for generating and validating problem variations')
    p.add_argument('--judge-model', default='llama3.1:8b', help='Model for evaluating answers')
    p.add_argument('--num-variations', type=int, default=5)
    args = p.parse_args()
    main(args.input_file, args.target_model, args.variation_model, args.judge_model, args.num_variations)
