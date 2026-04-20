"""Test your m-program on a single problem.

Shows the basics: define your m-program, provide a test problem and expected
answer, and see if it handles semantic variations of that problem correctly.

Run:
    python examples/testing/101_simple_test.py
"""
import sys, os, io, contextlib, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mellea import start_session
from mellea.backends import ModelOption
from mellea_contribs.tools.variation_engine import test_with_variations, analyze_robustness

G, R, Y, D, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"

# Suppress mellea/BenchDrift noise
for _name in ['BenchDrift', 'benchdrift', 'mellea_contribs', 'mellea',
               'httpx', 'httpcore', 'urllib3', 'requests', 'fancy_logger']:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
try:
    from mellea.helpers.fancy_logger import FancyLogger
    _fl = FancyLogger.get_logger()
    _fl.setLevel(logging.CRITICAL); _fl.handlers = []; _fl.propagate = False
except Exception:
    pass
os.environ['TQDM_DISABLE'] = '1'
os.environ['MELLEA_LOG_LEVEL'] = 'CRITICAL'

# Step 1: Start a Mellea session (the m-program under test)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    session = start_session(backend_name="ollama", model_id="granite3.3:8b",
                            model_options={ModelOption.TEMPERATURE: 0.1})

# Step 2: Define your m-program
def my_program(question: str) -> str:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        response = session.instruct(question)
    return response.value if hasattr(response, 'value') else str(response)

# Step 3: Define the problem and expected answer
problem = "You are facing north. You turn right, then turn right again, then turn left. What direction are you now facing?"
expected_answer = "east"

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

# Step 4: Run variation testing
print(f"\n{B}Testing M-Program with Problem Variations{X}")
print(f"{'─' * 70}")
print(f"  Problem  : {problem}")
print(f"  Expected : {expected_answer}")
print(f"{'─' * 70}")

variations = test_with_variations(
    problem=problem,
    expected_answer=expected_answer,
    program=my_program,
    mellea_session=session,
    config_overrides={"gen_model": "mistral:7b", "num_variations": 5, "no_enrich": True},
    progress_callback=on_progress,
)

# Step 5: Report
report = analyze_robustness(variations)
pc = G if report['pass_rate'] >= 0.7 else (Y if report['pass_rate'] >= 0.4 else R)
print(f"{'─' * 70}")
print(f"  Pass rate: {pc}{B}{report['pass_rate']:.0%}{X}  ({report['passed']}/{report['total']} variations)"
      f"  |  baseline: {'PASS' if report['baseline_correct'] else 'FAIL'}")
failed_types = [t for t, r in report['by_variation_type'].items() if r < 1.0]
if failed_types:
    print(f"  {Y}Your m-program is sensitive to: {', '.join(failed_types)}{X}")
