"""Variation engine for robustness testing of Mellea m-programs.

Uses BenchDrift (github.com/IBM/BenchDrift) to generate semantic variants of a
problem, then tests each variant through the m-program and evaluates correctness.
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from mellea import MelleaSession

from benchdrift.pipeline.feature_relevance import (
    get_problem_features,
    enrich_features_with_llm,
    rank_transformations_two_level,
    parse_axes,
    _get_valid_axes,
    _rank_axes_by_features,
    TRANSFORMATION_TO_AXIS,
)
from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine
from benchdrift.pipeline.comprehensive_variation_engine_v2 import (
    clean_model_response,
    is_valid_question,
)
from benchdrift.pipeline.council_validator import (
    get_judge_validation_prompt,
    build_judge_user_prompt,
    parse_judge_response,
)
from benchdrift.models.model_client import ModelClientFactory

logger = logging.getLogger(__name__)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VALID_CLIENTS = {'ollama', 'groq', 'rits', 'vllm', 'openai'}


def _parse_model_spec(spec: str) -> tuple:
    """Parse 'client/model' into (client_type, model_name). Default: ollama."""
    if '/' in spec:
        client, model = spec.split('/', 1)
        if client.lower() in VALID_CLIENTS:
            return client.lower(), model
    return 'ollama', spec


def _clean_response(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return clean_model_response(text)


def _generate_one_variation(gen_client, problem: str, trans_name: str,
                            config: dict, max_retries: int = 2) -> str:
    system_prompt = (
        f"You are an expert at creating intent-preserving question variations.\n\n"
        f"TASK: Create a {trans_name} variation of the given problem.\n\n"
        f"TRANSFORMATION GOAL: {config['prompt']}\n\n"
        f"UNIVERSAL RULES:\n"
        f"1. PRESERVE the exact answer\n"
        f"2. MAINTAIN all mathematical/logical relationships\n"
        f"3. Numbers: format can change (5 -> five), value CANNOT (5 -> 6)\n"
        f"4. Units: convert correctly or not at all\n"
        f"5. Use PLAIN TEXT only\n"
        f"6. Return ONLY the question inside <question> tags\n"
        f"7. Do NOT explain your reasoning\n\n"
        f"<question>Your transformed question here</question>"
    )
    user_prompt = f"Original: {problem}\n\nReturn only the <question>...</question>."

    for attempt in range(max_retries + 1):
        try:
            raw = gen_client.get_single_response(
                system_prompt=system_prompt, user_prompt=user_prompt,
                max_new_tokens=1024, temperature=0.5)
        except Exception:
            if attempt == max_retries:
                return ""
            continue
        cleaned = _clean_response(raw)
        if cleaned and cleaned.strip().upper() == "SKIP":
            return ""
        if cleaned and is_valid_question(cleaned):
            return cleaned
        user_prompt = (f"Original: {problem}\n\nReturn ONLY the question text "
                       f"inside <question> tags. No explanation, no analysis.")
    return ""


def _validate_variation(judge_client, original: str, variation: str,
                        ground_truth: str) -> bool:
    try:
        system_prompt = get_judge_validation_prompt()
        user_prompt = build_judge_user_prompt(original, variation, ground_truth)
        raw = judge_client.get_single_response(
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_new_tokens=32, temperature=0.0)
        return parse_judge_response(raw) == "VALID"
    except Exception:
        return True


def _get_ranked_transformations(baseline: str, features: dict, config: dict) -> tuple:
    top_k = config.get('num_variations', config.get('top_k', 10))
    all_types = UnifiedVariationEngine.get_all_transformation_types()
    valid_axes = _get_valid_axes(features)
    ranked_axes = _rank_axes_by_features(features, valid_axes)
    ranked = rank_transformations_two_level(baseline, features, all_types, top_k=top_k,
                                            pre_ranked_axes=ranked_axes)
    return all_types, ranked


def generate_variants(baseline: str, target: str, config: dict) -> list[dict]:
    """Generate and validate semantic variants of a baseline problem.

    Returns a list of dicts with keys: variation_type, variant_text.
    Use eval_and_score() to test each variant through your m-program.
    """
    gen_model = config.get('gen_model', 'qwen3:8b')
    ollama_url = config.get('ollama_url', OLLAMA_BASE_URL)
    timeout = config.get('timeout', 120)
    no_enrich = config.get('quick_mode', config.get('no_enrich', False))
    skip_validation = config.get('skip_validation', False)

    gen_client_type, gen_model_name = _parse_model_spec(gen_model)
    gen_client = ModelClientFactory.create_client(gen_client_type, gen_model_name)

    features = get_problem_features(baseline)
    if not no_enrich:
        try:
            features.update(enrich_features_with_llm(baseline, ollama_url, gen_model, timeout=timeout))
        except Exception:
            pass

    all_types, ranked = _get_ranked_transformations(baseline, features, config)

    def _gen_and_validate(trans_name, trans_config):
        variation = _generate_one_variation(gen_client, baseline, trans_name, trans_config)
        if not variation or variation.strip() == baseline.strip():
            return None
        if not skip_validation:
            if not _validate_variation(gen_client, baseline, variation, target):
                return None
        return {'variation_type': trans_name, 'variant_text': variation}

    valid_ranked = [(t, all_types[t]) for t, s, a in ranked if t in all_types]
    variants = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_gen_and_validate, t, cfg): t for t, cfg in valid_ranked}
        for f in as_completed(futures):
            result = f.result()
            if result:
                variants.append(result)

    return variants


def eval_and_score(variant_text: str, test: Any, target: str,
                   gen_session: Any, judge_session: Any) -> tuple[bool, str]:
    """Run variant through gen_session and score with TestBasedEval judge.

    Args:
        variant_text: The variant (or baseline) problem text to test.
        test: A TestBasedEval instance used as the judge component.
        target: Expected correct answer string.
        gen_session: Mellea session for generating the response.
        judge_session: Mellea session for scoring the response.

    Returns:
        (passed, prediction) — bool and the raw prediction string.
    """
    import io, contextlib
    from mellea.stdlib.components.simple import SimpleComponent

    try:
        from mellea.cli.eval.runner import parse_judge_output
    except ImportError:
        import re as _re, json as _json
        def parse_judge_output(text: str) -> tuple:
            try:
                clean = text.strip().strip('`').lstrip('json').strip()
                data = _json.loads(clean)
                return int(data.get('score', 0)), data.get('justification', '')
            except Exception:
                m = _re.search(r'"score"\s*:\s*([01])', text)
                return (int(m.group(1)), '') if m else (None, '')

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gen_output = gen_session.act(SimpleComponent(instruction=variant_text))
    prediction = str(gen_output)
    gen_session.reset()

    test.set_judge_context(input_text=variant_text, prediction=prediction,
                           targets_for_input=[target])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        judge_output = judge_session.act(test)
    judge_session.reset()

    score, _ = parse_judge_output(str(judge_output))
    passed = score == 1 if score is not None else False
    return passed, prediction


# --- Helpers for test_with_variations ---

def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    for pattern in [
        r'(?:total\s+cost|total|answer|result)\s*(?:is|=|:)\s*\$?([\d,]+\.?\d*)',
        r'=\s*\$?([\d,]+\.?\d*)\s*$',
        r'\*\*\$?([\d,]+\.?\d*)\*\*',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].replace(',', '')
    dollars = re.findall(r'\$([\d,]+\.?\d*)', text)
    if dollars:
        return dollars[-1].replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', re.sub(r'(\d),(\d)', r'\1\2', text))
    if nums:
        return nums[-1]
    return text.strip()


def _answers_match(predicted: str, truth: str) -> bool:
    def _normalize(s):
        if not s:
            return ""
        extracted = _extract_final_answer(s)
        if extracted:
            s = extracted
        s = s.lower().strip().lstrip('$')
        for suffix in (".", "!", "?"):
            if s.endswith(suffix):
                s = s[:-1].strip()
        frac = re.search(r'-?\d+\s*/\s*\d+', s)
        if frac:
            return re.sub(r'\s', '', frac.group())
        nums = re.findall(r'-?\d+\.?\d*', re.sub(r'(\d),(\d)', r'\1\2', s))
        return nums[-1] if nums else s

    a, b = _normalize(predicted), _normalize(truth)
    if a == b:
        return True
    try:
        def _f(x):
            return float(x.split('/')[0]) / float(x.split('/')[1]) if '/' in x else float(x)
        return abs(_f(a) - _f(b)) < 1e-6
    except (ValueError, ZeroDivisionError, IndexError):
        pass
    return a in b or b in a


def _llm_judge_answer(judge_client, problem: str, expected_answer: str, predicted: str) -> bool:
    try:
        system_prompt = (
            "You are an answer evaluation judge. Compare the predicted answer to the "
            "expected answer. They may be in different formats but represent the same value. "
            "Respond with ONLY 'CORRECT' or 'INCORRECT'."
        )
        user_prompt = (
            f"Problem: {problem}\n"
            f"Expected answer: {expected_answer}\n"
            f"Predicted: {predicted}\n\n"
            f"Is the predicted answer correct? Reply ONLY 'CORRECT' or 'INCORRECT'."
        )
        raw = judge_client.get_single_response(
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_new_tokens=32, temperature=0.0)
        return "CORRECT" in raw.upper() and "INCORRECT" not in raw.upper()
    except Exception:
        return _answers_match(predicted, expected_answer)


def _default_answer_extractor(response: Any) -> str:
    if hasattr(response, 'value'):
        return str(response.value)
    return str(response)


def _get_answer(problem, program, answer_extractor, target_client) -> str:
    try:
        if program is not None:
            return answer_extractor(program(problem))
        raw = target_client.get_single_response(
            system_prompt="Solve the problem. Return ONLY the final answer.",
            user_prompt=problem, max_new_tokens=256, temperature=0.1)
        return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
    except Exception as e:
        return f"ERROR: {e}"


def test_with_variations(
    problem: str,
    expected_answer: str,
    program: Optional[Callable[[str], Any]] = None,
    mellea_session: Optional[MelleaSession] = None,
    answer_extractor: Optional[Callable[[Any], str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, str, Dict], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate semantic variations of a problem and test the m-program on each.

    For each ranked variation:
      1. Generate a variation (gen_model via Ollama)
      2. Validate it preserves meaning (gen_model as judge)
      3. Test it through the m-program
      4. Evaluate correctness (string match, or LLM judge if use_llm_judge=True)

    Args:
        problem: The baseline problem text to generate variations for.
        expected_answer: The correct answer to the problem.
        program: The m-program function (takes str, returns Any).
        mellea_session: Mellea session (required with program).
        answer_extractor: Extract answer string from m-program response.
        config_overrides: See variation_config.yaml for available options.
        progress_callback: Called after each variation: (current, total, status, entry).

    Returns:
        List of variation dicts with: is_baseline, is_variant, variation_type,
        variant_answer, correct, valid, expected_answer.
    """
    if (program is None) != (mellea_session is None):
        raise ValueError("Both program and mellea_session must be provided together.")
    if config_overrides is None:
        raise ValueError("config_overrides is required.")

    advanced = config_overrides.get('_advanced', {}) if isinstance(config_overrides.get('_advanced'), dict) else {}

    gen_model = config_overrides.get('gen_model', 'qwen3:8b')
    judge_model = config_overrides.get('judge_model', gen_model)
    target_model = advanced.get('target_model', config_overrides.get('target_model', 'granite3.3:8b'))
    top_k = config_overrides.get('num_variations', config_overrides.get('top_k', 10))
    use_axes = config_overrides.get('variation_types', config_overrides.get('use_axes',
                            advanced.get('variation_types',
                            'linguistic,referential,pragmatic,structural,constraint_targeted')))
    ollama_url = config_overrides.get('ollama_url', OLLAMA_BASE_URL)
    timeout = config_overrides.get('timeout', 120)
    no_enrich = advanced.get('quick_mode', config_overrides.get('no_enrich', False))
    skip_validation = advanced.get('skip_validation', config_overrides.get('skip_validation', False))
    use_llm_judge = advanced.get('use_llm_judge', config_overrides.get('use_llm_judge', False))

    if answer_extractor is None:
        answer_extractor = _default_answer_extractor

    gen_client_type, gen_model_name = _parse_model_spec(gen_model)
    judge_client_type, judge_model_name = _parse_model_spec(judge_model)
    target_client_type, target_model_name = _parse_model_spec(target_model)

    gen_client = ModelClientFactory.create_client(gen_client_type, gen_model_name)
    judge_client = gen_client if judge_model == gen_model else ModelClientFactory.create_client(judge_client_type, judge_model_name)
    target_client = None if program else ModelClientFactory.create_client(target_client_type, target_model_name)

    features = get_problem_features(problem)
    if not no_enrich:
        try:
            features.update(enrich_features_with_llm(problem, ollama_url, gen_model, timeout=timeout))
        except Exception as e:
            logger.debug(f"Feature enrichment skipped: {e}")

    enabled_axes = parse_axes(use_axes)
    all_types = {k: v for k, v in UnifiedVariationEngine.get_all_transformation_types().items()
                 if TRANSFORMATION_TO_AXIS.get(k) in enabled_axes}
    valid_axes = _get_valid_axes(features, enabled_axes=enabled_axes)
    ranked_axes = _rank_axes_by_features(features, valid_axes)
    ranked = rank_transformations_two_level(
        problem, features, all_types, top_k=top_k,
        pre_ranked_axes=ranked_axes, enabled_axes=enabled_axes)

    baseline_answer = _get_answer(problem, program, answer_extractor, target_client)
    if use_llm_judge:
        baseline_correct = _llm_judge_answer(judge_client, problem, expected_answer, baseline_answer)
    else:
        baseline_correct = _answers_match(baseline_answer, expected_answer)

    results = [{
        'is_baseline': True, 'is_variant': False,
        'variation_type': 'baseline',
        'modified_problem': problem,
        'expected_answer': expected_answer,
        'variant_answer': baseline_answer,
        'correct': baseline_correct,
        'valid': True,
    }]

    if progress_callback:
        progress_callback(0, len(ranked), "baseline", results[0])

    # Generate + validate all variations in parallel (same as generate_variants)
    valid_ranked = [(t, s, a) for t, s, a in ranked if t in all_types]

    def _gen_and_validate(trans_name, score, axis):
        variation = _generate_one_variation(gen_client, problem, trans_name, all_types[trans_name])
        if not variation or variation.strip() == problem.strip():
            return None
        if not skip_validation:
            if not _validate_variation(judge_client, problem, variation, expected_answer):
                return None
        return (trans_name, score, axis, variation)

    generated = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_gen_and_validate, t, s, a): t for t, s, a in valid_ranked}
        for f in as_completed(futures):
            result = f.result()
            if result:
                generated.append(result)

    total = len(generated)
    for processed, (trans_name, score, axis, variation) in enumerate(generated, 1):
        if progress_callback:
            progress_callback(processed, total, "waiting", {})

        answer = _get_answer(variation, program, answer_extractor, target_client)
        if use_llm_judge:
            correct = _llm_judge_answer(judge_client, variation, expected_answer, answer)
        else:
            correct = _answers_match(answer, expected_answer)

        status = "PASS" if correct else "FAIL"
        entry = {
            'is_baseline': False, 'is_variant': True,
            'variation_type': trans_name,
            'transformation_axis': axis,
            'relevance_score': score,
            'modified_problem': variation,
            'expected_answer': expected_answer,
            'variant_answer': answer,
            'correct': correct,
            'valid': True,
        }
        results.append(entry)
        if progress_callback:
            progress_callback(processed, total, status, entry)

    return results


def analyze_robustness(variations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute robustness metrics: pass rate and per-type breakdown."""
    variants = [v for v in variations if v.get('is_variant')]
    if not variants:
        return {"error": "No variations", "pass_rate": 0.0, "total": 0}

    passed = sum(1 for v in variants if v.get('correct'))
    by_type = {}
    for v in variants:
        t = v.get('variation_type', 'unknown')
        if t not in by_type:
            by_type[t] = {'total': 0, 'passed': 0}
        by_type[t]['total'] += 1
        if v.get('correct'):
            by_type[t]['passed'] += 1

    baseline = next((v for v in variations if v.get('is_baseline')), None)
    return {
        "pass_rate": passed / len(variants),
        "total": len(variants),
        "passed": passed,
        "failed": len(variants) - passed,
        "baseline_correct": baseline.get('correct', False) if baseline else False,
        "by_variation_type": {t: s['passed'] / s['total'] for t, s in by_type.items()},
    }
