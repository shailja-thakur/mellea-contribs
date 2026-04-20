# Robustness Testing for Mellea M-Programs

Test how consistently your m-program answers semantic variations of a problem. Uses [BenchDrift](https://github.com/IBM/BenchDrift) (`demo-ui` branch) for variation generation.

## Setup

```bash
# 1. Install mellea-contribs with robustness dependencies
cd mellea-contribs
uv sync --extra robustness
# or: pip install -e ".[robustness]"

# 2. Ollama models
ollama serve
ollama pull granite3.3:8b   # m-program under test
ollama pull mistral:7b       # variation generation + validation
ollama pull llama3.1:8b      # answer evaluation (judge)

# 3. (Optional) For fast variation generation via Groq
export GROQ_API_KEY=gsk_your_key_here    # get from https://console.groq.com/keys
```

## Run

```bash
cd mellea-contribs

# Quick test on built-in sample problems
python test/test_mprogram_robustness.py --quick

# With a custom unit test JSON file
python test/test_mprogram_robustness.py --input-file test/data/sample_5problems.json

# Use Groq for faster variation generation
python test/test_mprogram_robustness.py --variation-model groq/llama-3.3-70b-versatile --quick

# Or use the example scripts directly
python examples/testing/101_simple_test.py
python examples/testing/102_test_with_variations.py --input-file test/data/sample_5problems.json
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-file` | `test/data/sample_5problems.json` | Unit test JSON file to load problems from |
| `--target-model` | `granite3.3:8b` | Ollama model for the m-program under test |
| `--variation-model` | `mistral:7b` | Model for generating and validating variations. Supports `client/model` format |
| `--judge-model` | `llama3.1:8b` | Model for evaluating answers |
| `--num-variations` | `10` | Number of variations to generate per problem |
| `--variation-types` | 3 core types | Variation types, comma-separated or `all` |
| `--quick` | off | Skip LLM feature enrichment (faster) |

**Model format:** `--variation-model` and `--judge-model` accept `client/model` format. Supported clients: `ollama`, `groq`, `rits`, `vllm`, `openai`.
- `mistral:7b` → Ollama local (default)
- `groq/llama-3.3-70b-versatile` → Groq cloud

You can mix clients — e.g., Groq for fast variation generation + Ollama for m-program testing:
```bash
--variation-model groq/llama-3.3-70b-versatile --target-model granite3.3:8b
```

**Recommended Groq models for variation generation:**
- `groq/llama-3.3-70b-versatile` — fast, good quality
- `groq/llama-3.1-8b-instant` — fastest
- `groq/qwen-qwq-32b` — strong reasoning
- `groq/gemma2-9b-it` — compact, good quality

Requires `GROQ_API_KEY` env var. Free at https://console.groq.com/keys

## Expected Output

```
Testing M-Program with Problem Variations
──────────────────────────────────────────────────────────────────────
  Target model     : granite3.3:8b
  Variation model  : mistral:7b  |  judge: llama3.1:8b
  Variations       : 5
  Input file       : test/data/sample_5problems.json  (5 samples)
──────────────────────────────────────────────────────────────────────

[1/5] spatial reasoning — test_spatial_001.1
  problem : You are facing north. You turn right, then turn right again, then turn left...
  expected: east
──────────────────────────────────────────────────────────────────────
  Baseline       PASS
  Variation 1/3  [passive_active_voice]   FAIL
  Variation 2/3  [order_dependency]       PASS
  Variation 3/3  [domain_shift]           FAIL
──────────────────────────────────────────────────────────────────────
  Pass rate: 33%  (1/3)  |  baseline: PASS
  Your m-program is sensitive to: passive_active_voice, domain_shift
```

Results are saved as JSON in `logs/` after each run, including all variation texts, m-program answers, and pass/fail status.

## Input File Format

Problems are loaded from a Mellea unit test JSON file. See `test/data/sample_5problems.json` for an example. Each test entry needs `name`, `instructions`, `inputs`, and `targets`.

## Note on the Variation Model

The variation model (`--variation-model`) generates and validates semantic variations of your problem. It should ideally be more capable than the model under test — it needs to understand the problem well enough to rephrase it while preserving the answer. Using Groq with a 70B model is recommended for both speed and quality.
