# prompt-eval by Alfredo Cardona (SilverBomb-Gaming)

A local-first CLI that loads a prompt suite (YAML or JSON), sends each case to a model, and scores the reply.

The pass/fail decision is the deterministic checks you wrote: substring, regex, a small JSON Schema subset, and an optional latency cap. An LLM judge is optional and does not override those checks. The default model is [Ollama](https://ollama.com) on your machine (`llama3.2`). No cloud API key is required. Prompts and replies stay on localhost unless you opt into an OpenAI-compatible endpoint.

The installable project name is `prompt-eval-harness`. The command is `prompt-eval`.

Built by Alfredo Cardona ([SilverBomb-Gaming](https://github.com/SilverBomb-Gaming)).

## In the owner's words

I wanted a prompt suite I could run on a laptop against Ollama, with pass or fail coming from checks I wrote, and a judge that only talks about text the model actually returned.

## What it is / isn't

**It is** a portfolio CLI for one job: run the cases in a file, print a pass/fail report with aggregate counts, and exit non-zero when a scored case fails. Every row is a case this process listed or attempted. Latency is the time this process waited on that call.

**It isn't** a hosted eval product, a dataset hub, or a writer that may invent case ids or scores. A judge pass cannot rescue a failed check. A judge reply that does not quote the model output does not get a score. Cases that were not selected, and cases that had not started when the provider failed, are left out of the report.

## Demo

To properly demo this, You will need Python 3.11+. Ollama is only required for the last command. The dry run does not call a model.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

prompt-eval run --suite samples/demo.yaml --dry-run
```

```text
Suite demo (4 cases) from samples/demo.yaml
- exact-refund  checks: contains, not_contains, regex, max_latency_ms  judge: yes
- json-priority  checks: contains, json_schema  judge: no
- unit-test-plain  checks: contains, not_contains  judge: yes
- greeting-bounds  checks: contains, not_contains, regex  judge: no
Dry run. No model calls.
```

`samples/demo.json` is the same suite. One case, still no model:

```bash
prompt-eval run --suite samples/demo.json --case greeting-bounds --dry-run --json
```

With Ollama:

```bash
ollama pull llama3.2
prompt-eval run --suite samples/demo.yaml
```

That runs all four cases. `exact-refund` and `unit-test-plain` also call the judge, so they make a second request. Skip the judge and score with checks only:

```bash
prompt-eval run --suite samples/demo.yaml --no-judge
```

The same document is written to `.prompt-eval/last-results.json`. JSON on stdout:

```bash
prompt-eval run --suite samples/demo.yaml --json --no-save
```

The first call can be slow while Ollama loads the model. `exact-refund` allows 120000 ms. If a cold start trips that check or `PROMPT_EVAL_TIMEOUT`, run `ollama run llama3.2` once and try again, or raise the cap in the suite.

A scored report uses this layout. `RESULT` follows the checks when the case has any. A local run fills `provider`, `model`, latency, and judge scores from the calls that actually ran. The numbers in the sample below only show the columns.

```text
prompt-eval  demo
suite: samples/demo.yaml
provider: ollama   model: llama3.2

ID               RESULT  LATENCY   CHECKS  JUDGE      SOURCE
exact-refund     PASS    842.0 ms  4/4     pass 1.00  checks
json-priority    FAIL    510.0 ms  1/2     -          checks
unit-test-plain  PASS    640.0 ms  3/3     fail 0.20  checks
greeting-bounds  PASS    430.0 ms  3/3     -          checks

4 cases, 3 passed, 1 failed, 0 errors

json-priority
  json_schema: $.priority: expected one of ['low']
unit-test-plain
  judge fail (not the pass/fail source): The output does not use the word function.
```

`unit-test-plain` still passes. The judge disagreed, and the line says the judge is not the pass/fail source.

## How a run is built

```text
suite file  ──►  load and select  ──►  one model call per case  ──►  deterministic checks
(yaml/json)      (this program)         (Ollama by default)          (this program)

optional rubric  ──►  second model call  ──►  drop ungrounded verdicts  ──►  report
                     (same provider)         (this program)
```

1. **Load.** UTF-8 YAML or JSON. Unknown keys, duplicate ids, a bad regex, and an unsupported schema keyword fail here, before any model call. `--dry-run` stops after printing the plan.
2. **Call.** One chat completion per selected case, temperature 0. The case is plain text. It is not forced into JSON.
3. **Score.** Every configured check runs, including checks after one has already failed. If the case has any deterministic check, those checks alone set pass/fail.
4. **Judge, when the case has a rubric and you did not pass `--no-judge`.** A second call asks for JSON. The system prompt forbids inventing facts that are not in the model output. Those sentences are pinned by `tests/test_prompts.py`. A pass is kept only when `evidence` is a substring of that output. Otherwise the judge score is null.
5. **Report.** Stdout is the table, or one JSON document with `--json`. The last run is also written to `.prompt-eval/last-results.json` unless you pass `--no-save` or the command is a dry run.

`--case ID` selects one id. An unknown id exits 2 and does not print a pass/fail row.

## How pass/fail is decided

| Deterministic checks | Judge rubric | `--no-judge` | What sets pass/fail |
| --- | --- | --- | --- |
| yes | no | no | Checks. The judge column is `-`. |
| yes | yes | no | Checks. The judge is recorded and does not flip the result. |
| yes | yes | yes | Checks. The judge is `skipped`. |
| no | yes | no | The judge. A grounded pass is the only way to pass. |
| no | yes | yes | Error, before any model call. There is nothing to score. |
| no | no | either | Rejected when the file is loaded. |

A judge status of `ungrounded` or `error` stores `score: null`. The harness does not substitute 0 or 1. When the judge is the only scorer, that case fails because it never earned a grounded pass. When checks are present, the case result stays on the checks.

A passing judge verdict must include `evidence` copied from the model output. A failing verdict may use an empty `evidence` string. The reason is kept only on a grounded verdict. Extra JSON keys, including a list of other case ids, are ignored.

The judge sees one case: its id, its rubric, the user task, and the model output. It does not see the rest of the suite. The task is labeled as context. The prompt tells the judge not to treat the task as something the model said.

The judge uses the same provider and model as the case. There is no separate judge model.

## Suite schema

A suite is a mapping with `name` and `cases`. Each case has an `id` and exactly one of `prompt` or `messages`.

```yaml
name: demo
cases:
  - id: exact-refund
    system: Follow the user instruction exactly.
    prompt: "Reply with exactly this sentence: Refunds are available for 30 days."
    checks:
      contains:
        - 30 days
      not_contains:
        - lifetime warranty
      regex:
        - "(?i)refunds are available"
      max_latency_ms: 120000
    judge:
      rubric: Pass only if the output states that refunds are available for 30 days.
```

`messages` replaces `prompt` and `system`. Put the system turn inside the list.

```yaml
  - id: greeting-bounds
    messages:
      - role: system
        content: You write one short greeting and nothing else.
      - role: user
        content: Greet Ada by name and wish her a good morning.
    checks:
      contains:
        - Ada
      regex:
        - "(?i)good morning"
```

| Field | Rule |
| --- | --- |
| `name` | Required, non-empty |
| `cases` | Required, at least one |
| `id` | Required, unique, 1–64 characters: letters, digits, `.`, `_`, `-` |
| `prompt` | User text. Mutually exclusive with `messages` |
| `system` | Optional system text, only with `prompt` |
| `messages` | List of `system`, `user`, and `assistant` turns. At least one `user` |
| `checks.contains` | Every substring must appear. Case-sensitive. A single string is accepted |
| `checks.not_contains` | None of these substrings may appear. Case-sensitive |
| `checks.regex` | Python `re.search`. All patterns must match. Invalid patterns fail at load |
| `checks.json_schema` | Light subset, below. The reply must parse as JSON |
| `checks.max_latency_ms` | Positive integer. The measured latency must be less than or equal to it |
| `judge.rubric` | Optional. The judge scores only this case's model output |

`contains`, `not_contains`, and `regex` accept either a string or a list. Unknown keys anywhere in the suite are an error, so a typo does not silently drop a check.

### JSON Schema subset

Supported keywords: `type`, `required`, `properties`, `items`, `enum`.

Supported types: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.

`integer` and `number` do not accept booleans. Extra object keys are allowed. Keywords such as `minLength` fail at load, so the file cannot pretend a keyword is enforced.

The check parses the raw reply, or a single fenced block, or the outermost `{...}` / `[...]` span. `contains` still sees the raw reply, fences included.

## Output

Human output is a table. `PASS` / `FAIL` is a scored case. `ERROR` means the model call did not return text, so no check scores were recorded (`passed` is null). `-` in JUDGE means the case has no rubric. `skipped` means you passed `--no-judge` on a case that has a rubric.

Stdout is the table, or one pretty JSON document with `--json`. Stderr carries `Saved ...`, `warning:` lines, and `error:` lines.

Scored JSON:

```json
{
  "schema_version": 1,
  "dry_run": false,
  "suite": "samples/demo.yaml",
  "name": "demo",
  "provider": "ollama",
  "model": "llama3.2",
  "ok": false,
  "aborted": null,
  "case_count": 4,
  "passed_count": 3,
  "failed_count": 1,
  "error_count": 0,
  "results": [
    {
      "id": "exact-refund",
      "passed": true,
      "status": "scored",
      "latency_ms": 842.0,
      "output": "Refunds are available for 30 days.",
      "error": null,
      "pass_source": "checks",
      "checks": [
        {"kind": "contains", "expected": "30 days", "passed": true, "detail": null}
      ],
      "judge": {
        "status": "scored",
        "passed": true,
        "score": 1.0,
        "reason": "The output states a 30 day refund window.",
        "evidence": "30 days",
        "detail": null
      }
    }
  ],
  "results_path": ".prompt-eval/last-results.json"
}
```

`case_count` is the number of rows. `passed_count + failed_count + error_count` equals `case_count`. `pass_source` is `checks` or `judge`. Dry-run JSON has `cases` (id, check names, whether a rubric exists) and does not have scores.

Exit codes:

| Code | When |
| --- | --- |
| 0 | Every scored case passed, or the command was a dry run |
| 1 | At least one scored case failed, and every selected case was attempted |
| 2 | Bad suite, unknown id, `--results` with `--no-save`, a judge-only case with `--no-judge`, or the provider failed before the suite finished |

A provider failure stops the loop. The report includes cases that already finished and the case that failed to connect (`status: "error"`, no check rows). Later cases are omitted. The error is also printed on stderr.

## Results file

Each scored run replaces `.prompt-eval/last-results.json` in the current working directory. The directory is gitignored. `--results PATH` chooses another file. `--no-save` writes nothing. A dry run writes nothing. `--results` and `--no-save` together are an error. If you pass `--results` on a dry run, the command warns and still does not write.

The file is the same object printed by `--json`. This is a single file, not a history of runs.

## Configuration

Copy `.env.example` to `.env` in the working directory (the directory you run the command from), or export the variables yourself. Existing environment variables win over `.env`. `.env` is gitignored.

| Variable | Default | Role |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Chat model for the Ollama path |
| `OLLAMA_NUM_CTX` | `8192` | Context window sent to Ollama |
| `PROMPT_EVAL_PROVIDER` | `ollama` | `ollama` or `openai` |
| `PROMPT_EVAL_TIMEOUT` | `120` | Seconds for each model call |
| `OPENAI_API_KEY` | empty | Only for the OpenAI-compatible path. Ollama does not use it |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible base URL, usually ending in `/v1` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name when the provider is `openai` |

Ollama is the default, and it does not need an API key. The OpenAI-compatible path is opt-in. Setting `OPENAI_BASE_URL` in the environment for some other tool does not switch this CLI. You have to set `PROMPT_EVAL_PROVIDER=openai` (or pass `--provider openai`).

```bash
# Remote or local OpenAI-compatible server (LM Studio, a proxy, api.openai.com, …)
export PROMPT_EVAL_PROVIDER=openai
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
prompt-eval run --suite samples/demo.yaml
```

`api.openai.com` refuses to run without `OPENAI_API_KEY`. A local compatible server may omit the key. `--provider` and `--model` override the environment for one command.

Case calls are plain chat at temperature 0. Judge calls also request JSON mode (`format: json` on Ollama, `response_format` on the OpenAI-compatible path). The samples fit easily inside `OLLAMA_NUM_CTX`.

## Scope / out of scope

**In scope**

- One YAML or JSON suite per command, with a single-case filter
- Ollama by default, OpenAI-compatible chat as an option
- Deterministic checks: substring, regex, a small JSON Schema subset, latency
- An optional judge that cannot override those checks
- A dry run, a JSON report, and a local last-run file
- A non-zero exit when a scored case fails

**Out of scope**

- Hosted dashboards, dataset downloads, and a database of past runs
- BLEU, embeddings, semantic similarity, or a second model used only as the judge
- Parallel case execution, retries, and a temperature flag
- The rest of JSON Schema (`minLength`, `$ref`, composition keywords, and so on)
- Inventing case ids, check rows, or judge scores for work that did not happen
- A guarantee that two models will phrase the same answer the same way. The checks are the contract

## Samples

| File | What it is for |
| --- | --- |
| `samples/demo.yaml` | Four cases: an exact refund sentence with a latency cap and a rubric, a JSON object, a one-sentence definition with a rubric, and a multi-message greeting |
| `samples/demo.json` | The same suite in JSON |

The prompts ask the model to say the phrases the checks look for, so a local `llama3.2` usually passes after the model is loaded.

## Layout

```text
src/prompt_eval/
  cli.py           # prompt-eval run …
  suite.py         # YAML / JSON loading
  models.py        # suite schema
  scorers.py       # contains, not_contains, regex, json_schema, latency
  schema_check.py  # the JSON Schema subset
  prompts.py       # judge system prompt and the no-fabrication rules
  judge.py         # parse a judge reply and drop ungrounded scores
  runner.py        # case loop
  report.py        # table and JSON
  llm.py           # Ollama and OpenAI-compatible clients
  store.py         # .prompt-eval/last-results.json
  dotenv.py        # optional .env loader
samples/
tests/             # pytest, no live model
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

`pytest` mocks the model client. It does not start Ollama and does not call the network. Suite loading, the scorers, the schema subset, the no-fabrication judge prompt, ungrounded judge scores, the runner's pass/fail rule, dry-run, and the CLI are covered offline.

## License

MIT © 2026 Alfredo Cardona
