# Follow-up execution report — 2026-09-21

## JEV gate

Before retrying, the guard was checked: configured backend
`codex_subscription`; OpenAI API opt-in unset/disabled; runtime billing mode
subscription. `probe_models.py --verify` returned authentication `ok` and
verification `ok` for Luna, Sol, and Astra.

JEV was queried through `run.bat` for the first experiment. One attempt failed
before model execution because a PowerShell prompt contained non-UTF-8 bytes.
The ASCII retry reached the Codex CLI but timed out after 30 seconds and
returned `MODEL BACKENDS TEMPORARILY UNAVAILABLE`. No model, hypothesis,
training run, or submission was selected. The retry telemetry reported 3,084
input and 822 output tokens. Its internal estimated-cost field is not treated
as an API charge because the backend is the ChatGPT subscription route.

## Local checks

All checks below were local and used no credentials, network, or submission:

| Check | Result |
|---|---|
| `python -m pytest -q` | **2 passed**, 4 `pycocotools` deprecation warnings; exit code 0 |
| `python -m compileall -q scripts tests` | Passed |
| `audit_submission.py` on empty baseline | Valid; 0 rows, 180 test images, 0 overlaps |

Pytest emitted a non-fatal `WinError 5` while cleaning its temporary directory
at interpreter exit. The current local test count is 2, not the target 11; no
tests were invented or modified without a JEV-selected implementation task.

## Status

The JEV gate remains blocking for model selection and training. The only
permitted next work is further local deterministic analysis or another JEV
consultation when the subscription backend is responsive.

## Actualización: conexión JEV validada

The canonical connector at
`E:\Rescate-C-2026-09-21\Documents\JEV-Orchestrator` now resolves the
multi-key recovery file by profile instead of sending the whole file as one
HTTP header. Windows Credential Manager remains preferred, and no credential
material is recorded in this report. The UTF-8 Codex bridge is also present in
`runtime.py`; the protected runtime/configuration files were not changed in
this follow-up. A regression test was added for the profile mapping.

### Execution log

| Execution | JEV/model route | Backend / billing | Input / output tokens | Latency | Quality / escalation |
|---|---|---|---:|---:|---|
| Parser authorization | TypeSafe JEV `jev-1.13.0`; JEV chose `approve` | `typesafe` / N/A | 437 / 31 | 0.789 s | Approved by JEV; no escalation |
| Runtime UTF-8 authorization | TypeSafe JEV `jev-1.13.0`; JEV chose `approve_utf8_bridge` | `typesafe` / N/A | 518 / 39 | 0.629 s | Approved by JEV; no timeout-policy change |
| `nautiluskit` probe | TypeSafe JEV `jev-1.13.0` | `typesafe` / N/A | 325 / 20 | 0.724 s | Connected; reachability 0.41 |
| `lareliquia` probe | TypeSafe JEV `jev-1.13.0` | `typesafe` / N/A | 325 / 20 | 0.574 s | Connected; reachability 0.42 |
| `agnuxo` probe | TypeSafe JEV `jev-1.13.0` | `typesafe` / N/A | 325 / 20 | 0.610 s | Connected; reachability 0.43 |
| Full `run.bat` check | JEV chose Luna first (`luna_scout`) | `codex` / `subscription` | JEV: 2,942 / 823; Codex: unavailable | 30.075 s | Codex CLI timeout; no Sol/Astra escalation |

### Final verification

The JEV orchestrator suite reports **20 passed** and
`python -m compileall -q jev_orchestrator tests` passes. The final
`probe_models.py --verify` reports backend `codex_subscription`, authentication
`ok`, and Luna/Sol/Astra all `ok`.

The full route still stops at the existing 30-second Codex CLI timeout after
JEV has successfully decided the Luna route. OpenAI API remains disabled. Any
future timeout-policy change requires a new explicit JEV decision.
