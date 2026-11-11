# Cowork procedural-memory pilot

This project adapts the fixed [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) control loop to the official [Tencent WorkBuddy Office v1.0](https://huggingface.co/datasets/tencent/workbuddy-bench) tasks. It is an executable research starting point. The Office tasks test professional file workflows; they do not by themselves establish performance under live user revisions, months-long collaboration, or a new memory algorithm.

## Fixed inputs

- Dataset revision: `3e8ad839685c589d83a1dd60fe2d15eabb75f27a`.
- Download archive SHA256: `89d4480b483ca2e0555ba96cc7d73c16c78802b17cd822044d5f7f27ab0558bf`.
- mini-swe-agent commit: `04d809ceab9df28f9adaed044884180159172930`.
- Official benchmark framework commit: `625b2233093ae4f23e76be28c1f341d41cc70373`.
- Agent runtime: Python 3.12.13, Bubblewrap 0.4.0; exact installed Python package versions are recorded in `agent-requirements.lock.txt`.
- Split seed: `20261112`; 30 evolution, 10 selection, 10 sealed tasks. The frozen manifest is `splits/office-v1.json`. Its SHA256 is `e641b5541df40b236cab3402eefe866c90258591061c169990fb81fbcb969bfc`.
- Grouping joins equal source cases, exact workspace archives and exact instruction bytes. This release has 50 unique groups by these rules. It does not prove absence of semantic overlap or model pretraining contamination.
- User-supplied model path: `/data/L202500306/L202500306/tlb/models/Qwen3.8-27B`. Architecture metadata is Qwen3.5; checkpoint directory provenance beyond this supplied location has not been independently certified.
- Model server: vLLM 0.25.1, 4 x RTX5090, tensor parallel 4, BF16, context 24576, temperature 0, thinking disabled. The execution wrapper fixes seed 20261112. The deployment script contains the remaining flags.

Dataset/framework licenses remain with their upstream owners. In particular the Tencent repository has a custom license; retain its license and attribution. The agent repository has its own license. No upstream source was modified for this pilot.

## Split policy

Only evolution tasks may be run by the pilot runner or consumed by the memory learner. The runner rejects selection and sealed task identifiers. Selection is for aggregate candidate choice, not task-specific feedback memory. Sealed evaluation is reserved for a later fixed model/harness/memory snapshot, with memory writes disabled and state reset per task. Any future variant inherits its source task's split.

## Pilot protocol (registered before task execution)

1. Run the original instructions of `contract-extract-L3-014` and `procurement-reconcile-L3-015` with no procedural memory, each with at most 40 model calls, 3072 generated tokens per call and 1800 seconds.
2. Generate one procedural candidate from the contract task's user-visible request and execution trace. Do not supply gold answers, rubric text, task verifier code or hidden diagnostic values to the learner. Guidance is capped at 1800 characters and must state applicability and execution evidence.
3. Run both tasks again from fresh source archives with the same fixed execution settings and the generated guidance supplied in the user context. Candidate generation is an additional cost and must be reported.
4. Compare official rule pass fractions per task. Source-task replay and transfer to the other evolution task are distinct results. Provisional acceptance requires a positive aggregate gain, no per-task regression and positive gain on a task other than the memory source. Otherwise reject or label inconclusive.
5. Do not tune the memory after inspecting the transfer task's candidate result. Do not claim statistical significance or global performance from two evolution tasks. This is not yet the A5 matched-total-budget comparison of retained evolution against independent candidate search; implement that experiment separately before making such a claim.

## Execution and evaluation boundaries

The controller runs upstream `DefaultAgent` with upstream `LitellmModel`. Bash actions run under Bubblewrap in isolated user/PID/network namespaces. Only the task workspace is writable; the Python runtime is read-only. The task cannot see the dataset tests/gold, project checkout, SSH credentials or model API. An empty proc directory avoids exposing host-process filesystem roots on a host which forbids mounting proc inside namespaces.

The trusted rule grader is a separate invocation with read-only task outputs and read-only tests/gold, with only verifier logs writable. It invokes each task's official pytest tests and official score reducer. `tests/test.sh` is a Harbor compatibility stub which exits 64 and must not be used as the actual grader.

Some tasks combine rule and LLM scores (for example contract extraction specifies 0.85 rule + 0.15 LLM). All pilot scores are explicitly **official rule component only**. They are not composite scores or leaderboard results. The sandbox environment uses a shared package superset rather than each task's original Docker image; this is an adapted baseline environment. Node and Office formula recalculation are not currently provisioned; tasks requiring them need an explicit environment extension, not a changed scoring interpretation.

Trajectories preserve raw model responses and API usage. Execution records preserve the model, seed, script hash, split hash, upstream commits, wall time, calls and memory hash. The framework's unknown local-model dollar cost is not an estimate of GPU cost; use actual tokens and wall time, and report GPU-hour accounting separately.

## Entry points

```bash
# Run from a provisioned remote project root. Provisioning uses the Tsinghua PyPI mirror.
bash scripts/provision-agent.sh
/tmp/cowork-agent-env/bin/python scripts/preflight.py
/tmp/cowork-agent-env/bin/python scripts/run_office.py --task contract-extract-L3-014 --run baseline-v1
/tmp/cowork-agent-env/bin/python scripts/run_office.py --task procurement-reconcile-L3-015 --run baseline-v1
/tmp/cowork-agent-env/bin/python scripts/learn_memory.py --task contract-extract-L3-014 --source-run baseline-v1 --output procedural-v1
/tmp/cowork-agent-env/bin/python scripts/run_office.py --task contract-extract-L3-014 --run memory-v1 --memory memory/procedural-v1.json
/tmp/cowork-agent-env/bin/python scripts/run_office.py --task procurement-reconcile-L3-015 --run memory-v1 --memory memory/procedural-v1.json
/tmp/cowork-agent-env/bin/python scripts/compare_pilot.py --baseline baseline-v1 --candidate-run memory-v1 --memory memory/procedural-v1.json --tasks contract-extract-L3-014 procurement-reconcile-L3-015 --output results/pilot-v1.json
```

The runner refuses to overwrite an existing run. `--grade-only` reruns grading on an existing artifact without calling the model. Infrastructure recovery must preserve the failed attempt and use a new run label when generation needs to restart. Run records and candidate files should remain immutable after completion.
