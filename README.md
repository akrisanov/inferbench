# Inferbench

An experiment harness for benchmarking and diagnosing LLM serving stacks.

## Why Inferbench?

Benchmarking an LLM endpoint is relatively straightforward. Understanding **why** its performance changes is harder.

A production LLM serving path often contains several layers:

```text
Client
  ↓
API Gateway
  ↓
LLM-aware Router
  ↓
Inference Server
  ↓
GPU
```

A load generator can tell you that p99 TTFT increased or throughput dropped,
but that alone does not explain what happened inside the serving stack.

Inferbench is designed to connect workload configuration, client-side benchmark results,
server-side telemetry, and environment metadata into a single reproducible experiment.

## Goals

1. **Reproducible workloads**: Describes benchmark workloads declaratively and run
   the same experiment against different serving configurations.
2. **Serving-stack targets**: Benchmarks different points in an LLM serving path using the same workload.
   Targets are generic OpenAI-compatible endpoints. Inferbench does not depend on a particular gateway or routing implementation.
3. **Client and server telemetry**: Combines client-observed performance metrics with telemetry from the serving system.
4. **Experiment provenance**: A benchmark result contains enough context to understand and reproduce the experiment.
5. **Comparison and diagnosis**: Inferbench makes it easy to compare experiments and understand performance trade-offs.

A comparison will eventually look like:

```text
Metric                    Baseline    Candidate      Delta
TTFT p99                    820 ms       510 ms     -37.8%
TPOT p99                   31.2 ms      32.1 ms      +2.9%
Throughput                42 req/s      55 req/s     +31.0%
KV cache utilization p95       91%           73%     -18 pp
Prefix cache hit rate          34%           71%     +37 pp
Preemptions                     47             3        -44
```

Inferbench reports measurements and trade-offs and does not automatically decide which configuration is better.

## Design principles

1. **Experiment harness, not load generator**: Inferbench does not aim to implement another high-performance load generator.
   Instead, it integrates existing benchmark tools behind drivers.
2. **Client metrics are not enough**: A benchmark helps explain performance, not only measure it.
3. **Reproducibility over convenience**: Experiments explicitly capture the parameters that materially affect results.
4. **Raw evidence before interpretation**: Inferbench preserves raw benchmark results and collected telemetry whenever practical.
5. **Serving-stack agnostic**: Inferbench provides integrations for specific systems, but its experiment model
   does not require a particular gateway, router, inference server, or observability stack.

## Architecture

The intended high-level data flow is:

```text
Experiment
    │
    ▼
Experiment Runner
    │
    ├──► Load Driver
    │       │
    │       └──► vllm bench serve
    │
    ├──► Telemetry Collectors
    │       │
    │       ├──► serving metrics
    │       └──► GPU / infrastructure metrics
    │
    ├──► Metadata Collectors
    │
    ▼
Experiment Artifact
    │
    ├──► Raw results
    ├──► Normalized metrics
    ├──► Metadata
    └──► Report
```

## Roadmap

Near-term development is focused on completing the first useful vertical slice:

```text
Experiment configuration
        ↓
vLLM benchmark execution
        ↓
Benchmark result
        ↓
Experiment artifact
        ↓
Report
```

After that, the project will expand toward:

- Prometheus telemetry collection
- environment and serving configuration capture
- experiment comparison
- rate sweeps and saturation analysis
- cold-cache and warm-cache experiments
- prefix-cache-aware workloads
- multi-replica routing experiments
- additional benchmark drivers

## Development

The project requires Python 3.12 or newer and uses [uv](https://docs.astral.sh/uv/) for project and dependency management.

Install dependencies:

```bash
uv sync
```

Run the quality checks:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

Run the CLI:

```bash
uv run inferbench --help
```

## Project scope test

When considering a new feature, ask:

> Does this help run a reproducible LLM serving experiment or explain the difference between experiments?

If not, it is probably outside the current scope of Inferbench.

---

(C) 2026, Andrey Krisanov
