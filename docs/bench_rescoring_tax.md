# Rescoring Tax Benchmark

This benchmark measures the latency and recall tradeoff of rescoring depth $K$ for the IsoSearch pipeline.

## Run

```bash
cargo bench --bench rescoring_tax
```

### Determinism

For fully deterministic results, run with a single Rayon thread:

```bash
RAYON_NUM_THREADS=1 cargo bench --bench rescoring_tax
```

Set `RAYON_NUM_THREADS` to a higher value to study multi-core behavior.

### Optional Poincare Stage

You can toggle the hyperbolic projection stage with an environment flag:

```bash
ISOSEARCH_USE_POINCARE=1 cargo bench --bench rescoring_tax
```

## Output CSV

Results are written to:

```
bench-results/rescoring_tax.csv
```

Columns:
- `k`: rescoring depth.
- `recall_at_k`: overlap with exact top-K results.
- `t_candidates_ns`: bucket intersection + HNSW traversal time.
- `t_rescore_ns`: rescoring time for top-K candidates.
- `t_total_ns`: end-to-end time (normalize + project + hash + quantize + candidates + rescore).

## How to Interpret the Pareto Frontier

Plot `recall_at_k` vs `t_total_ns` for each $K$.
- Points that improve recall without a large latency increase form the Pareto frontier.
- Choose the smallest $K$ on the frontier that meets your recall target.
- If recall improvements flatten while latency continues to rise, the break-even point has been exceeded.
