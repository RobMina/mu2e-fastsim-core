# Warm-rate timer (`mu2efastsim.bench`)

One way to time a fast workflow, so every throughput number is measured the same way:

- the entry point runs for **three passes in one process**, by default over the same in-memory input, or with
  `--fresh-inputs` over new inputs every pass (see below; use it for headline rows);
- **warm rate** = events/s over passes 2 and 3, with the device synchronised before the clock stops
  (`jax.block_until_ready` on the returned value, then a CuPy synchronise if CuPy is loaded);
- **setup** = pass 1 minus the mean warm pass (compile, table and library loading, paid once per process);
- **N10** = `9 x setup_s x warm_rate`: the job size at which setup is 10 % of the wall time;
- the **device label** comes from the job's node: `nvidia-smi -L` (with the MIG profile when the job holds a slice),
  or the CPU model and the number of usable threads on a CPU run;
- the **dtype** is the one you pass, else the JAX default float (the x64 flag), else float64.

The multiplier against stock is `warm_rate` divided by the stock events/s per single-core grid slot **at the same
boundary** (input the stock workflow reads -> output the analysis consumes). A number at a different boundary is a
stage number; say so.

## Entry-point contract

```python
def run_chain(events_or_input, **kwargs):
    ...                      # all the work of one pass
    return device_outputs    # return the device arrays (or a pytree of them): the timer waits on them
```

- Each pass must redo all the work for the events it is given; do not cache results between passes.
- If the function returns nothing, or only host values, while device work is still queued, the warm rate is wrong.
  Return the outputs, or call `jax.block_until_ready` yourself before returning.
- Input loading into host memory belongs in a separate prepare function (reported as `prepare_s`, not setup).
- If the pass writes its output, pass `--output-included` so the report says so.

## Fresh inputs (use for headline rows)

With fixed inputs, passes 2 and 3 see exactly the events of pass 1, so every shape-dependent compile (a `jit` retrace
for a new array length, an eager operation on a new shape) is paid in pass 1 and hits JAX's cache afterwards. A
production job draws new events all the time and can keep compiling, so a fixed-input warm rate can overstate the
production rate. `--fresh-inputs` makes each pass draw its own inputs:

- the prepare hook is called before every pass, **outside the clock**, as `PREPARE(N, seed=SEED + pass_index)`
  (`--seed`, default 1, so seeds 1, 2, 3); its time is reported per pass in `prepare_pass_s` and summed in
  `prepare_s`;
- the hook must take a `seed` keyword (or `**kwargs`) and draw different events for different seeds. A hook without
  it is refused with an error naming the hook (exit code 2), and so is a hook that returns the very same object for
  two seeds: the timer never falls back to fixed inputs;
- the warm rate then includes the compiles that new events cause, as in production; `compile_per_pass` shows them.

**Headline rows (scorecard multipliers) should be measured with `--fresh-inputs` from now on.** The default stays
fixed inputs so that rows measured before stay comparable; every report says which mode ran (`inputs`).

## Command line

```bash
python3 -m mu2efastsim.bench mypkg.flow:run_chain --events 20000 \
    --prepare mypkg.flow:load_input --kwargs '{"seed": 1}' --dtype float32 \
    --workflow W5 --boundary "generated CRY cosmics -> CRV coincidences" --out bench.json
```

`--prepare MODULE:FUNC` is called once as `FUNC(N)` and its return value is passed to the entry point each pass;
without it the entry point gets `N`. `--kwargs` goes to the entry point only. `--passes` defaults to 3. Fresh inputs:

```bash
python3 -m mu2efastsim.bench mypkg.flow:run_chain --events 20000 \
    --prepare mypkg.flow:draw_input --fresh-inputs --seed 1 --dtype float32 --workflow W2 --out bench.json
```

## From Python

```python
from mu2efastsim import bench

data = load_input(20000)
report = bench.run(run_chain, events=20000, args=(data,), kwargs={"seed": 1}, dtype="float32",
                   workflow="W5", boundary="generated CRY cosmics -> CRV coincidences")
print(bench.format_report(report))

# fresh inputs: draw_input(20000, seed=1), (..., seed=2), (..., seed=3), one per pass, outside the clock
report = bench.run(run_chain, events=20000, inputs=bench.fresh_inputs(draw_input, 20000, seed=1),
                   seeds=[1, 2, 3], dtype="float32", workflow="W2")
```

## Report (JSON)

| Key | Meaning |
|---|---|
| `pass_s` | wall seconds of each pass |
| `warm_rate` | events/s over passes 2..P |
| `setup_s`, `n10` | as above |
| `device`, `backend`, `dtype`, `dtype_source` | where and how it ran |
| `prepare_s`, `output_included` | input loading time (summed over passes with fresh inputs); whether the pass writes output |
| `inputs`, `seeds`, `prepare_pass_s` | `fixed` or `fresh`; with fresh inputs, the seed and prepare seconds of each pass |
| `compile_per_pass`, `compile_source` | JAX compiles of each pass from `jax.monitoring` (below), else null |
| `max_rss_mb`, `python`, `jax`, `slurm_job_id`, `time_utc`, `command` | provenance |

## Compile counts

When JAX is loaded, one `jax.monitoring` listener counts, for each pass: `backend_compiles` and `backend_compile_s`
(every compile request that misses JAX's in-memory cache, from `jit` or an eager operation, including requests served
by the persistent compilation cache), `persistent_cache_hits`, and the tracing (`traces`, `trace_s`) and lowering
(`lowerings`, `lowering_s`) before them. A pass that started before JAX was imported has `null` (typically pass 1
when the entry point imports JAX itself). The counts cost nothing measurable and do not change the timing; a fixed-input
run should show zero compiles in passes 2..P, and a fresh-input run shows what new events cost.

## Cold start

The timer measures inside one process. To time a fresh process through exit (interpreter start, imports, compile
cache), wrap the whole command with a process timer such as the umbrella's `mixing/process_timer.py` and report
that wall time beside `setup_s`.
