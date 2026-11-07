# Warm-rate timer (`mu2efastsim.bench`)

One way to time a fast workflow, so every throughput number is measured the same way:

- the entry point runs for **three passes in one process** over the same in-memory input;
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

- Each pass must redo all the work for the same events; do not cache results between passes.
- If the function returns nothing, or only host values, while device work is still queued, the warm rate is wrong.
  Return the outputs, or call `jax.block_until_ready` yourself before returning.
- Input loading into host memory belongs in a separate prepare function (reported as `prepare_s`, not setup).
- If the pass writes its output, pass `--output-included` so the report says so.

## Command line

```bash
python3 -m mu2efastsim.bench mypkg.flow:run_chain --events 20000 \
    --prepare mypkg.flow:load_input --kwargs '{"seed": 1}' --dtype float32 \
    --workflow W5 --boundary "generated CRY cosmics -> CRV coincidences" --out bench.json
```

`--prepare MODULE:FUNC` is called once as `FUNC(N)` and its return value is passed to the entry point each pass;
without it the entry point gets `N`. `--kwargs` goes to the entry point only. `--passes` defaults to 3.

## From Python

```python
from mu2efastsim import bench

data = load_input(20000)
report = bench.run(run_chain, events=20000, args=(data,), kwargs={"seed": 1}, dtype="float32",
                   workflow="W5", boundary="generated CRY cosmics -> CRV coincidences")
print(bench.format_report(report))
```

## Report (JSON)

| Key | Meaning |
|---|---|
| `pass_s` | wall seconds of each pass |
| `warm_rate` | events/s over passes 2..P |
| `setup_s`, `n10` | as above |
| `device`, `backend`, `dtype`, `dtype_source` | where and how it ran |
| `prepare_s`, `output_included` | input loading time; whether the pass writes output |
| `max_rss_mb`, `python`, `jax`, `slurm_job_id`, `time_utc`, `command` | provenance |

## Cold start

The timer measures inside one process. To time a fresh process through exit (interpreter start, imports, compile
cache), wrap the whole command with a process timer such as the umbrella's `mixing/process_timer.py` and report
that wall time beside `setup_s`.
