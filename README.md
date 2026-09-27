# 2D Shallow-Water Solver Service

A **local** HTTP service that integrates the two-dimensional shallow-water
equations on a rectangular grid. Submit a grid, initial water-surface level,
bathymetry and boundary conditions; poll the job; then fetch the surface
level `eta` and the two depth-averaged velocity components `u`, `v` at a
requested time. Jobs are persisted with atomic checkpoints and can resume
after cancellation, timeout or a process restart.

- Language/runtime: Python 3.14, FastAPI, NumPy
- Numerical core: cell-centred finite volumes, Rusanov flux with hydrostatic
  reconstruction, explicit Euler with adaptive CFL time stepping
- Concurrency: bounded thread pool (numpy releases the GIL in the sweeps),
  one isolated solver instance per job

---

## 1. Quick start

```bash
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/docs for the interactive schema
```

Run the tests:

```bash
uv run pytest
```

Reproduce the documented convergence numbers:

```bash
uv run python -m experiments.convergence
```

### Minimal request

```bash
curl -X POST http://127.0.0.1:8000/jobs -H 'content-type: application/json' -d '{
  "grid": {"nx": 32, "ny": 32, "dx": 0.1, "dy": 0.1},
  "initial_level": [1.0, 1.0, "... 1024 floats total ..."],
  "bathymetry": [0.0, 0.0, "... 1024 floats total ..."],
  "boundaries": {
    "west":  {"type": "wall"},
    "east":  {"type": "wall"},
    "south": {"type": "wall"},
    "north": {"type": "wall"}
  },
  "config": {"target_time": 0.5, "cfl": 0.5}
}'
```

Arrays are **row-major flattened** (`shape = [ny, nx]`); cell `(i, j)` is
index `i * nx + j`.

---

## 2. HTTP API

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs` | Submit a solve job (optional `?timeout=seconds`) |
| `GET` | `/jobs` | List persisted job records |
| `GET` | `/jobs/{id}` | Get a job record (status, committed progress) |
| `GET` | `/jobs/{id}/result` | Get the final result (`409` unless completed) |
| `POST` | `/jobs/{id}/cancel` | Cooperatively cancel a running job |
| `POST` | `/jobs/{id}/resume` | Resume a cancelled/interrupted job from its last checkpoint |
| `GET` | `/health` | Liveness probe |

### Statuses

`pending → running → completed | failed | cancelled`

- `completed`: response body also carries `result` and the final checkpoint is
  committed.
- `failed`: a numerical/invalid failure (see failure codes); **no intermediate
  result is ever returned**.
- `cancelled`: cancellation or timeout; the last committed checkpoint remains
  available for `/resume`.

### Job record (excerpt)

```json
{
  "job_id": "…", "status": "running",
  "time": 0.12, "step": 24,
  "result": null,
  "failure_reason": null, "failure_message": null
}
```

### Result shape

```json
{
  "time": 0.5, "step": 91,
  "eta": {"shape": [32, 32], "values": ["…"]},
  "u":   {"shape": [32, 32], "values": ["…"]},
  "v":   {"shape": [32, 32], "values": ["…"]}
}
```

---

## 3. Numerical formulation

### Governing equations

With bed elevation `b`, surface level `eta`, water depth `h = eta - b`,
discharges `hu`, `hv` and velocities `u = hu/h`, `v = hv/h`:

```
h_t   + (hu)_x + (hv)_y = 0
(hu)_t + (hu^2 + g h^2/2)_x + (huv)_y = -g h b_x
(hv)_t + (huv)_x + (hv^2 + g h^2/2)_y = -g h b_y
```

### Space discretisation

- **Cell-centred finite volumes** on a uniform rectangular grid.
- **Rusanov (local Lax–Friedrichs) numerical flux**, wave speed
  `a = |u| + sqrt(g h)` (and the analogous `v` term), applied dimension by
  dimension (x-sweep then y-sweep each step).
- **Hydrostatic reconstruction (Audusse et al., 2004)**: at each face the
  bed is reconstructed as `z_f = max(b_L, b_R)` with face depths
  `h_L* = max(eta_L - z_f, 0)`, `h_R* = max(eta_R - z_f, 0)`. The pressure
  flux uses the reconstructed depths and a balancing topography source is
  added. This makes a **flat lake at rest exactly preserved** over arbitrary
  (including discontinuous) bathymetry — verified to machine precision in the
  tests (`max |eta - eta0| = 0`, speeds `< 1e-10`).
- **Wet/dry treatment**: cells with `h <= 1e-6 m` are treated as dry (zero
  velocity); divisions are guarded so dry beds never produce NaNs.

### Time stepping

Explicit Euler with a global, CFL-controlled step recomputed every step:

```
dt = cfl / ( max(|u|+sqrt(g h))/dx + max(|v|+sqrt(g h))/dy )
```

The last step is clipped so the integration ends **exactly** at `target_time`.
A purely stationary state (flat lake, no driving boundary) jumps directly to
the target time because nothing can evolve.

### Boundary conditions (one ghost cell)

| Type | Ghost-cell rule |
|---|---|
| `wall` | Reflecting: copy depth/bed, negate the **normal** discharge (`hu` on west/east, `hv` on south/north) |
| `periodic` | Wrap the opposite interior cell (may be set independently per axis) |
| `level` | Still ghost at the prescribed surface elevation (`h = max(level - b, 0)`, zero velocity) |

All four sides (`west`, `east`, `south`, `north`) must be specified; they may
be mixed freely. A `level` boundary requires a numeric `level`.

### Stability, failures and "no fake convergence"

The integrator never silently returns a bogus converged field. Distinguishable
failure codes (`failure_reason`) are:

| Code | Trigger |
|---|---|
| `negative_depth` | After a step the depth goes below `-1e-6 m`, or a checkpoint/initial state contains a negative depth beyond tolerance |
| `diverged` | Non-finite (`NaN/Inf`) field, depth beyond `1e4 ×` the initial scale (hard cap `1e6 m`), or speed beyond `100·sqrt(g·h0)` (hard cap `1e4 m/s`) |
| `max_steps_exceeded` | `max_steps` reached before `target_time` |
| `cancelled` | Cancel requested |
| `timeout` | Wall-clock deadline passed between steps |
| `invalid_input` | Wrong array lengths, non-finite input, or initial negative depth |

Use `cfl <= 0.5–0.9` for stable integration. `cfl > 1` is deliberately allowed
for stress testing; the explicit scheme is then unstable and the run is
expected to terminate with `diverged` (covered by tests) rather than return a
plausible-looking but wrong answer.

### Accuracy / error

The scheme is **first-order** in space and time (Rusanov + forward Euler).
Self-convergence of a smooth small-amplitude periodic wave at `t = 0.05 s`
(`experiments/convergence.py`):

| nx | L2 error vs 128 grid | observed L2 order |
|---|---|---|
| 16 | 2.26e-04 | – |
| 32 | 9.80e-05 | 1.21 |
| 64 | 3.27e-05 | 1.58 |

The measured order trends toward 1 on the coarser grids (transient/shock
content) and is limited by first-order reconstruction near gradients. For
production accuracy refine `dx, dy` and/or lower `cfl`; mass is conserved to
machine precision under periodic/wall boundaries (asserted in tests).

### Determinism / reproducibility

- Operation order is fixed (identical NumPy code path per grid); reruns with
  the same request give **bit-identical** results on the same platform/build.
- The only randomness is the optional initial perturbation
  `noise_amplitude`, drawn once from `np.random.default_rng(seed)` at job
  construction. It is **not** re-applied on resume, so a resumed run exactly
  reproduces an uninterrupted one (asserted in tests).

---

## 4. Configuration reference (`config`)

| Key | Default | Meaning |
|---|---|---|
| `target_time` | required (>0) | Physical integration time in seconds |
| `cfl` | `0.5` | CFL number; `<= 0.9` recommended, `> 1` allowed (unstable → `diverged`) |
| `max_steps` | `1_000_000` | Hard cap on accepted steps (`max_steps_exceeded`) |
| `g` | `9.81` | Gravitational acceleration (m/s²) |
| `checkpoint_every` | `100` | Commit a checkpoint every N accepted steps (and always at the final step) |
| `noise_amplitude` | `0.0` | Amplitude (m) of an optional uniform `[-a/2, a/2]` perturbation of the initial level |
| `seed` | `0` | Seed for the deterministic perturbation |

Grid (`grid`): `nx`, `ny` (cells, both `> 1`), `dx`, `dy` (cell size, `> 0`).

Environment variables for the service:

| Variable | Default | Meaning |
|---|---|---|
| `SWS_STORE` | `./.sws_jobs` | Directory for durable job data |
| `SWS_WORKERS` | `4` | Worker-thread count (concurrent jobs) |

Per-request timeout: `POST /jobs?timeout=30` or a `timeout` body field on
`/resume`.

---

## 5. Persistence and checkpoint/resume

```
<SWS_STORE>/<job_id>/record.json       # job metadata (atomically replaced)
<SWS_STORE>/<job_id>/checkpoint.json   # newest committed checkpoint
```

- Every write goes to a unique temp file in the same directory followed by
  `fsync` + `os.replace`: readers and restarts observe only a **complete**
  previous or new file, never a torn/partial write. No `*.tmp` is left behind.
- A checkpoint stores the request plus the full numerical state
  `(eta, hu, hv, b)` and the committed `time/step`.
- **Explicit cancellation** (`POST /cancel`) and **timeout** keep the last
  checkpoint and set the job to a terminal `cancelled`; no intermediate result
  is served. Such a job is *not* restarted automatically — it only continues
  when you call `/resume` — and resumes from the committed time.
- **Service shutdown** (SIGINT / graceful stop) with an unfinished job leaves
  it marked `running` with its last checkpoint. On startup the manager scans
  the store and resumes those `running` jobs from their newest checkpoint; the
  first checkpoint after restart is at or beyond the pre-stop step, so
  committed work is **never recomputed from the initial condition**
  (explicitly asserted in the tests).
- Orphaned `.*.tmp` files left by a process that died mid-write are removed
  when the store is opened; they are never-visible partial writes.

The service is designed for a **single process owning a store directory**. Do
not run two processes against the same `SWS_STORE` concurrently (no distributed
locking; each would try to recover the same running records).

---

## 6. Concurrency and isolation

- A bounded `ThreadPoolExecutor` runs jobs in parallel (numpy sweeps release
  the GIL). The bookkeeping lock is never held during integration.
- Each job owns a private solver instance, state arrays, cancellation token and
  deadline — there is no shared mutable numerical state.
- All exceptions are caught at the job boundary: one job failing (divergence,
  invalid checkpoint, unexpected error) cannot affect another job or corrupt
  another job's files.

---

## 7. Project layout

```
shallow_water/
  models.py    # request/result models, boundary & failure enums, tokens
  solver.py    # numerical core (FV + Rusanov + hydrostatic reconstruction)
  storage.py   # atomic per-job JSON record/checkpoint store
  manager.py   # thread pool, isolation, cancel/timeout, checkpoint/resume
  service.py   # FastAPI app and routes
main.py        # `app = create_app()` for uvicorn
tests/         # solver, storage, manager, end-to-end API suites
experiments/   # convergence measurement
```
