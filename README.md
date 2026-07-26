# AI Systems Atlas

An interactive, inspectable map of models, kernels, memory, collectives, and clusters.

The first learning path follows one gradient through a four-rank DDP iteration:

```text
loss.backward()
  -> Autograd scheduling
  -> CUDA gradient kernel
  -> gradient bucket ready
  -> Ring Reduce-Scatter
  -> Ring All-Gather
  -> parameter.grad writeback
  -> optimizer update
```

## What is included

- Concept Mode for deterministic, reversible teaching scenarios
- Trace Mode with a multi-lane CPU, CUDA Stream, NCCL, NVLink, and HBM timeline
- Local Chrome Trace / PyTorch Profiler JSON import with X and B/E event support
- Five synchronized views: system topology, rank/process, GPU internals, Ring collective, and AdamW
- A rotatable 3D system topology built with React Three Fiber
- Play, pause, single-step, seek, and speed controls
- A deterministic DDP event model with explicit dependencies
- A numeric Ring All-Reduce simulator with per-rank, per-chunk, per-round state
- Concrete gradient values, send/receive routes, and reduction equations
- CPU thread, CUDA Stream/Event, HBM address, SM, Block, Warp, and thread-lane drill-downs
- A causal inspector that explains prerequisites, actions, results, call paths, and misconceptions
- Responsive light and dark themes
- Reduced-motion support
- GitHub Pages deployment workflow

## Architecture

The simulation model is independent from the rendering layer:

```text
Scenario events
      |
Simulation state
      |
      +-- 3D scene
      +-- Rank process and CUDA streams
      +-- GPU / SM / Warp cutaway
      +-- Numeric Ring debugger
      +-- AdamW state transition
      +-- Timeline and causal inspector

Chrome / PyTorch Trace JSON
      |
Trace importer and normalization
      |
      +-- Process and thread lanes
      +-- CUDA compute and communication streams
      +-- Event arguments and overlapping work
      +-- Links back to Concept Mode
```

Important directories:

```text
src/
  domain/       Shared scenario and event types
  scenarios/    Educational simulations such as DDP
  sim/          Pure timeline, validation, and Ring All-Reduce functions
  trace/        Chrome Trace parser and category inference
  traces/       Built-in profiler-style teaching traces
  store/        Playback and selection state
  components/   3D and 2D presentation layers
```

This separation makes it possible to add tokenizer, Transformer, optimizer,
FSDP, diffusion, and inference paths without rebuilding the player.

## Local development

Requirements:

- Node.js 18 or newer
- npm 9 or newer

Install and run:

```bash
npm install
npm run dev
```

The local app is served at `http://localhost:5173/AI-Systems-Atlas/` so local
development matches the GitHub Pages base path.

Validation:

```bash
npm run typecheck
npm test
npm run build
```

## Importing a trace

Open **Trace 分析**, then choose **导入 Trace JSON**. The importer accepts Chrome
Trace Event JSON, including complete `X` slices and paired `B` / `E` slices.
PyTorch Profiler exports use this event format. Files are parsed locally in the
browser and are not uploaded. The current browser importer intentionally limits
files to 25 MB; trim long captures before importing.

A small reference file is available at
`public/samples/minimal-pytorch-trace.json`.

## Adding a scenario

1. Create a scenario file under `src/scenarios/`.
2. Describe each event with `start`, `duration`, `actor`, `layer`, and
   `dependencies`.
3. Validate it with `validateScenario()`.
4. Add view rules only when the scenario introduces a new visual concept.

The DDP scenario in `src/scenarios/ddp.ts` is the reference implementation.

## GitHub Pages

The workflow in `.github/workflows/deploy.yml` tests and builds the project on
pushes to `main` or `master`, then publishes `dist/` to GitHub Pages.

In the repository settings, select **Settings -> Pages -> GitHub Actions** as
the deployment source.

The Vite base path is configured for:

```text
https://TobinZuo.github.io/AI-Systems-Atlas/
```

## License

MIT
