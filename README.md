# AI Systems Atlas

An interactive, inspectable knowledge atlas for large-model and AIGC systems.

The site is organized by large-model systems topics.
Papers, source code, tests, and profiler traces are evidence attached to each topic.

The root page currently maps 45 topics across seven domains:

- Data and representation
- Model architecture
- Training mechanics
- GPU and performance
- Distributed training
- Inference systems
- Generative and multimodal systems

The first active journey follows one gradient from Autograd through AdamW and GPU execution,
then compares how DDP, ZeRO-1, and FSDP manage distributed model state:

```text
loss.backward()
  -> Autograd scheduling
  -> AdamW first and second moments
  -> CUDA gradient kernel
  -> gradient bucket ready
  -> Ring Reduce-Scatter
  -> Ring All-Gather
  -> parameter.grad writeback
  -> optimizer update

DDP
  -> replicate parameters, gradients, and optimizer state
ZeRO-1
  -> shard AdamW state, update on owners, broadcast parameters
FSDP
  -> shard parameters, gradients, and optimizer state
  -> All-Gather one layer for compute, then reshard
```

## What is included

- A topic-based knowledge map with prerequisites and cross-domain learning paths
- An interactive Autograd lab for dynamic graph recording, saved tensors, reverse scheduling, leaf-gradient accumulation, and version-counter errors
- An interactive Gradient lab for per-sample derivatives, batch reduction, `.grad` storage, finite-difference checks, optimizer direction, and DDP averaging
- An interactive AdamW lab for moment history, bias correction, decoupled weight decay, parameter groups, CUDA execution, and distributed memory ownership
- An interactive Process and Rank lab for torchrun identities, rendezvous, Process Groups, GPU binding, control-plane failures, and collective data paths
- An interactive Collective lab comparing nine communication contracts, partial receive buffers, backend call metadata, and rank mismatch failures
- An interactive GPU execution-model lab from CPU launch through Grid, Block, SM, Warp, Lane, registers, and HBM
- An interactive Tensor-to-kernel journey through the PyTorch Dispatcher, CUDA launch configuration, and asynchronous error boundaries
- An interactive CUDA Stream timeline comparing serial execution, safe Event synchronization, and a missing-dependency race
- Hash routes that work directly on GitHub Pages
- Route-level code splitting so each interactive topic loads only when opened
- Complete DDP, ZeRO-1 Sharded Optimizer, and FSDP topics
- A shared DDP, ZeRO-1, and FSDP comparison lab with the same model-size and world-size inputs
- Concept Mode for deterministic, reversible teaching scenarios
- Trace Mode with a multi-lane CPU, CUDA Stream, NCCL, NVLink, and HBM timeline
- Local Chrome Trace / PyTorch Profiler JSON import with X and B/E event support
- A single-screen DDP Playground inspired by direct-manipulation teaching tools
- Editable per-rank gradients with immediate, deterministic recomputation
- Manual phase controls for every Reduce-Scatter and All-Gather round
- Stable system flow, chunk matrix, selected-rank transfer, and causal explanation views
- Semantic colors for framework, compute, collective, network, memory, and optimizer work
- A deterministic DDP event model with explicit dependencies
- A numeric Ring All-Reduce simulator with per-rank, per-chunk, per-round state
- Concrete gradient values, send/receive routes, and reduction equations
- A causal inspector that explains why work starts, what the system does, and what state changes
- Responsive light and dark themes
- Reduced-motion support
- GitHub Pages deployment workflow

## Architecture

The topic catalog, executable simulations, and rendering layer are independent:

```text
Knowledge domains and topic relationships
      |
      +-- Knowledge map
      +-- Learning journeys
      +-- Topic routes
      +-- Evidence metadata

Scenario events
      |
Simulation state
      |
      +-- Editable rank-local gradients
      +-- Numeric Ring state matrix
      +-- Selected-rank send and receive equations
      +-- AdamW state transition
      +-- Stable causal inspector

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
  content/      Topic catalog, prerequisites, and learning journeys
  domain/       Shared scenario and event types
  scenarios/    Educational simulations such as DDP
  sim/          Pure timeline, validation, and Ring All-Reduce functions
  trace/        Chrome Trace parser and category inference
  traces/       Built-in profiler-style teaching traces
  playground/   Pure teaching models for GPU execution and distributed training
  components/   Playground and Trace presentation layers
```

This separation makes it possible to add tokenizer, Transformer, optimizer,
FSDP, diffusion, and inference topics without rebuilding the whole site.

## Site routes

```text
#/                    Knowledge map
#/paths               Cross-topic learning journeys
#/training/autograd   Dynamic graph, backward scheduling, and leaf gradients
#/training/gradient   Sample contributions, batch reduction, grad buffers, and DDP averaging
#/training/adamw      AdamW moments, decay, parameter groups, HBM, and state sharding
#/gpu/architecture    GPU execution-model and gradient kernel playground
#/gpu/cuda-kernel     Tensor operator, dispatcher, launch, and error journey
#/gpu/cuda-stream     CUDA Stream, Event, overlap, and race playground
#/distributed/process-rank  OS processes, rank identity, rendezvous, and Process Groups
#/distributed/collective    Broadcast, Reduce, Gather, Scatter, AllReduce, and AllToAll contracts
#/distributed/ddp     DDP concept playground and trace workspace
#/distributed/zero-1  Sharded Optimizer ownership and broadcast playground
#/distributed/fsdp    FSDP parameter lifecycle and memory playground
#/distributed/compare Persistent-state, communication, and tradeoff comparison
```

Planned topics already live in the catalog. A topic becomes available when it
receives a route and an executable explanation.

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
