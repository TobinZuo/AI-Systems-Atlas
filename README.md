# AI Systems Atlas

An interactive map of models, kernels, memory, collectives, and clusters.

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

- A rotatable 3D system topology built with React Three Fiber
- Play, pause, single-step, seek, and speed controls
- A deterministic DDP event model with explicit dependencies
- Animated Ring All-Reduce tensor chunks
- A component inspector that explains role and knowledge boundaries
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
      +-- Timeline
      +-- Event explanation
      +-- Component inspector
```

Important directories:

```text
src/
  domain/       Shared scenario and event types
  scenarios/    Educational simulations such as DDP
  sim/          Pure timeline and validation functions
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
