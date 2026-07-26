import {
  ArrowRight,
  Cube,
  GithubLogo,
  Moon,
  Play,
  Sun,
} from "@phosphor-icons/react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useReducedMotion } from "motion/react";
import { ComponentRail } from "./components/ComponentRail";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { EventTimeline } from "./components/EventTimeline";
import { Inspector } from "./components/Inspector";
import { LearningPaths } from "./components/LearningPaths";
import { SceneLoading } from "./components/SceneLoading";
import { usePlayback } from "./hooks/usePlayback";
import { ddpScenario } from "./scenarios/ddp";
import { eventAtTime } from "./sim/simulator";
import { useSimulationStore } from "./store/simulation";

const SceneCanvas = lazy(() =>
  import("./components/SceneCanvas").then((module) => ({
    default: module.SceneCanvas,
  })),
);

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  const saved = window.localStorage.getItem("atlas-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

function App() {
  usePlayback();
  const reducedMotion = Boolean(useReducedMotion());
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const currentTime = useSimulationStore((state) => state.currentTime);
  const play = useSimulationStore((state) => state.play);
  const activeEvent = useMemo(
    () => eventAtTime(ddpScenario.events, currentTime),
    [currentTime],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("atlas-theme", theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="AI Systems Atlas home">
          <span className="brand-mark" aria-hidden="true">
            <Cube size={22} weight="duotone" />
          </span>
          <span>AI Systems Atlas</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#explorer">Explorer</a>
          <a href="#learning-paths">Learning paths</a>
          <a
            className="github-link"
            href="https://github.com/TobinZuo/AI-Systems-Atlas"
            target="_blank"
            rel="noreferrer"
          >
            <GithubLogo size={18} weight="fill" aria-hidden="true" />
            <span>GitHub</span>
          </a>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </nav>
      </header>

      <main id="top">
        <section className="explorer-intro" id="explorer">
          <div>
            <p className="eyebrow">Interactive systems model</p>
            <h1>Follow one gradient through DDP.</h1>
            <p className="intro-copy">
              Rotate the system, step through each event, and inspect what every
              layer knows.
            </p>
          </div>
          <button type="button" className="primary-action" onClick={play}>
            <Play size={17} weight="fill" aria-hidden="true" />
            Start journey
          </button>
        </section>

        <section className="explorer-workspace" aria-label="DDP interactive explorer">
          <ComponentRail />

          <div className="stage-column">
            <div className="stage-header">
              <div>
                <strong>System view</strong>
                <span>Drag to orbit. Select a component to inspect it.</span>
              </div>
              <div className="stage-context">
                <span>4 ranks</span>
                <span>Ring All-Reduce</span>
              </div>
            </div>
            <div className="scene-wrap">
              <div className="scene-event-label" aria-hidden="true">
                <span>{activeEvent.layer}</span>
                <strong>{activeEvent.compactTitle}</strong>
              </div>
              <ErrorBoundary>
                <Suspense fallback={<SceneLoading />}>
                  <SceneCanvas
                    event={activeEvent}
                    reducedMotion={reducedMotion}
                  />
                </Suspense>
              </ErrorBoundary>
            </div>
            <EventTimeline />
          </div>

          <Inspector />
        </section>

        <LearningPaths />

        <section className="project-note">
          <div>
            <h2>Built as a simulation, not a video</h2>
            <p>
              Every visual is derived from structured events, so new models and
              systems can reuse the same timeline, inspector, and camera.
            </p>
          </div>
          <a
            href="https://github.com/TobinZuo/AI-Systems-Atlas"
            target="_blank"
            rel="noreferrer"
          >
            Explore the source
            <ArrowRight size={17} aria-hidden="true" />
          </a>
        </section>
      </main>

      <footer>
        <span>AI Systems Atlas</span>
        <span>Models, kernels, memory, and clusters.</span>
      </footer>
    </div>
  );
}

export default App;
