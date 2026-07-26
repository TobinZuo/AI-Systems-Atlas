import {
  ArrowRight,
  BookOpen,
  Cube,
  GithubLogo,
  Moon,
  Play,
  Sun,
} from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { AtlasModeBar, type AtlasMode } from "./components/AtlasModeBar";
import { DetailedInspector } from "./components/DetailedInspector";
import { DetailStage } from "./components/DetailStage";
import { JourneyRail } from "./components/JourneyRail";
import { LearningPaths } from "./components/LearningPaths";
import { TraceWorkspace } from "./components/TraceWorkspace";
import { usePlayback } from "./hooks/usePlayback";
import { ddpScenario } from "./scenarios/ddp";
import { useSimulationStore } from "./store/simulation";

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
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [mode, setModeState] = useState<AtlasMode>("concept");
  const play = useSimulationStore((state) => state.play);
  const pause = useSimulationStore((state) => state.pause);
  const seek = useSimulationStore((state) => state.seek);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("atlas-theme", theme);
  }, [theme]);

  const setMode = (nextMode: AtlasMode) => {
    if (nextMode === "trace") pause();
    setModeState(nextMode);
  };

  const openConceptEvent = (eventId: string) => {
    const event = ddpScenario.events.find((item) => item.id === eventId);
    if (event) seek(event.start + 0.01);
    setModeState("concept");
  };

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
          <a href="#explorer">探索器</a>
          <a href="#learning-paths">学习路径</a>
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
            <p className="eyebrow">Interactive systems debugger · DDP</p>
            <h1>跟着一个梯度，走进 AI 系统底层。</h1>
            <p className="intro-copy">
              用可逆模拟建立直觉，再用多轨 Trace 对照 CPU、CUDA、NCCL、内存和链路的真实时间关系。
            </p>
          </div>
          <button type="button" className="primary-action" onClick={() => mode === "concept" ? play() : setMode("concept")}>
            {mode === "concept" ? <Play size={17} weight="fill" aria-hidden="true" /> : <BookOpen size={17} weight="fill" aria-hidden="true" />}
            {mode === "concept" ? "从 backward 开始" : "返回教学模拟"}
          </button>
        </section>

        <AtlasModeBar mode={mode} onChange={setMode} />

        {mode === "concept" ? (
          <>
            <section className="scenario-contract" aria-label="本次模拟的固定条件">
              <div><span>并行规模</span><strong>4 ranks / 4 GPUs</strong><small>单机 NVLink Ring</small></div>
              <div><span>具体梯度</span><strong>8 × fp32 / rank</strong><small>拆成 C0…C3，每块 2 个数</small></div>
              <div><span>集合通信</span><strong>SUM → ÷ world_size</strong><small>3 轮 Reduce-Scatter + 3 轮 All-Gather</small></div>
              <div><span>教学目标</span><strong>从语义到字节</strong><small>Framework → Runtime → Kernel → Link</small></div>
            </section>
            <section className="explorer-workspace" aria-label="DDP interactive explorer">
              <JourneyRail />
              <DetailStage />
              <DetailedInspector />
            </section>
          </>
        ) : <TraceWorkspace onOpenConcept={openConceptEvent} />}

        <LearningPaths />

        <section className="project-note">
          <div>
            <h2>它不是一段只能播放的视频。</h2>
            <p>
              每个数字、chunk 状态和箭头都来自结构化模拟。你可以暂停、逐事件前进，切换观察层级，并验证同一事件在不同系统层看到的事实。
            </p>
          </div>
          <a
            href="https://github.com/TobinZuo/AI-Systems-Atlas"
            target="_blank"
            rel="noreferrer"
          >
            查看模拟器源码
            <ArrowRight size={17} aria-hidden="true" />
          </a>
        </section>
      </main>

      <footer>
        <span>AI Systems Atlas</span>
        <span>把模型、Kernel、内存与集群串成一条因果链。</span>
      </footer>
    </div>
  );
}

export default App;
