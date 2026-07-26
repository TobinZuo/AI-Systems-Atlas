import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Cube } from "@phosphor-icons/react/Cube";
import { GithubLogo } from "@phosphor-icons/react/GithubLogo";
import { Moon } from "@phosphor-icons/react/Moon";
import { SlidersHorizontal } from "@phosphor-icons/react/SlidersHorizontal";
import { Sun } from "@phosphor-icons/react/Sun";
import { useEffect, useState } from "react";
import { AtlasModeBar, type AtlasMode } from "./components/AtlasModeBar";
import { DDPPlayground } from "./components/DDPPlayground";
import { LearningPaths } from "./components/LearningPaths";
import { TraceWorkspace } from "./components/TraceWorkspace";

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  const saved = window.localStorage.getItem("atlas-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [mode, setModeState] = useState<AtlasMode>("concept");
  const [conceptEventId, setConceptEventId] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("atlas-theme", theme);
  }, [theme]);

  const setMode = (nextMode: AtlasMode) => setModeState(nextMode);

  const openConceptEvent = (eventId: string) => {
    setConceptEventId(eventId);
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
            <p className="eyebrow">DDP Playground</p>
            <h1>跟着一个梯度，走进 AI 系统底层。</h1>
            <p className="intro-copy">
              改输入、点轮次，直接观察 GPU、NCCL、显存与链路如何协作。
            </p>
          </div>
          <a className="primary-action" href={mode === "concept" ? "#ddp-playground" : "#trace-workspace"}>
            <SlidersHorizontal size={17} weight="fill" aria-hidden="true" />
            {mode === "concept" ? "开始操作" : "查看时间线"}
          </a>
        </section>

        <AtlasModeBar mode={mode} onChange={setMode} />

        {mode === "concept"
          ? <DDPPlayground focusEventId={conceptEventId} />
          : <TraceWorkspace onOpenConcept={openConceptEvent} />}

        <LearningPaths />

        <section className="project-note">
          <div>
            <h2>不是看动画，是亲手改变系统状态。</h2>
            <p>
              每个数字和 chunk 状态都来自可测试的模拟器。修改任意 rank 的梯度，所有中间结果立即重新计算。
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
