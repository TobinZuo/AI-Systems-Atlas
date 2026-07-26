import { useEffect, useState } from "react";
import { CUDAKernelTopicPage } from "./components/CUDAKernelTopicPage";
import { CUDAStreamsTopicPage } from "./components/CUDAStreamsTopicPage";
import { DDPTopicPage } from "./components/DDPTopicPage";
import { DistributedComparisonTopicPage } from "./components/DistributedComparisonTopicPage";
import { FSDPTopicPage } from "./components/FSDPTopicPage";
import { GPUArchitectureTopicPage } from "./components/GPUArchitectureTopicPage";
import { HomePage } from "./components/HomePage";
import { ShardedOptimizerTopicPage } from "./components/ShardedOptimizerTopicPage";
import { SiteHeader } from "./components/SiteHeader";
import { parseHash, type AtlasRoute } from "./router";

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
  const [route, setRoute] = useState<AtlasRoute>(() => parseHash(window.location.hash));

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("atlas-theme", theme);
  }, [theme]);

  useEffect(() => {
    const syncRoute = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (route.kind === "home" && route.section === "paths") {
        document.getElementById("learning-paths")?.scrollIntoView();
      } else {
        window.scrollTo({ top: 0 });
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [route]);

  const current = route.kind === "topic"
    ? route.topicId
    : route.kind === "home"
      ? route.section
      : "other";

  return (
    <div className="app-shell">
      <SiteHeader
        theme={theme}
        current={current}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      />

      {route.kind === "home" && <HomePage />}
      {route.kind === "topic" && route.topicId === "gpu-architecture" && <GPUArchitectureTopicPage />}
      {route.kind === "topic" && route.topicId === "cuda-kernel" && <CUDAKernelTopicPage />}
      {route.kind === "topic" && route.topicId === "cuda-stream" && <CUDAStreamsTopicPage />}
      {route.kind === "topic" && route.topicId === "ddp" && <DDPTopicPage />}
      {route.kind === "topic" && route.topicId === "zero-1" && <ShardedOptimizerTopicPage />}
      {route.kind === "topic" && route.topicId === "fsdp" && <FSDPTopicPage />}
      {route.kind === "topic" && route.topicId === "compare" && <DistributedComparisonTopicPage />}
      {route.kind === "not-found" && (
        <main className="not-found" id="top">
          <p className="eyebrow">Topic not found</p>
          <h1>这个专题还没有开放。</h1>
          <p>{route.path} 已经可以加入知识地图，但目前没有独立页面。</p>
          <a className="primary-action" href="#/">返回知识地图</a>
        </main>
      )}

      <footer>
        <span>AI Systems Atlas</span>
        <span>把数据、模型、Kernel、内存与集群串成一条因果链。</span>
      </footer>
    </div>
  );
}

export default App;
