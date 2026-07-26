import { lazy, Suspense, useEffect, useState } from "react";
import { HomePage } from "./components/HomePage";
import { SiteHeader } from "./components/SiteHeader";
import { parseHash, type AtlasRoute } from "./router";

const AutogradTopicPage = lazy(() => import("./components/AutogradTopicPage").then((module) => ({ default: module.AutogradTopicPage })));
const AdamWTopicPage = lazy(() => import("./components/AdamWTopicPage").then((module) => ({ default: module.AdamWTopicPage })));
const CUDAKernelTopicPage = lazy(() => import("./components/CUDAKernelTopicPage").then((module) => ({ default: module.CUDAKernelTopicPage })));
const CUDAStreamsTopicPage = lazy(() => import("./components/CUDAStreamsTopicPage").then((module) => ({ default: module.CUDAStreamsTopicPage })));
const CollectiveTopicPage = lazy(() => import("./components/CollectiveTopicPage").then((module) => ({ default: module.CollectiveTopicPage })));
const DDPTopicPage = lazy(() => import("./components/DDPTopicPage").then((module) => ({ default: module.DDPTopicPage })));
const DistributedComparisonTopicPage = lazy(() => import("./components/DistributedComparisonTopicPage").then((module) => ({ default: module.DistributedComparisonTopicPage })));
const FSDPTopicPage = lazy(() => import("./components/FSDPTopicPage").then((module) => ({ default: module.FSDPTopicPage })));
const GradientTopicPage = lazy(() => import("./components/GradientTopicPage").then((module) => ({ default: module.GradientTopicPage })));
const GPUArchitectureTopicPage = lazy(() => import("./components/GPUArchitectureTopicPage").then((module) => ({ default: module.GPUArchitectureTopicPage })));
const MemoryHierarchyTopicPage = lazy(() => import("./components/MemoryHierarchyTopicPage").then((module) => ({ default: module.MemoryHierarchyTopicPage })));
const ProcessRankTopicPage = lazy(() => import("./components/ProcessRankTopicPage").then((module) => ({ default: module.ProcessRankTopicPage })));
const RingAllReduceTopicPage = lazy(() => import("./components/RingAllReduceTopicPage").then((module) => ({ default: module.RingAllReduceTopicPage })));
const ShardedOptimizerTopicPage = lazy(() => import("./components/ShardedOptimizerTopicPage").then((module) => ({ default: module.ShardedOptimizerTopicPage })));

type Theme = "light" | "dark";

function TopicPageSkeleton() {
  return (
    <main className="topic-page-skeleton" aria-live="polite" aria-label="正在加载专题">
      <div className="topic-skeleton-breadcrumb" />
      <div className="topic-skeleton-grid">
        <div><i /><i /><i /></div>
        <div><i /><i /><i /><i /></div>
      </div>
      <div className="topic-skeleton-workspace"><i /><i /><i /></div>
      <span className="visually-hidden">正在加载交互专题</span>
    </main>
  );
}

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
      {route.kind === "topic" && (
        <Suspense fallback={<TopicPageSkeleton />}>
          {route.topicId === "autograd" && <AutogradTopicPage />}
          {route.topicId === "gradient" && <GradientTopicPage />}
          {route.topicId === "adamw" && <AdamWTopicPage />}
          {route.topicId === "gpu-architecture" && <GPUArchitectureTopicPage />}
          {route.topicId === "cuda-kernel" && <CUDAKernelTopicPage />}
          {route.topicId === "cuda-stream" && <CUDAStreamsTopicPage />}
          {route.topicId === "memory-hierarchy" && <MemoryHierarchyTopicPage />}
          {route.topicId === "process-rank" && <ProcessRankTopicPage />}
          {route.topicId === "collective" && <CollectiveTopicPage />}
          {route.topicId === "ring-allreduce" && <RingAllReduceTopicPage />}
          {route.topicId === "ddp" && <DDPTopicPage />}
          {route.topicId === "zero-1" && <ShardedOptimizerTopicPage />}
          {route.topicId === "fsdp" && <FSDPTopicPage />}
          {route.topicId === "compare" && <DistributedComparisonTopicPage />}
        </Suspense>
      )}
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
