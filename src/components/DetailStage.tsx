import { ArrowsClockwise, Cpu, Cube, GridFour, Target, Wrench } from "@phosphor-icons/react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useReducedMotion } from "motion/react";
import { lessonFor, type DetailView } from "../content/ddpLesson";
import { ddpScenario } from "../scenarios/ddp";
import { eventAtTime } from "../sim/simulator";
import { useSimulationStore } from "../store/simulation";
import { ErrorBoundary } from "./ErrorBoundary";
import { EventTimeline } from "./EventTimeline";
import { SceneLoading } from "./SceneLoading";
import { GpuExecutionView } from "./views/GpuExecutionView";
import { OptimizerView } from "./views/OptimizerView";
import { RankProcessView } from "./views/RankProcessView";
import { RingCollectiveView } from "./views/RingCollectiveView";

const SceneCanvas = lazy(() => import("./SceneCanvas").then((module) => ({ default: module.SceneCanvas })));

const tabs: Array<{ id: DetailView; label: string; icon: typeof Cube }> = [
  { id: "overview", label: "系统总览", icon: Cube },
  { id: "rank", label: "Rank / 进程", icon: Cpu },
  { id: "gpu", label: "GPU / SM / Warp", icon: GridFour },
  { id: "ring", label: "Ring 通信", icon: ArrowsClockwise },
  { id: "optimizer", label: "AdamW", icon: Wrench },
];

export function DetailStage() {
  const currentTime = useSimulationStore((state) => state.currentTime);
  const active = useMemo(() => eventAtTime(ddpScenario.events, currentTime), [currentTime]);
  const recommended = lessonFor(active).view;
  const [view, setView] = useState<DetailView>(recommended);
  const [autoFollow, setAutoFollow] = useState(true);
  const reducedMotion = Boolean(useReducedMotion());

  useEffect(() => {
    if (autoFollow) setView(recommended);
  }, [active.id, autoFollow, recommended]);

  const selectView = (next: DetailView) => {
    setView(next);
    setAutoFollow(false);
  };

  return (
    <div className="stage-column detail-stage">
      <div className="view-toolbar">
        <div className="view-tabs" role="tablist" aria-label="观察层级">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return <button type="button" role="tab" aria-selected={view === tab.id} className={view === tab.id ? "is-active" : ""} onClick={() => selectView(tab.id)} key={tab.id}><Icon size={15} /><span>{tab.label}</span>{recommended === tab.id && <i title="当前事件推荐视图" />}</button>;
          })}
        </div>
        <button type="button" className={autoFollow ? "follow-toggle is-active" : "follow-toggle"} onClick={() => { setAutoFollow(true); setView(recommended); }}><Target size={15} /><span>跟随事件</span></button>
      </div>

      <div className="detail-stage-body" role="tabpanel">
        {view === "overview" && <div className="overview-stage"><div className="scene-event-label"><span>{active.layer}</span><strong>{lessonFor(active).title}</strong></div><ErrorBoundary><Suspense fallback={<SceneLoading />}><SceneCanvas event={active} reducedMotion={reducedMotion} /></Suspense></ErrorBoundary></div>}
        {view === "rank" && <RankProcessView event={active} />}
        {view === "gpu" && <GpuExecutionView event={active} />}
        {view === "ring" && <RingCollectiveView event={active} />}
        {view === "optimizer" && <OptimizerView event={active} />}
      </div>
      <EventTimeline />
    </div>
  );
}
