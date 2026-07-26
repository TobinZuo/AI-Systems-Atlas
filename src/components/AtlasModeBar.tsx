import { BookOpen, Pulse } from "@phosphor-icons/react";

export type AtlasMode = "concept" | "trace";

export function AtlasModeBar({ mode, onChange }: { mode: AtlasMode; onChange: (mode: AtlasMode) => void }) {
  return (
    <section className="atlas-mode-bar" aria-label="Atlas data mode">
      <div className="mode-choice" role="tablist" aria-label="数据来源">
        <button type="button" role="tab" aria-selected={mode === "concept"} className={mode === "concept" ? "is-active" : ""} onClick={() => onChange("concept")}>
          <BookOpen size={16} /><span>教学模拟</span><small>确定性小数据，可以逐步倒放</small>
        </button>
        <button type="button" role="tab" aria-selected={mode === "trace"} className={mode === "trace" ? "is-active" : ""} onClick={() => onChange("trace")}>
          <Pulse size={16} /><span>Trace 分析</span><small>Profiler 时间线，可以导入 JSON</small>
        </button>
      </div>
      <div className="mode-context">
        <span>当前场景</span>
        <strong>DDP：一个梯度的旅程</strong>
        <small>{mode === "concept" ? "结构化 Scenario" : "14 条系统轨道"}</small>
      </div>
    </section>
  );
}
