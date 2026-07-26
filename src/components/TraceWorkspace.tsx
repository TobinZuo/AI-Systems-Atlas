import { FileArrowUp } from "@phosphor-icons/react/FileArrowUp";
import { MagnifyingGlass } from "@phosphor-icons/react/MagnifyingGlass";
import { WarningCircle } from "@phosphor-icons/react/WarningCircle";
import { useRef, useState } from "react";
import type { TraceCategory, TraceDataset, TraceEvent } from "../domain/trace";
import { parseChromeTrace } from "../trace/chromeTrace";
import { ddpSampleTrace } from "../traces/ddpSample";
import { TraceInspector } from "./trace/TraceInspector";
import { TraceCategoryFilter, TraceTimeline } from "./trace/TraceTimeline";

type ImportState = { status: "idle" | "loading" | "success" | "error"; message: string };

export function TraceWorkspace({ onOpenConcept }: { onOpenConcept: (eventId: string) => void }) {
  const [dataset, setDataset] = useState<TraceDataset>(ddpSampleTrace);
  const [selectedEvent, setSelectedEvent] = useState<TraceEvent | null>(() => ddpSampleTrace.events.find((event) => event.id === "rs0-r0") ?? null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<TraceCategory | "all">("all");
  const [zoom, setZoom] = useState(1.4);
  const [importState, setImportState] = useState<ImportState>({ status: "idle", message: "" });
  const fileInput = useRef<HTMLInputElement>(null);

  const loadFile = async (file: File) => {
    if (file.size > 25 * 1024 * 1024) {
      setImportState({ status: "error", message: "当前浏览器版本限制为 25 MB。请先裁剪 Trace 时间范围。" });
      return;
    }
    setImportState({ status: "loading", message: `正在读取 ${file.name}` });
    try {
      const parsed = parseChromeTrace(JSON.parse(await file.text()), file.name);
      setDataset(parsed.dataset);
      setSelectedEvent(parsed.dataset.events[0] ?? null);
      setQuery("");
      setCategory("all");
      setImportState({ status: "success", message: `已导入 ${parsed.importedEvents} 个事件${parsed.skippedEvents ? `，跳过 ${parsed.skippedEvents} 个` : ""}` });
    } catch (error) {
      setImportState({ status: "error", message: error instanceof Error ? error.message : "无法解析这个文件。" });
    } finally {
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  return (
    <section className="trace-workspace" id="trace-workspace" aria-label="Trace analysis workspace">
      <header className="trace-workspace-header">
        <div><span>Trace source</span><strong>{dataset.name}</strong><small>{dataset.description}</small></div>
        <div className="trace-source-facts"><span>{dataset.events.length} events</span><span>{dataset.lanes.length} lanes</span><span>{dataset.totalDuration.toFixed(2)} ms</span></div>
        <input ref={fileInput} className="visually-hidden" type="file" accept=".json,application/json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void loadFile(file); }} />
        <div className="trace-actions">
          <a href={`${import.meta.env.BASE_URL}samples/minimal-pytorch-trace.json`} download>下载示例 JSON</a>
          <button type="button" className="trace-import-button" onClick={() => fileInput.current?.click()} disabled={importState.status === "loading"}><FileArrowUp size={16} />{importState.status === "loading" ? "读取中" : "导入 Trace JSON"}</button>
        </div>
      </header>

      <div className="trace-toolbar">
        <label className="trace-search"><MagnifyingGlass size={15} /><span className="visually-hidden">搜索事件</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索 kernel、NCCL、lane" /></label>
        <TraceCategoryFilter value={category} onChange={setCategory} />
        <label className="trace-zoom"><span>Zoom</span><input type="range" min="1" max="4" step="0.25" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><code>{zoom.toFixed(2)}x</code></label>
      </div>

      {(importState.message || dataset.warnings.length > 0) && <div className={`trace-notice is-${importState.status}`} aria-live="polite"><WarningCircle size={15} /><span>{importState.message || dataset.warnings[0]}</span>{dataset.source === "chrome-trace" && <button type="button" onClick={() => { setDataset(ddpSampleTrace); setSelectedEvent(ddpSampleTrace.events.find((event) => event.id === "rs0-r0") ?? null); setImportState({ status: "idle", message: "" }); }}>恢复内置样例</button>}</div>}

      <div className="trace-main-grid">
        <TraceTimeline dataset={dataset} selectedEventId={selectedEvent?.id ?? null} query={query} category={category} zoom={zoom} onSelect={setSelectedEvent} />
        <TraceInspector dataset={dataset} event={selectedEvent} onOpenConcept={onOpenConcept} onSelect={setSelectedEvent} />
      </div>
    </section>
  );
}
