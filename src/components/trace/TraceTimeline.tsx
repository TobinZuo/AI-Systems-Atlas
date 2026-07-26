import type { TraceCategory, TraceDataset, TraceEvent } from "../../domain/trace";

const categories: Array<{ id: TraceCategory | "all"; label: string }> = [
  { id: "all", label: "全部" },
  { id: "framework", label: "Framework" },
  { id: "cpu", label: "CPU" },
  { id: "compute", label: "Compute" },
  { id: "collective", label: "Collective" },
  { id: "network", label: "Network" },
  { id: "memory", label: "Memory" },
];

export function TraceCategoryFilter({ value, onChange }: { value: TraceCategory | "all"; onChange: (value: TraceCategory | "all") => void }) {
  return (
    <div className="trace-category-filter" aria-label="事件类型筛选">
      {categories.map((category) => <button type="button" className={`filter-${category.id}${value === category.id ? " is-active" : ""}`} onClick={() => onChange(category.id)} key={category.id}>{category.label}</button>)}
    </div>
  );
}

export function TraceTimeline({
  dataset,
  selectedEventId,
  query,
  category,
  zoom,
  onSelect,
}: {
  dataset: TraceDataset;
  selectedEventId: string | null;
  query: string;
  category: TraceCategory | "all";
  zoom: number;
  onSelect: (event: TraceEvent) => void;
}) {
  const normalizedQuery = query.trim().toLowerCase();
  const visibleEvents = dataset.events.filter((event) => {
    if (category !== "all" && event.category !== category) return false;
    if (!normalizedQuery) return true;
    const lane = dataset.lanes.find((item) => item.id === event.laneId);
    return `${event.name} ${lane?.label ?? ""} ${lane?.group ?? ""}`.toLowerCase().includes(normalizedQuery);
  });
  const eventsByLane = new Map<string, TraceEvent[]>();
  for (const event of visibleEvents) {
    const laneEvents = eventsByLane.get(event.laneId) ?? [];
    laneEvents.push(event);
    eventsByLane.set(event.laneId, laneEvents);
  }
  const canvasWidth = Math.round(940 * zoom);
  const selected = dataset.events.find((event) => event.id === selectedEventId);

  return (
    <section className="trace-timeline" aria-label="System trace timeline">
      <div className="trace-lane-list" aria-hidden="true">
        <div className="trace-lane-head"><strong>执行轨道</strong><small>{dataset.lanes.length} lanes</small></div>
        {dataset.lanes.map((lane, index) => {
          const groupChanged = index === 0 || dataset.lanes[index - 1].group !== lane.group;
          return <div className={`trace-lane-label${groupChanged ? " starts-group" : ""}`} key={lane.id}><span>{lane.group}</span><strong>{lane.label}</strong><small>{lane.detail}</small></div>;
        })}
      </div>

      <div className="trace-horizontal-scroll">
        <div className="trace-canvas" style={{ width: canvasWidth }}>
          <div className="trace-ruler">
            {Array.from({ length: 11 }, (_, index) => {
              const time = dataset.totalDuration * index / 10;
              return <span key={index} style={{ left: `${index * 10}%` }}><i />{time.toFixed(time < 10 ? 1 : 0)} ms</span>;
            })}
          </div>
          <div className="trace-track-stack">
            {selected && <span className="trace-time-cursor" style={{ left: `${selected.start / dataset.totalDuration * 100}%` }} />}
            {dataset.lanes.map((lane, index) => {
              const groupChanged = index === 0 || dataset.lanes[index - 1].group !== lane.group;
              return (
                <div className={`trace-track${groupChanged ? " starts-group" : ""}`} key={lane.id}>
                  {(eventsByLane.get(lane.id) ?? []).map((event) => {
                    const left = event.start / dataset.totalDuration * 100;
                    const width = Math.max(event.duration / dataset.totalDuration * 100, 0.22);
                    return (
                      <button
                        type="button"
                        className={`trace-slice category-${event.category}${selectedEventId === event.id ? " is-selected" : ""}`}
                        style={{ left: `${left}%`, width: `${width}%` }}
                        onClick={() => onSelect(event)}
                        title={`${event.name} · ${event.duration.toFixed(3)} ms`}
                        key={event.id}
                      >
                        <span>{event.name}</span><small>{event.duration.toFixed(2)} ms</small>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {!visibleEvents.length && <div className="trace-empty"><strong>没有匹配事件</strong><span>清除搜索词或切换事件类型。</span></div>}
    </section>
  );
}
