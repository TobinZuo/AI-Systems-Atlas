import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Clock } from "@phosphor-icons/react/Clock";
import { LinkSimple } from "@phosphor-icons/react/LinkSimple";
import { Stack } from "@phosphor-icons/react/Stack";
import type { TraceDataset, TraceEvent } from "../../domain/trace";

const displayValue = (value: unknown) => {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try { return JSON.stringify(value); } catch { return "[unserializable]"; }
};

export function TraceInspector({ dataset, event, onOpenConcept, onSelect }: { dataset: TraceDataset; event: TraceEvent | null; onOpenConcept: (eventId: string) => void; onSelect: (event: TraceEvent) => void }) {
  if (!event) {
    return <aside className="trace-inspector trace-inspector-empty"><Stack size={24} /><strong>选择一个时间片</strong><p>点击任意 CPU、CUDA、NCCL 或链路事件，查看时间、参数和并发关系。</p></aside>;
  }
  const lane = dataset.lanes.find((item) => item.id === event.laneId);
  const eventEnd = event.start + event.duration;
  const overlapping = dataset.events.filter((candidate) => candidate.id !== event.id && candidate.laneId !== event.laneId && candidate.start < eventEnd && candidate.start + candidate.duration > event.start).slice(0, 6);

  return (
    <aside className="trace-inspector" aria-label="Trace event details">
      <section className="trace-inspector-lead">
        <span className={`trace-type category-${event.category}`}>{event.category}</span>
        <h2>{event.name}</h2>
        <p>{lane?.group} / {lane?.label}</p>
      </section>
      <section className="trace-timing-facts">
        <div><Clock size={15} /><span>Start</span><strong>{event.start.toFixed(3)} ms</strong></div>
        <div><ArrowRight size={15} /><span>Duration</span><strong>{event.duration.toFixed(3)} ms</strong></div>
        <div><Stack size={15} /><span>Lane</span><strong>{lane?.detail}</strong></div>
      </section>
      <section className="trace-arguments">
        <span>Event arguments</span>
        <dl>{Object.entries(event.args).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{displayValue(value)}</dd></div>)}</dl>
      </section>
      <section className="trace-overlap">
        <span>同一时刻还在运行</span>
        {overlapping.length ? <div>{overlapping.map((item) => {
          const overlapLane = dataset.lanes.find((laneItem) => laneItem.id === item.laneId);
          return <button type="button" onClick={() => onSelect(item)} key={item.id}><strong>{item.name}</strong><small>{overlapLane?.group} / {overlapLane?.label}</small></button>;
        })}</div> : <p>没有跨轨道重叠事件。</p>}
      </section>
      {event.conceptEventId && <section className="trace-concept-link"><LinkSimple size={16} /><div><strong>关联教学步骤</strong><span>切回 Concept Mode，查看这个事件的因果解释和具体数值。</span><button type="button" onClick={() => onOpenConcept(event.conceptEventId!)}>打开教学步骤</button></div></section>}
    </aside>
  );
}
