import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowsLeftRight } from "@phosphor-icons/react/ArrowsLeftRight";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { LinkSimple } from "@phosphor-icons/react/LinkSimple";
import { Network } from "@phosphor-icons/react/Network";
import { Pause } from "@phosphor-icons/react/Pause";
import { Play } from "@phosphor-icons/react/Play";
import { Queue } from "@phosphor-icons/react/Queue";
import { Stack } from "@phosphor-icons/react/Stack";
import { Timer } from "@phosphor-icons/react/Timer";
import { Warning } from "@phosphor-icons/react/Warning";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  activeCudaStreamEvents,
  bucketSnapshotsAt,
  nearestBoundaryIndex,
  simulateCudaStreams,
  type BucketSnapshot,
  type CudaStreamEvent,
  type StreamExecutionMode,
  type StreamLaneId,
} from "../playground/cudaStreams";

const modes: Array<{ id: StreamExecutionMode; label: string; description: string }> = [
  { id: "single", label: "单 Stream 串行", description: "顺序正确，但通信挡住后续 backward" },
  { id: "dual-safe", label: "双 Stream + Event", description: "安全重叠，是推荐观察模式" },
  { id: "dual-unsafe", label: "双 Stream 无 Event", description: "产生真实的数据依赖竞态" },
];

const laneMeta: Record<StreamLaneId, { label: string; detail: string }> = {
  cpu: { label: "CPU Thread", detail: "只负责发起和入队" },
  compute: { label: "Compute Stream", detail: "backward 与 optimizer 队列" },
  comm: { label: "Comm Stream", detail: "NCCL collective 队列" },
};

const stateLabels: Record<BucketSnapshot["state"], string> = {
  empty: "空",
  computing: "写入中",
  "local-ready": "本地就绪",
  syncing: "同步中",
  race: "读写竞态",
  synchronized: "同步完成",
  invalid: "结果失效",
};

function eventStyle(event: CudaStreamEvent, totalDuration: number): CSSProperties {
  return {
    "--event-left": `${(event.start / totalDuration) * 100}%`,
    "--event-width": `${(event.duration / totalDuration) * 100}%`,
  } as CSSProperties;
}

function eventKindLabel(event: CudaStreamEvent): string {
  const labels: Record<CudaStreamEvent["kind"], string> = {
    host: "CPU API",
    compute: "Compute kernel",
    "event-record": "Event record",
    "event-wait": "Stream wait",
    collective: "NCCL kernel",
    optimizer: "Optimizer kernel",
  };
  return labels[event.kind];
}

function activeSummary(events: CudaStreamEvent[], laneId: StreamLaneId): string {
  const active = events.filter((event) => event.laneId === laneId);
  if (!active.length) return laneId === "cpu" ? "CPU 可继续准备下一批工作" : "此刻没有执行任务";
  return active.map((event) => event.compactLabel).join(" + ");
}

export function CUDAStreamsPlayground() {
  const [mode, setMode] = useState<StreamExecutionMode>("dual-safe");
  const [bucketCount, setBucketCount] = useState(3);
  const [communicationDuration, setCommunicationDuration] = useState(2.4);
  const [time, setTime] = useState(0);
  const [selectedEventId, setSelectedEventId] = useState("backward-b2");
  const [selectedBucketId, setSelectedBucketId] = useState(2);
  const [isPlaying, setIsPlaying] = useState(false);

  const simulation = useMemo(
    () => simulateCudaStreams({ mode, bucketCount, communicationDuration }),
    [mode, bucketCount, communicationDuration],
  );
  const snapshots = bucketSnapshotsAt(simulation, time);
  const activeEvents = activeCudaStreamEvents(simulation, time);
  const selectedEvent = simulation.events.find((event) => event.id === selectedEventId) ?? simulation.events[0];
  const selectedBucket = snapshots.find((bucket) => bucket.bucketId === selectedBucketId) ?? snapshots[0];
  const currentBoundaryIndex = nearestBoundaryIndex(simulation, time);
  const timelineStyle = { "--cursor-scale": time / simulation.totalDuration } as CSSProperties;

  useEffect(() => {
    setTime(0);
    setIsPlaying(false);
    setSelectedBucketId((current) => Math.min(current, bucketCount - 1));
    setSelectedEventId(`backward-b${bucketCount - 1}`);
  }, [mode, bucketCount, communicationDuration]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setTime((current) => {
        const boundaryIndex = nearestBoundaryIndex(simulation, current);
        const currentBoundary = simulation.boundaries[boundaryIndex];
        const nextIndex = current < currentBoundary - 0.001 ? boundaryIndex : boundaryIndex + 1;
        if (nextIndex >= simulation.boundaries.length) {
          setIsPlaying(false);
          return simulation.totalDuration;
        }
        return simulation.boundaries[nextIndex];
      });
    }, 850);
    return () => window.clearInterval(timer);
  }, [isPlaying, simulation]);

  const moveBoundary = (direction: -1 | 1) => {
    const boundary = simulation.boundaries[currentBoundaryIndex];
    let nextIndex = currentBoundaryIndex + direction;
    if (direction > 0 && time < boundary - 0.001) nextIndex = currentBoundaryIndex;
    if (direction < 0 && time > boundary + 0.001) nextIndex = currentBoundaryIndex;
    setTime(simulation.boundaries[Math.max(0, Math.min(simulation.boundaries.length - 1, nextIndex))]);
    setIsPlaying(false);
  };

  const reset = () => {
    setMode("dual-safe");
    setBucketCount(3);
    setCommunicationDuration(2.4);
    setTime(0);
    setSelectedBucketId(2);
    setSelectedEventId("backward-b2");
    setIsPlaying(false);
  };

  const selectBucket = (bucketId: number) => {
    setSelectedBucketId(bucketId);
    const collective = simulation.events.find((event) => event.id === `allreduce-b${bucketId}`);
    const producer = simulation.events.find((event) => event.id === `backward-b${bucketId}`);
    const nextEvent = time >= (collective?.start ?? Number.POSITIVE_INFINITY) ? collective : producer;
    if (nextEvent) setSelectedEventId(nextEvent.id);
  };

  return (
    <section className="cuda-stream-lab" id="cuda-stream-lab" aria-labelledby="stream-lab-title">
      <header className="stream-lab-header">
        <div>
          <span>Gradient bucket scheduling lab</span>
          <h2 id="stream-lab-title">拖动时间，看队列、依赖和数据状态一起变化</h2>
          <p>三个模式使用相同的 backward 与 AllReduce 工作量。只有 Stream 分配和 Event 依赖不同。</p>
        </div>
        <div className="stream-core-equation" aria-label="核心关系">
          <code>Stream = ordered work queue</code>
          <span>Event = cross-stream happens-before edge</span>
        </div>
      </header>

      <div className="stream-mode-tabs" role="tablist" aria-label="调度模式">
        {modes.map((item) => (
          <button type="button" role="tab" aria-selected={mode === item.id} className={mode === item.id ? "is-active" : ""} onClick={() => setMode(item.id)} key={item.id}>
            {item.id === "dual-unsafe" ? <Warning size={18} weight="duotone" aria-hidden="true" /> : item.id === "single" ? <Queue size={18} weight="duotone" aria-hidden="true" /> : <LinkSimple size={18} weight="duotone" aria-hidden="true" />}
            <span><strong>{item.label}</strong><small>{item.description}</small></span>
          </button>
        ))}
      </div>

      <div className="stream-controls">
        <label><span>Gradient buckets</span><select value={bucketCount} onChange={(event) => setBucketCount(Number(event.target.value))}>{[2, 3, 4].map((value) => <option value={value} key={value}>{value} buckets</option>)}</select></label>
        <label><span>每个 AllReduce</span><select value={communicationDuration} onChange={(event) => setCommunicationDuration(Number(event.target.value))}>{[1.6, 2.4, 3.2].map((value) => <option value={value} key={value}>{value.toFixed(1)} ms</option>)}</select></label>
        <div className="stream-playback-actions">
          <button type="button" className="stream-play-button" onClick={() => {
            if (!isPlaying && time >= simulation.totalDuration) setTime(0);
            setIsPlaying((current) => !current);
          }}>
            {isPlaying ? <Pause size={16} weight="fill" aria-hidden="true" /> : <Play size={16} weight="fill" aria-hidden="true" />}
            {isPlaying ? "暂停" : time >= simulation.totalDuration ? "重放" : "按事件播放"}
          </button>
          <button type="button" onClick={reset} aria-label="重置 CUDA Stream 实验"><ArrowCounterClockwise size={17} aria-hidden="true" /></button>
        </div>
      </div>

      <div className="stream-metrics" aria-label="调度结果">
        <div><span>训练片段耗时</span><strong>{simulation.totalDuration.toFixed(2)} ms</strong><small>串行基线 {simulation.serialDuration.toFixed(2)} ms</small></div>
        <div><span>计算通信重叠</span><strong>{simulation.overlapDuration.toFixed(2)} ms</strong><small>{(simulation.overlapRatio * 100).toFixed(0)}% 通信被隐藏</small></div>
        <div><span>{simulation.safe ? "相对串行速度" : "表面速度"}</span><strong>{simulation.speedup.toFixed(2)}×</strong><small>{simulation.safe ? "教学调度模型，不代表真实 GPU 吞吐" : "结果已损坏，这个速度没有训练价值"}</small></div>
        <div className={simulation.safe ? "is-safe" : "is-hazard"}><span>数据正确性</span><strong>{simulation.safe ? "依赖完整" : `${simulation.hazardBucketIds.length} 个 bucket 竞态`}</strong><small>{simulation.safe ? "读取发生在 producer 完成后" : "最终等待也无法补救错误读取"}</small></div>
      </div>

      <section className="stream-timeline-section" aria-labelledby="stream-timeline-title">
        <header>
          <div><span>Queue timeline</span><h3 id="stream-timeline-title">谁在什么时候排队，谁在什么时候真正执行</h3></div>
          <div className="stream-time-readout"><Timer size={16} aria-hidden="true" /><strong>{time.toFixed(2)} ms</strong><span>/ {simulation.totalDuration.toFixed(2)} ms</span></div>
        </header>

        <div className="stream-timeline" style={timelineStyle}>
          <div className="stream-ruler" aria-hidden="true">
            {Array.from({ length: 6 }, (_, index) => <span style={{ left: `${index * 20}%` }} key={index}>{(simulation.totalDuration * index / 5).toFixed(1)}</span>)}
          </div>
          {(["cpu", "compute", "comm"] as StreamLaneId[]).map((laneId) => {
            const laneEvents = simulation.events.filter((event) => event.laneId === laneId);
            return (
              <div className={`stream-timeline-lane lane-${laneId}`} key={laneId}>
                <div className="stream-lane-label"><strong>{laneMeta[laneId].label}</strong><span>{laneMeta[laneId].detail}</span></div>
                <div className="stream-lane-track">
                  {laneEvents.map((event) => (
                    <button
                      type="button"
                      className={`stream-event event-${event.kind}${event.hazard ? " is-hazard" : ""}${selectedEvent.id === event.id ? " is-selected" : ""}${event.duration === 0 ? " is-marker" : ""}`}
                      style={eventStyle(event, simulation.totalDuration)}
                      title={`${event.label}, ${event.start.toFixed(2)} ms`}
                      onClick={() => {
                        setSelectedEventId(event.id);
                        if (event.bucketId !== undefined) setSelectedBucketId(event.bucketId);
                        setTime(event.start);
                        setIsPlaying(false);
                      }}
                      key={event.id}
                    >
                      <span>{event.compactLabel}</span>
                    </button>
                  ))}
                  {laneId === "comm" && mode === "single" && <div className="stream-empty-lane">没有独立 Comm Stream，AllReduce 在 Compute Stream 串行执行</div>}
                  <i className="stream-lane-cursor" aria-hidden="true" />
                </div>
              </div>
            );
          })}
        </div>

        <div className="stream-scrubber">
          <button type="button" onClick={() => moveBoundary(-1)} disabled={time <= 0} aria-label="上一个时间边界"><CaretLeft size={16} /></button>
          <input type="range" min={0} max={simulation.totalDuration} step={0.01} value={time} onChange={(event) => {
            setTime(Number(event.target.value));
            setIsPlaying(false);
          }} aria-label="拖动 CUDA Stream 时间线" />
          <button type="button" onClick={() => moveBoundary(1)} disabled={time >= simulation.totalDuration} aria-label="下一个时间边界"><CaretRight size={16} /></button>
        </div>

        <div className="stream-now-strip" aria-live="polite">
          {(["cpu", "compute", "comm"] as StreamLaneId[]).map((laneId) => (
            <div className={activeEvents.some((event) => event.laneId === laneId) ? "is-active" : ""} key={laneId}><span>{laneMeta[laneId].label}</span><strong>{activeSummary(activeEvents, laneId)}</strong></div>
          ))}
        </div>
      </section>

      <div className="stream-data-layout">
        <section className="stream-bucket-board" aria-labelledby="stream-buckets-title">
          <header><div><span>HBM data state</span><h3 id="stream-buckets-title">通信传的是 bucket 数据，不是只有地址</h3></div><Database size={22} weight="duotone" aria-hidden="true" /></header>
          <div className="stream-bucket-grid">
            {snapshots.slice().reverse().map((bucket) => (
              <button type="button" className={`state-${bucket.state}${selectedBucket.bucketId === bucket.bucketId ? " is-selected" : ""}`} onClick={() => selectBucket(bucket.bucketId)} key={bucket.bucketId}>
                <span>Bucket {bucket.bucketId}</span><strong>{stateLabels[bucket.state]}</strong><code>{bucket.value}</code><small>{bucket.location}</small>
              </button>
            ))}
          </div>
          <div className={`stream-bucket-inspector state-${selectedBucket.state}`}>
            <span>当前选中 Bucket {selectedBucket.bucketId}</span><strong>{selectedBucket.label}</strong><p>{selectedBucket.state === "race" || selectedBucket.state === "invalid" ? "Comm Stream 已经读取了 producer 尚未写完的显存。之后再等待，只能等错误操作结束，不能恢复正确值。" : "同一块 HBM buffer 的地址保持不变，kernel 与 NCCL 按依赖关系在不同时间读写其中的数据。"}</p>
          </div>
        </section>

        <aside className={`stream-event-inspector${selectedEvent.hazard ? " is-hazard" : ""}`} aria-live="polite">
          <header><span>{eventKindLabel(selectedEvent)}</span><strong>{selectedEvent.label}</strong><code>{selectedEvent.start.toFixed(2)} → {(selectedEvent.start + selectedEvent.duration).toFixed(2)} ms</code></header>
          <p>{selectedEvent.explanation}</p>
          <dl>
            <div><dt>所在队列</dt><dd>{laneMeta[selectedEvent.laneId].label}</dd></div>
            <div><dt>读取</dt><dd>{selectedEvent.reads.join(" · ")}</dd></div>
            <div><dt>写入</dt><dd>{selectedEvent.writes.join(" · ")}</dd></div>
            <div><dt>必须先完成</dt><dd>{selectedEvent.dependencies.length ? selectedEvent.dependencies.join(" · ") : "没有显式依赖"}</dd></div>
          </dl>
          <div className="stream-event-verdict">
            {selectedEvent.hazard ? <Warning size={18} weight="fill" aria-hidden="true" /> : <CheckCircle size={18} weight="fill" aria-hidden="true" />}
            <span><strong>{selectedEvent.hazard ? "这一步读取得太早" : "这一步满足当前依赖"}</strong><small>{selectedEvent.hazard ? "地址有效不代表里面的数据已就绪" : "队列顺序和 Event 共同建立 happens-before"}</small></span>
          </div>
        </aside>
      </div>

      <section className="stream-hardware-cutaway" aria-labelledby="stream-hardware-title">
        <header><span>Logical queues, shared hardware</span><h3 id="stream-hardware-title">Compute Stream 和 Comm Stream 不是两套物理 GPU</h3><p>这两个名字表达队列的用途，并不是 CUDA 的两种特殊 Stream 类型。它们处在同一进程、同一 CUDA context 中，backward kernel 与 NCCL kernel 都可能占用 SM、HBM 带宽和缓存资源。</p></header>
        <div className="stream-hardware-flow">
          <article className="hardware-host"><Cpu size={20} weight="duotone" /><span>Host process</span><strong>CUDA API + NCCL API</strong><small>异步提交工作与显存指针</small></article>
          <ArrowsLeftRight className="stream-flow-arrow" size={18} aria-hidden="true" />
          <div className="stream-queue-pair">
            <article className="compute-queue"><Lightning size={18} weight="duotone" /><span>Compute Stream</span><strong>BW B2 → BW B1 → BW B0</strong></article>
            <i className={mode === "dual-safe" ? "is-active" : mode === "dual-unsafe" ? "is-missing" : ""}><LinkSimple size={15} aria-hidden="true" />{mode === "dual-safe" ? "CUDA Event 建立依赖" : mode === "dual-unsafe" ? "缺少跨队列依赖" : "同一队列天然有序"}</i>
            <article className="comm-queue"><Network size={18} weight="duotone" /><span>Comm Stream</span><strong>{mode === "single" ? "未使用" : "AR B2 → AR B1 → AR B0"}</strong></article>
          </div>
          <ArrowsLeftRight className="stream-flow-arrow" size={18} aria-hidden="true" />
          <article className="hardware-device"><Stack size={20} weight="duotone" /><span>同一块 GPU</span><strong>SM + memory fabric + HBM</strong><small>是否真并行取决于硬件资源与竞争</small></article>
        </div>
        <div className="stream-hardware-note"><Network size={17} weight="duotone" aria-hidden="true" /><p>NCCL kernel 从 HBM 读梯度数据，通过 NVLink、PCIe 或网络发送字节，再把归约结果写回 HBM。API 传入地址只是告诉通信 kernel 去哪里读写。</p></div>
      </section>

      <section className="stream-contracts" aria-labelledby="stream-contracts-title">
        <header><h3 id="stream-contracts-title">四条最重要的系统契约</h3></header>
        <div>
          <article><span>01 · Queue</span><strong>同一 Stream 保持顺序</strong><p>后入队的工作要等前面的工作到达可执行条件。它是正确性的第一层基础。</p></article>
          <article><span>02 · Dependency</span><strong>不同 Stream 默认没有顺序</strong><p>共享同一块显存也不会自动建立依赖。需要 Event 明确表达 producer 与 consumer。</p></article>
          <article><span>03 · Async</span><strong>CPU 入队不等于 GPU 完成</strong><p>Host 可以继续准备后续工作。真正完成时间要看 Stream 进度、Event 与设备执行。</p></article>
          <article><span>04 · Overlap</span><strong>并发机会不等于并发保证</strong><p>多 Stream 提供重叠机会，但 kernel 仍可能争夺 SM、缓存、HBM 和互连带宽。</p></article>
        </div>
      </section>

      <footer className="stream-reference-footer">
        <strong>官方依据</strong>
        <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html" target="_blank" rel="noreferrer">CUDA Streams、Events 与异步执行</a>
        <a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html" target="_blank" rel="noreferrer">NCCL 的 CUDA Stream 语义</a>
        <a href="https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html" target="_blank" rel="noreferrer">PyTorch DDP gradient bucket hooks</a>
      </footer>
    </section>
  );
}
