export type StreamExecutionMode = "single" | "dual-safe" | "dual-unsafe";
export type StreamLaneId = "cpu" | "compute" | "comm";
export type StreamEventKind =
  | "host"
  | "compute"
  | "event-record"
  | "event-wait"
  | "collective"
  | "optimizer";

export interface CudaStreamConfig {
  mode: StreamExecutionMode;
  bucketCount: number;
  communicationDuration: number;
}

export interface CudaStreamEvent {
  id: string;
  label: string;
  compactLabel: string;
  laneId: StreamLaneId;
  kind: StreamEventKind;
  start: number;
  duration: number;
  bucketId?: number;
  dependencies: string[];
  reads: string[];
  writes: string[];
  explanation: string;
  hazard: boolean;
}

export type BucketState =
  | "empty"
  | "computing"
  | "local-ready"
  | "syncing"
  | "race"
  | "synchronized"
  | "invalid";

export interface BucketSnapshot {
  bucketId: number;
  state: BucketState;
  label: string;
  value: string;
  location: string;
}

export interface CudaStreamSimulation {
  config: CudaStreamConfig;
  events: CudaStreamEvent[];
  boundaries: number[];
  totalDuration: number;
  serialDuration: number;
  overlapDuration: number;
  overlapRatio: number;
  speedup: number;
  safe: boolean;
  hazardBucketIds: number[];
  computeFinishedAt: number;
  communicationFinishedAt: number;
}

const GPU_START = 0.8;
const OPTIMIZER_DURATION = 1.6;

const rounded = (value: number) => Number(value.toFixed(3));

function assertConfig(config: CudaStreamConfig): void {
  if (!Number.isInteger(config.bucketCount) || config.bucketCount < 2 || config.bucketCount > 4) {
    throw new RangeError("bucketCount must be an integer from 2 to 4");
  }
  if (!Number.isFinite(config.communicationDuration) || config.communicationDuration <= 0) {
    throw new RangeError("communicationDuration must be positive");
  }
}

function backwardDuration(order: number): number {
  return rounded(3.1 - order * 0.35);
}

function hostEvents(bucketCount: number, mode: StreamExecutionMode): CudaStreamEvent[] {
  const events: CudaStreamEvent[] = [
    {
      id: "loss-backward",
      label: "Python 调用 loss.backward()",
      compactLabel: "loss.backward()",
      laneId: "cpu",
      kind: "host",
      start: 0,
      duration: 0.42,
      dependencies: [],
      reads: ["loss tensor"],
      writes: ["Autograd work queue"],
      explanation: "CPU 进入 Autograd Engine。调用会持续提交 GPU 工作，但 CPU 不会亲自逐元素计算梯度。",
      hazard: false,
    },
  ];

  for (let order = 0; order < bucketCount; order += 1) {
    const bucketId = bucketCount - 1 - order;
    const enqueueStart = 0.46 + order * 0.58;
    events.push({
      id: `enqueue-backward-b${bucketId}`,
      label: `提交 Bucket ${bucketId} 的 backward kernel`,
      compactLabel: `enqueue BW B${bucketId}`,
      laneId: "cpu",
      kind: "host",
      start: enqueueStart,
      duration: 0.2,
      bucketId,
      dependencies: order === 0 ? ["loss-backward"] : [`enqueue-collective-b${bucketId + 1}`],
      reads: ["Autograd node", "CUDA tensor pointers"],
      writes: ["Compute Stream queue"],
      explanation: "CPU 把 kernel 和显存指针加入 Compute Stream。入队完成不等于 GPU 已经计算完成。",
      hazard: false,
    });

    events.push({
      id: `enqueue-collective-b${bucketId}`,
      label: mode === "dual-safe"
        ? `提交 Event 依赖与 Bucket ${bucketId} AllReduce`
        : mode === "dual-unsafe"
          ? `直接提交 Bucket ${bucketId} AllReduce`
          : `把 Bucket ${bucketId} AllReduce 排进同一 Stream`,
      compactLabel: mode === "dual-safe" ? `queue wait E${bucketId}` : `queue AR B${bucketId}`,
      laneId: "cpu",
      kind: "host",
      start: enqueueStart + 0.24,
      duration: 0.2,
      bucketId,
      dependencies: [`enqueue-backward-b${bucketId}`],
      reads: [mode === "dual-safe" ? `CUDA Event E${bucketId}` : `bucket ${bucketId} pointer`],
      writes: [mode === "single" ? "Compute Stream queue" : "Comm Stream queue"],
      explanation: mode === "dual-safe"
        ? "CPU 建立跨 Stream 依赖。Comm Stream 会在 GPU 上等待 Event，不需要阻塞 CPU。"
        : mode === "dual-unsafe"
          ? "没有 Event 时，Comm Stream 只看到一个地址，不知道 producer kernel 是否已经写完。"
          : "Collective 排在 producer 后面。两者处于同一 Stream，因此不需要额外 Event。",
      hazard: false,
    });
  }
  return events;
}

function intersectionDuration(
  left: Array<{ start: number; duration: number }>,
  right: Array<{ start: number; duration: number }>,
): number {
  let overlap = 0;
  for (const a of left) {
    for (const b of right) {
      overlap += Math.max(0, Math.min(a.start + a.duration, b.start + b.duration) - Math.max(a.start, b.start));
    }
  }
  return rounded(overlap);
}

function buildEvents(config: CudaStreamConfig): {
  events: CudaStreamEvent[];
  computeFinishedAt: number;
  communicationFinishedAt: number;
  hazardBucketIds: number[];
} {
  const events = hostEvents(config.bucketCount, config.mode);
  const producerWindows = new Map<number, { start: number; end: number; eventId: string }>();
  const hazardBucketIds: number[] = [];
  let computeCursor = GPU_START;
  let communicationCursor = GPU_START;

  for (let order = 0; order < config.bucketCount; order += 1) {
    const bucketId = config.bucketCount - 1 - order;
    const computeStart = computeCursor;
    const computeDuration = backwardDuration(order);
    const computeEnd = rounded(computeStart + computeDuration);
    const computeEventId = `backward-b${bucketId}`;

    events.push({
      id: computeEventId,
      label: `Backward kernel 生成 Bucket ${bucketId}`,
      compactLabel: `BW B${bucketId}`,
      laneId: "compute",
      kind: "compute",
      start: computeStart,
      duration: computeDuration,
      bucketId,
      dependencies: order === 0 ? ["enqueue-backward-b" + bucketId] : [`backward-b${bucketId + 1}`],
      reads: ["activation", "upstream gradient", "parameter"],
      writes: [`bucket ${bucketId} local gradient`],
      explanation: "GPU SM 执行 backward kernel，并把这一组参数的本地梯度写入 HBM bucket。",
      hazard: false,
    });
    producerWindows.set(bucketId, { start: computeStart, end: computeEnd, eventId: computeEventId });

    if (config.mode === "single") {
      const collectiveStart = computeEnd;
      events.push({
        id: `allreduce-b${bucketId}`,
        label: `同一 Stream 执行 Bucket ${bucketId} AllReduce`,
        compactLabel: `AR B${bucketId}`,
        laneId: "compute",
        kind: "collective",
        start: collectiveStart,
        duration: config.communicationDuration,
        bucketId,
        dependencies: [computeEventId, `enqueue-collective-b${bucketId}`],
        reads: [`bucket ${bucketId} local gradient`],
        writes: [`bucket ${bucketId} synchronized gradient`],
        explanation: "Collective 和 backward 在同一 Stream，队列顺序天然保证正确，但下一段计算必须等待通信。",
        hazard: false,
      });
      computeCursor = rounded(collectiveStart + config.communicationDuration);
      communicationCursor = computeCursor;
    } else {
      computeCursor = computeEnd;
    }
  }

  const computeFinishedAt = computeCursor;

  if (config.mode !== "single") {
    communicationCursor = GPU_START;
    for (let order = 0; order < config.bucketCount; order += 1) {
      const bucketId = config.bucketCount - 1 - order;
      const producer = producerWindows.get(bucketId)!;
      const hostEnqueueTime = rounded(0.9 + order * 0.58);
      const waitStart = Math.max(communicationCursor, hostEnqueueTime);
      let collectiveStart = waitStart;

      if (config.mode === "dual-safe") {
        events.push({
          id: `record-e${bucketId}`,
          label: `Compute Stream 记录 Event E${bucketId}`,
          compactLabel: `record E${bucketId}`,
          laneId: "compute",
          kind: "event-record",
          start: producer.end,
          duration: 0,
          bucketId,
          dependencies: [producer.eventId, `enqueue-collective-b${bucketId}`],
          reads: [`Compute Stream progress`],
          writes: [`CUDA Event E${bucketId}`],
          explanation: "Event 排在 producer kernel 后面。只有前面的 kernel 完成，Event 才会变成 completed。",
          hazard: false,
        });
        collectiveStart = Math.max(waitStart, producer.end);
        events.push({
          id: `wait-e${bucketId}`,
          label: `Comm Stream 等待 Event E${bucketId}`,
          compactLabel: `wait E${bucketId}`,
          laneId: "comm",
          kind: "event-wait",
          start: waitStart,
          duration: rounded(collectiveStart - waitStart),
          bucketId,
          dependencies: [`record-e${bucketId}`, `enqueue-collective-b${bucketId}`],
          reads: [`CUDA Event E${bucketId}`],
          writes: ["Comm Stream ordering dependency"],
          explanation: "等待发生在 GPU 队列里。CPU 可以继续提交其他任务，Comm Stream 只阻塞 Event 之后的工作。",
          hazard: false,
        });
      }

      const hazard = collectiveStart < producer.end;
      if (hazard) hazardBucketIds.push(bucketId);
      events.push({
        id: `allreduce-b${bucketId}`,
        label: `Comm Stream 执行 Bucket ${bucketId} AllReduce`,
        compactLabel: `AR B${bucketId}`,
        laneId: "comm",
        kind: "collective",
        start: rounded(collectiveStart),
        duration: config.communicationDuration,
        bucketId,
        dependencies: config.mode === "dual-safe" ? [`wait-e${bucketId}`] : [`enqueue-collective-b${bucketId}`],
        reads: [`bucket ${bucketId} local gradient`],
        writes: [`bucket ${bucketId} synchronized gradient`],
        explanation: hazard
          ? "Collective 在 producer 完成前读取 bucket。这是跨 Stream 缺少依赖导致的数据竞态。"
          : "NCCL kernel 在 Comm Stream 上读取 bucket，并通过 GPU 互连归约梯度。",
        hazard,
      });
      communicationCursor = rounded(collectiveStart + config.communicationDuration);
    }
  }

  const communicationFinishedAt = communicationCursor;
  const backwardFinishedAt = config.mode === "single"
    ? communicationFinishedAt
    : Math.max(...[...producerWindows.values()].map((window) => window.end));
  const optimizerStart = Math.max(backwardFinishedAt, communicationFinishedAt);

  if (config.mode !== "single" && communicationFinishedAt > backwardFinishedAt) {
    events.push({
      id: "compute-waits-gradients",
      label: "Compute Stream 等待全部 AllReduce 完成",
      compactLabel: "wait gradients",
      laneId: "compute",
      kind: "event-wait",
      start: backwardFinishedAt,
      duration: rounded(communicationFinishedAt - backwardFinishedAt),
      dependencies: Array.from({ length: config.bucketCount }, (_, bucketId) => `allreduce-b${bucketId}`),
      reads: ["Comm Stream completion Event"],
      writes: ["optimizer ordering dependency"],
      explanation: "optimizer.step() 必须等同步梯度可见。这里保留最终依赖，以单独观察 producer 侧 Event 缺失造成的竞态。",
      hazard: false,
    });
  }

  events.push({
    id: "optimizer-step",
    label: hazardBucketIds.length ? "AdamW 读取可能损坏的梯度" : "AdamW 读取同步梯度",
    compactLabel: "AdamW",
    laneId: "compute",
    kind: "optimizer",
    start: rounded(optimizerStart),
    duration: OPTIMIZER_DURATION,
    dependencies: Array.from({ length: config.bucketCount }, (_, bucketId) => `allreduce-b${bucketId}`),
    reads: ["parameter", "gradient buckets", "m", "v"],
    writes: ["updated parameter", "updated m", "updated v"],
    explanation: hazardBucketIds.length
      ? "最终等待只能保证通信结束，不能修复通信期间已经读取到的未完成数据。"
      : "Compute Stream 在所有梯度同步完成后执行优化器 kernel。",
    hazard: hazardBucketIds.length > 0,
  });

  return {
    events: events.map((event) => ({
      ...event,
      start: rounded(event.start),
      duration: rounded(event.duration),
    })),
    computeFinishedAt: rounded(backwardFinishedAt),
    communicationFinishedAt: rounded(communicationFinishedAt),
    hazardBucketIds,
  };
}

export function simulateCudaStreams(config: CudaStreamConfig): CudaStreamSimulation {
  assertConfig(config);
  const built = buildEvents(config);
  const totalDuration = rounded(
    Math.max(...built.events.map((event) => event.start + event.duration)),
  );
  const computeEvents = built.events.filter((event) => event.kind === "compute");
  const communicationEvents = built.events.filter((event) => event.kind === "collective");
  const overlapDuration = intersectionDuration(computeEvents, communicationEvents);
  const totalCommunication = config.communicationDuration * config.bucketCount;
  const serialDuration = config.mode === "single"
    ? totalDuration
    : simulateCudaStreams({ ...config, mode: "single" }).totalDuration;
  const boundaries = [...new Set(
    built.events.flatMap((event) => [event.start, rounded(event.start + event.duration)]),
  )].sort((a, b) => a - b);

  return {
    config: { ...config },
    events: built.events.sort((a, b) => a.start - b.start || a.laneId.localeCompare(b.laneId)),
    boundaries,
    totalDuration,
    serialDuration,
    overlapDuration,
    overlapRatio: totalCommunication === 0 ? 0 : rounded(overlapDuration / totalCommunication),
    speedup: rounded(serialDuration / totalDuration),
    safe: built.hazardBucketIds.length === 0,
    hazardBucketIds: built.hazardBucketIds,
    computeFinishedAt: built.computeFinishedAt,
    communicationFinishedAt: built.communicationFinishedAt,
  };
}

export function bucketSnapshotsAt(
  simulation: CudaStreamSimulation,
  time: number,
): BucketSnapshot[] {
  return Array.from({ length: simulation.config.bucketCount }, (_, bucketId) => {
    const producer = simulation.events.find((event) => event.id === `backward-b${bucketId}`)!;
    const collective = simulation.events.find((event) => event.id === `allreduce-b${bucketId}`)!;
    const producerEnd = producer.start + producer.duration;
    const collectiveEnd = collective.start + collective.duration;
    let state: BucketState = "empty";

    if (time >= producer.start && time < producerEnd) state = "computing";
    if (time >= producerEnd && time < collective.start) state = "local-ready";
    if (time >= collective.start && time < collectiveEnd) {
      state = collective.hazard ? "race" : "syncing";
    }
    if (time >= collectiveEnd) state = collective.hazard ? "invalid" : "synchronized";

    const copy: Record<BucketState, Omit<BucketSnapshot, "bucketId" | "state">> = {
      empty: { label: "尚未生成", value: "空", location: "HBM bucket buffer" },
      computing: { label: "Backward 写入中", value: "partial local dW", location: "SM → HBM" },
      "local-ready": { label: "本地梯度就绪", value: "complete local dW", location: "HBM" },
      syncing: { label: "AllReduce 中", value: "partial global SUM", location: "HBM + GPU interconnect" },
      race: { label: "发生读写竞态", value: "unfinished local dW", location: "Compute / Comm 同时访问" },
      synchronized: { label: "同步完成", value: "averaged global dW", location: "HBM parameter.grad" },
      invalid: { label: "结果不可信", value: "race-contaminated dW", location: "HBM parameter.grad" },
    };
    return { bucketId, state, ...copy[state] };
  });
}

export function activeCudaStreamEvents(
  simulation: CudaStreamSimulation,
  time: number,
): CudaStreamEvent[] {
  return simulation.events.filter((event) => {
    if (event.duration === 0) return Math.abs(event.start - time) < 0.04;
    return event.start <= time && time < event.start + event.duration;
  });
}

export function nearestBoundaryIndex(simulation: CudaStreamSimulation, time: number): number {
  let nearest = 0;
  let distance = Number.POSITIVE_INFINITY;
  simulation.boundaries.forEach((boundary, index) => {
    const nextDistance = Math.abs(boundary - time);
    if (nextDistance < distance) {
      nearest = index;
      distance = nextDistance;
    }
  });
  return nearest;
}
