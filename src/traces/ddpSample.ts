import type { TraceDataset, TraceEvent, TraceLane } from "../domain/trace";

const lanes: TraceLane[] = [];
const events: TraceEvent[] = [];

const addEvent = (
  id: string,
  name: string,
  laneId: string,
  start: number,
  duration: number,
  category: TraceEvent["category"],
  args: Record<string, unknown>,
  conceptEventId?: string,
) => events.push({ id, name, laneId, start, duration, category, args, conceptEventId });

for (let rank = 0; rank < 4; rank += 1) {
  const lag = rank * 0.08;
  lanes.push(
    { id: `cpu-r${rank}`, label: `Python / Autograd`, group: `Rank ${rank}`, detail: `CPU process rank ${rank}`, rank },
    { id: `compute-r${rank}`, label: `Compute stream`, group: `GPU ${rank}`, detail: `cudaStream ${rank}:0`, rank },
    { id: `comm-r${rank}`, label: `Communication stream`, group: `GPU ${rank}`, detail: `cudaStream ${rank}:7`, rank },
  );

  addEvent(`backward-r${rank}`, "loss.backward()", `cpu-r${rank}`, 0.15 + lag, 1.05, "framework", {
    rank,
    batch: `local_batch_${rank}`,
    callsite: "train.py:84",
  }, "python-backward");
  addEvent(`autograd-r${rank}`, "Autograd::evaluate_function", `cpu-r${rank}`, 1.28 + lag, 1.15, "cpu", {
    rank,
    node: "MmBackward0",
  }, "autograd-schedules");
  addEvent(`hook-r${rank}`, "Reducer::mark_bucket_ready", `cpu-r${rank}`, 5.38 + lag, 0.34, "framework", {
    rank,
    bucket: 0,
    bytes: 32,
  }, "bucket-ready");
  addEvent(`step-r${rank}`, "optimizer.step()", `cpu-r${rank}`, 12.05 + lag, 2.28, "framework", {
    rank,
    optimizer: "AdamW",
  }, "optimizer-update");

  addEvent(`gemm0-r${rank}`, "gemm_backward.weight", `compute-r${rank}`, 1.55 + lag, 2.08 + rank * 0.03, "compute", {
    rank,
    grid: "128 blocks",
    block: "256 threads",
    output: "weight.grad",
  }, "gradient-kernel");
  addEvent(`gemm1-r${rank}`, "gemm_backward.proj", `compute-r${rank}`, 3.78 + lag, 1.72 + rank * 0.04, "compute", {
    rank,
    grid: "64 blocks",
    stream: "compute",
  }, "gradient-kernel");
  addEvent(`tail-r${rank}`, "next_layer_backward", `compute-r${rank}`, 5.82 + lag, 1.62, "compute", {
    rank,
    overlap: "runs while bucket 0 communicates",
  });
  addEvent(`adamw-r${rank}`, "multi_tensor_adamw", `compute-r${rank}`, 12.37 + lag, 1.64, "compute", {
    rank,
    tensors: 6,
    lr: 0.001,
  }, "optimizer-update");

  addEvent(`wait-r${rank}`, "cudaStreamWaitEvent(E0)", `comm-r${rank}`, 5.52 + lag, 0.22, "cpu", {
    rank,
    waitsFor: "bucket 0 compute completion",
  }, "bucket-ready");
  for (let round = 0; round < 3; round += 1) {
    addEvent(`rs${round}-r${rank}`, `nccl ReduceScatter round ${round + 1}`, `comm-r${rank}`, 6.02 + round * 0.72, 0.62, "collective", {
      rank,
      round: round + 1,
      sendChunk: ((rank - round) % 4 + 4) % 4,
      peer: (rank + 1) % 4,
      op: "SUM",
    }, `reduce-scatter-${round}`);
  }
  for (let round = 0; round < 3; round += 1) {
    addEvent(`ag${round}-r${rank}`, `nccl AllGather round ${round + 1}`, `comm-r${rank}`, 8.28 + round * 0.63, 0.54, "collective", {
      rank,
      round: round + 1,
      sendChunk: ((rank + 1 - round) % 4 + 4) % 4,
      peer: (rank + 1) % 4,
    }, `all-gather-${round}`);
  }
}

lanes.push(
  { id: "fabric", label: "NVLink fabric", group: "Interconnect", detail: "Four unidirectional P2P edges" },
  { id: "hbm", label: "HBM read / write", group: "Memory", detail: "Aggregated device-memory activity" },
);

for (let round = 0; round < 3; round += 1) {
  addEvent(`fabric-rs-${round}`, `4 parallel chunk transfers`, "fabric", 6.07 + round * 0.72, 0.52, "network", {
    phase: "Reduce-Scatter",
    round: round + 1,
    links: "R0→R1, R1→R2, R2→R3, R3→R0",
    payload: "4 × 8 bytes",
  }, `reduce-scatter-${round}`);
  addEvent(`fabric-ag-${round}`, `4 parallel chunk transfers`, "fabric", 8.33 + round * 0.63, 0.44, "network", {
    phase: "All-Gather",
    round: round + 1,
    links: "R0→R1, R1→R2, R2→R3, R3→R0",
    payload: "4 × 8 bytes",
  }, `all-gather-${round}`);
}

addEvent("hbm-grad-write", "weight.grad global stores", "hbm", 3.02, 2.42, "memory", {
  address: "0x7f20_0000",
  bytes: "4 ranks × 32 B",
}, "gradient-writeback");
addEvent("hbm-nccl", "NCCL read / reduce / write", "hbm", 6.02, 4.55, "memory", {
  buffer: "bucket 0",
  operation: "load partial, reduce, store",
}, "reduce-scatter-0");
addEvent("hbm-optimizer", "AdamW state read / write", "hbm", 12.37, 1.88, "memory", {
  tensors: "parameter, grad, m, v",
}, "optimizer-update");

export const ddpSampleTrace: TraceDataset = {
  id: "ddp-profiler-sample",
  name: "DDP 单机四卡示例 Trace",
  description: "教学用的 profiler 风格样例，展示 CPU、Compute Stream、Communication Stream、NVLink 和 HBM 的时间重叠。",
  source: "sample",
  timeUnit: "ms",
  totalDuration: 14.8,
  lanes,
  events: events.sort((a, b) => a.start - b.start),
  warnings: ["这是结构准确的教学样例，不是从真实机器采集的性能数据"],
  metadata: {
    worldSize: 4,
    backend: "NCCL",
    topology: "single-node NVLink ring",
    eventCount: events.length,
  },
};
