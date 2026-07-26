export const CUDA_WARP_SIZE = 32;
export const TEACHING_MAX_BLOCKS_PER_SM = 2;

export type GPUKernelPhaseId =
  | "cpu-launch"
  | "stream-queue"
  | "grid-expand"
  | "block-schedule"
  | "warp-issue"
  | "hbm-read"
  | "register-compute"
  | "hbm-write";

export interface GPUKernelPhase {
  id: GPUKernelPhaseId;
  label: string;
  compactLabel: string;
  actor: "cpu" | "runtime" | "scheduler" | "memory" | "compute";
  explanation: string;
}

export const gpuKernelPhases: GPUKernelPhase[] = [
  {
    id: "cpu-launch",
    label: "CPU 发起 kernel",
    compactLabel: "CPU 发起",
    actor: "cpu",
    explanation: "Python 进程准备参数、指针和 launch configuration。CPU 不会逐元素计算梯度。",
  },
  {
    id: "stream-queue",
    label: "任务进入 Compute Stream",
    compactLabel: "Stream 入队",
    actor: "runtime",
    explanation: "Kernel launch 被加入有序任务队列。调用通常异步返回，GPU 可以稍后执行。",
  },
  {
    id: "grid-expand",
    label: "Grid 展开全部 Block",
    compactLabel: "Grid 展开",
    actor: "runtime",
    explanation: "Grid 是这次 kernel launch 的全部线程 Block。Grid 可以比 GPU 的 SM 数量大得多。",
  },
  {
    id: "block-schedule",
    label: "Block 被调度到 SM",
    compactLabel: "Block 驻留",
    actor: "scheduler",
    explanation: "一个 Block 整体驻留在一个 SM 上。真实调度由硬件动态决定，页面使用可重复的轮转调度帮助观察。",
  },
  {
    id: "warp-issue",
    label: "Warp Scheduler 发射指令",
    compactLabel: "Warp 发射",
    actor: "scheduler",
    explanation: "Block 内线程按 32 个一组形成 Warp。Warp Scheduler 选择一个就绪 Warp 发射下一条指令。",
  },
  {
    id: "hbm-read",
    label: "相邻 Lane 合并读取 HBM",
    compactLabel: "HBM 读取",
    actor: "memory",
    explanation: "活跃 Lane 读取相邻 float，内存请求经过缓存层级，并尽量合并成更少的显存事务。",
  },
  {
    id: "register-compute",
    label: "每个 Lane 在寄存器中计算",
    compactLabel: "寄存器计算",
    actor: "compute",
    explanation: "每个活跃 Lane 持有自己的输入寄存器，并执行相同的乘法指令。不同 Lane 处理不同元素。",
  },
  {
    id: "hbm-write",
    label: "结果写回 HBM",
    compactLabel: "HBM 写回",
    actor: "memory",
    explanation: "相邻 Lane 把结果写回连续地址。Kernel 完成后，后续同 Stream 任务可以安全读取输出。",
  },
];

export interface GPUKernelConfig {
  vectorLength: number;
  threadsPerBlock: number;
  smCount: number;
  scale: number;
}

export type LaneState = "active" | "guarded" | "unused";

export interface KernelLane {
  laneId: number;
  threadIdx: number | null;
  globalIndex: number | null;
  state: LaneState;
  inputValue: number | null;
  outputValue: number | null;
  inputAddress: number | null;
  outputAddress: number | null;
  inputSegment: number | null;
  outputSegment: number | null;
}

export interface KernelWarp {
  warpId: number;
  lanes: KernelLane[];
  activeLaneCount: number;
  guardedLaneCount: number;
  unusedLaneCount: number;
  inputTransactionCount: number;
  outputTransactionCount: number;
}

export interface KernelBlock {
  blockId: number;
  smId: number;
  wave: number;
  residentSlot: number;
  warps: KernelWarp[];
  activeThreadCount: number;
}

export interface GPUKernelSimulation {
  config: GPUKernelConfig;
  gridDim: number;
  blockDim: number;
  warpSize: number;
  warpsPerBlock: number;
  totalProgramThreads: number;
  totalScheduledLanes: number;
  usefulLaneRatio: number;
  waveCount: number;
  blocks: KernelBlock[];
}

const FLOAT_BYTES = 4;
const TEACHING_MEMORY_SEGMENT_BYTES = 32;
const INPUT_BASE_ADDRESS = 0x1000;
const OUTPUT_BASE_ADDRESS = 0x2000;

function rounded(value: number): number {
  return Number(value.toFixed(4));
}

function gradientValue(index: number): number {
  return rounded(((index % 9) - 4) * 0.25);
}

function segmentFor(address: number): number {
  return Math.floor(address / TEACHING_MEMORY_SEGMENT_BYTES);
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
}

export function simulateGradientScaleKernel(config: GPUKernelConfig): GPUKernelSimulation {
  assertPositiveInteger("vectorLength", config.vectorLength);
  assertPositiveInteger("threadsPerBlock", config.threadsPerBlock);
  assertPositiveInteger("smCount", config.smCount);
  if (!Number.isFinite(config.scale)) throw new RangeError("scale must be finite");

  const gridDim = Math.ceil(config.vectorLength / config.threadsPerBlock);
  const warpsPerBlock = Math.ceil(config.threadsPerBlock / CUDA_WARP_SIZE);
  const blocks: KernelBlock[] = Array.from({ length: gridDim }, (_, blockId) => {
    const warps: KernelWarp[] = Array.from({ length: warpsPerBlock }, (_, warpId) => {
      const lanes: KernelLane[] = Array.from({ length: CUDA_WARP_SIZE }, (_, laneId) => {
        const candidateThreadIdx = warpId * CUDA_WARP_SIZE + laneId;
        const threadExists = candidateThreadIdx < config.threadsPerBlock;
        const globalIndex = threadExists
          ? blockId * config.threadsPerBlock + candidateThreadIdx
          : null;
        const active = globalIndex !== null && globalIndex < config.vectorLength;
        const inputAddress = active ? INPUT_BASE_ADDRESS + globalIndex * FLOAT_BYTES : null;
        const outputAddress = active ? OUTPUT_BASE_ADDRESS + globalIndex * FLOAT_BYTES : null;
        const inputValue = active ? gradientValue(globalIndex) : null;

        return {
          laneId,
          threadIdx: threadExists ? candidateThreadIdx : null,
          globalIndex,
          state: active ? "active" : threadExists ? "guarded" : "unused",
          inputValue,
          outputValue: inputValue === null ? null : rounded(inputValue * config.scale),
          inputAddress,
          outputAddress,
          inputSegment: inputAddress === null ? null : segmentFor(inputAddress),
          outputSegment: outputAddress === null ? null : segmentFor(outputAddress),
        };
      });
      const inputSegments = new Set(
        lanes.flatMap((lane) => lane.inputSegment === null ? [] : [lane.inputSegment]),
      );
      const outputSegments = new Set(
        lanes.flatMap((lane) => lane.outputSegment === null ? [] : [lane.outputSegment]),
      );

      return {
        warpId,
        lanes,
        activeLaneCount: lanes.filter((lane) => lane.state === "active").length,
        guardedLaneCount: lanes.filter((lane) => lane.state === "guarded").length,
        unusedLaneCount: lanes.filter((lane) => lane.state === "unused").length,
        inputTransactionCount: inputSegments.size,
        outputTransactionCount: outputSegments.size,
      };
    });

    return {
      blockId,
      smId: blockId % config.smCount,
      wave: Math.floor(blockId / (config.smCount * TEACHING_MAX_BLOCKS_PER_SM)),
      residentSlot: Math.floor(blockId / config.smCount) % TEACHING_MAX_BLOCKS_PER_SM,
      warps,
      activeThreadCount: warps.reduce((count, warp) => count + warp.activeLaneCount, 0),
    };
  });
  const totalScheduledLanes = gridDim * warpsPerBlock * CUDA_WARP_SIZE;

  return {
    config: { ...config },
    gridDim,
    blockDim: config.threadsPerBlock,
    warpSize: CUDA_WARP_SIZE,
    warpsPerBlock,
    totalProgramThreads: gridDim * config.threadsPerBlock,
    totalScheduledLanes,
    usefulLaneRatio: config.vectorLength / totalScheduledLanes,
    waveCount: Math.max(...blocks.map((block) => block.wave)) + 1,
    blocks,
  };
}

export function selectedKernelLane(
  simulation: GPUKernelSimulation,
  blockId: number,
  warpId: number,
  laneId: number,
): KernelLane {
  const block = simulation.blocks[blockId] ?? simulation.blocks[0];
  const warp = block.warps[warpId] ?? block.warps[0];
  return warp.lanes[laneId] ?? warp.lanes[0];
}

export function formatDeviceAddress(address: number | null): string {
  return address === null ? "没有地址" : `0x${address.toString(16).padStart(4, "0")}`;
}

