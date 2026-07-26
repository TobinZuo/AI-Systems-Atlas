export type CollectiveOperation = "broadcast" | "reduce" | "all-reduce" | "scatter" | "gather" | "all-gather" | "reduce-scatter" | "all-to-all" | "barrier";
export type CollectiveReduction = "sum" | "avg" | "max";
export type CollectiveBackend = "nccl" | "gloo";
export type CollectivePhase = "inputs" | "contract" | "transfer" | "complete";
export type CollectiveFault = "none" | "missing-rank" | "operation-mismatch" | "count-mismatch" | "root-mismatch";
export type CollectiveSlotState = "resident" | "pending" | "partial" | "received" | "not-applicable" | "coordination";

export interface CollectiveConfig {
  operation: CollectiveOperation;
  reduction: CollectiveReduction;
  backend: CollectiveBackend;
  rootRank: number;
  fault: CollectiveFault;
}

export interface CollectiveOperationDefinition {
  id: CollectiveOperation;
  label: string;
  family: "one-to-many" | "many-to-one" | "many-to-many" | "synchronization";
  rootRequired: boolean;
  reductionRequired: boolean;
  inputDescription: string;
  outputDescription: string;
  commonUse: string;
}

export interface CollectiveSlot {
  slot: number;
  label: string;
  values: number[] | null;
  state: CollectiveSlotState;
  sourceRanks: number[];
  destinationRank: number | null;
}

export interface CollectiveRankView {
  rank: number;
  inputSlots: CollectiveSlot[];
  outputSlots: CollectiveSlot[];
  participates: boolean;
  callName: string;
  count: number;
}

export interface LogicalCollectiveRoute {
  from: number;
  to: number;
  slot: number;
  label: string;
  action: "copy" | "reduce" | "exchange" | "arrive";
}

export interface CollectiveCallPacket {
  pythonApi: string;
  backendApi: string;
  sendPointer: string;
  receivePointer: string;
  count: number;
  dtype: "torch.float32 → ncclFloat32" | "torch.float32 → gloo::float";
  reduceOp: string;
  root: string;
  communicator: string;
  stream: string;
  inPlace: boolean;
}

export interface CollectiveFaultResult {
  fault: CollectiveFault;
  failurePhase: "contract" | null;
  affectedRanks: number[];
  title: string;
  explanation: string;
}

export interface CollectiveSimulation {
  config: CollectiveConfig;
  definition: CollectiveOperationDefinition;
  worldSize: 4;
  ranks: CollectiveRankView[];
  routes: LogicalCollectiveRoute[];
  fault: CollectiveFaultResult;
}

export const collectiveOperationDefinitions: CollectiveOperationDefinition[] = [
  { id: "broadcast", label: "Broadcast", family: "one-to-many", rootRequired: true, reductionRequired: false, inputDescription: "root 提供一个 Tensor", outputDescription: "每个 rank 得到 root Tensor", commonUse: "同步配置、参数或初始状态" },
  { id: "reduce", label: "Reduce", family: "many-to-one", rootRequired: true, reductionRequired: true, inputDescription: "每个 rank 提供同 shape Tensor", outputDescription: "只有 root 得到逐元素归约", commonUse: "集中统计结果" },
  { id: "all-reduce", label: "AllReduce", family: "many-to-many", rootRequired: false, reductionRequired: true, inputDescription: "每个 rank 提供同 shape Tensor", outputDescription: "每个 rank 得到同一归约结果", commonUse: "DDP 梯度同步" },
  { id: "scatter", label: "Scatter", family: "one-to-many", rootRequired: true, reductionRequired: false, inputDescription: "root 提供按目标 rank 排列的 chunks", outputDescription: "每个 rank 得到自己的 chunk", commonUse: "从中心切分输入" },
  { id: "gather", label: "Gather", family: "many-to-one", rootRequired: true, reductionRequired: false, inputDescription: "每个 rank 提供一个 chunk", outputDescription: "root 按 rank 顺序拼接所有 chunks", commonUse: "集中保存或评估输出" },
  { id: "all-gather", label: "AllGather", family: "many-to-many", rootRequired: false, reductionRequired: false, inputDescription: "每个 rank 提供一个 chunk", outputDescription: "每个 rank 按 rank 顺序得到全部 chunks", commonUse: "FSDP 临时恢复完整参数" },
  { id: "reduce-scatter", label: "ReduceScatter", family: "many-to-many", rootRequired: false, reductionRequired: true, inputDescription: "每个 rank 提供 N 个 chunks", outputDescription: "先按 chunk 归约，再让 rank r 保留 chunk r", commonUse: "FSDP 梯度归约与分片" },
  { id: "all-to-all", label: "AllToAll", family: "many-to-many", rootRequired: false, reductionRequired: false, inputDescription: "每个 rank 为每个目标准备不同 chunk", outputDescription: "每个 rank 从每个来源收到一个 chunk", commonUse: "MoE token dispatch" },
  { id: "barrier", label: "Barrier", family: "synchronization", rootRequired: false, reductionRequired: false, inputDescription: "没有用户 Tensor payload", outputDescription: "所有 rank 到达后一起继续", commonUse: "阶段同步与调试" },
];

export const defaultCollectiveConfig: CollectiveConfig = {
  operation: "all-gather",
  reduction: "sum",
  backend: "nccl",
  rootRank: 0,
  fault: "none",
};

const worldSize = 4 as const;
const rankVector = (rank: number): number[] => [rank * 10 + 1, rank * 10 + 2];
const matrixChunk = (source: number, destination: number): number[] => [source * 100 + destination * 10 + 1, source * 100 + destination * 10 + 2];

function definitionFor(operation: CollectiveOperation): CollectiveOperationDefinition {
  const definition = collectiveOperationDefinitions.find((item) => item.id === operation);
  if (!definition) throw new Error(`missing collective definition ${operation}`);
  return definition;
}

function validateConfig(config: CollectiveConfig): void {
  definitionFor(config.operation);
  if (!["sum", "avg", "max"].includes(config.reduction)) throw new Error("unsupported reduction");
  if (!["nccl", "gloo"].includes(config.backend)) throw new Error("unsupported backend");
  if (!Number.isInteger(config.rootRank) || config.rootRank < 0 || config.rootRank >= worldSize) throw new Error("root rank out of range");
  if (!["none", "missing-rank", "operation-mismatch", "count-mismatch", "root-mismatch"].includes(config.fault)) throw new Error("unsupported collective fault");
}

function reduceVectors(vectors: number[][], reduction: CollectiveReduction): number[] {
  if (vectors.length === 0) throw new Error("cannot reduce empty vectors");
  return vectors[0].map((_, index) => {
    const values = vectors.map((vector) => vector[index]);
    if (reduction === "max") return Math.max(...values);
    const sum = values.reduce((total, value) => total + value, 0);
    return reduction === "avg" ? sum / values.length : sum;
  });
}

function inputSlots(operation: CollectiveOperation, rank: number, rootRank: number): CollectiveSlot[] {
  if (operation === "barrier") return [{ slot: 0, label: "arrival token", values: null, state: "coordination", sourceRanks: [rank], destinationRank: null }];
  if (operation === "broadcast") return [{ slot: 0, label: rank === rootRank ? "root send buffer" : "receive buffer", values: rank === rootRank ? rankVector(rootRank) : null, state: rank === rootRank ? "resident" : "pending", sourceRanks: [rootRank], destinationRank: rank }];
  if (operation === "scatter") return rank === rootRank
    ? Array.from({ length: worldSize }, (_, destination) => ({ slot: destination, label: `to R${destination}`, values: matrixChunk(rootRank, destination), state: "resident" as const, sourceRanks: [rootRank], destinationRank: destination }))
    : [{ slot: rank, label: `receive C${rank}`, values: null, state: "pending", sourceRanks: [rootRank], destinationRank: rank }];
  if (operation === "reduce-scatter" || operation === "all-to-all") return Array.from({ length: worldSize }, (_, destination) => ({ slot: destination, label: `for R${destination}`, values: matrixChunk(rank, destination), state: "resident", sourceRanks: [rank], destinationRank: destination }));
  return [{ slot: rank, label: `from R${rank}`, values: rankVector(rank), state: "resident", sourceRanks: [rank], destinationRank: null }];
}

function completedOutputSlots(config: CollectiveConfig, rank: number): CollectiveSlot[] {
  const { operation, rootRank, reduction } = config;
  const vectors = Array.from({ length: worldSize }, (_, source) => rankVector(source));
  if (operation === "barrier") return [{ slot: 0, label: "all arrived", values: null, state: "coordination", sourceRanks: [0, 1, 2, 3], destinationRank: rank }];
  if (operation === "broadcast") return [{ slot: 0, label: `from root R${rootRank}`, values: rankVector(rootRank), state: "received", sourceRanks: [rootRank], destinationRank: rank }];
  if (operation === "reduce" || operation === "all-reduce") {
    if (operation === "reduce" && rank !== rootRank) return [{ slot: 0, label: "no defined receive result", values: null, state: "not-applicable", sourceRanks: [0, 1, 2, 3], destinationRank: rootRank }];
    return [{ slot: 0, label: `${reduction.toUpperCase()} from all ranks`, values: reduceVectors(vectors, reduction), state: "received", sourceRanks: [0, 1, 2, 3], destinationRank: rank }];
  }
  if (operation === "scatter") return [{ slot: rank, label: `chunk C${rank}`, values: matrixChunk(rootRank, rank), state: "received", sourceRanks: [rootRank], destinationRank: rank }];
  if (operation === "gather" || operation === "all-gather") {
    if (operation === "gather" && rank !== rootRank) return [{ slot: 0, label: "no defined receive result", values: null, state: "not-applicable", sourceRanks: [rank], destinationRank: rootRank }];
    return Array.from({ length: worldSize }, (_, source) => ({ slot: source, label: `from R${source}`, values: rankVector(source), state: "received", sourceRanks: [source], destinationRank: rank }));
  }
  if (operation === "reduce-scatter") return [{ slot: rank, label: `reduced C${rank}`, values: reduceVectors(Array.from({ length: worldSize }, (_, source) => matrixChunk(source, rank)), reduction), state: "received", sourceRanks: [0, 1, 2, 3], destinationRank: rank }];
  return Array.from({ length: worldSize }, (_, source) => ({ slot: source, label: `from R${source}`, values: matrixChunk(source, rank), state: "received", sourceRanks: [source], destinationRank: rank }));
}

function routesFor(config: CollectiveConfig): LogicalCollectiveRoute[] {
  const { operation, rootRank } = config;
  if (operation === "barrier") return Array.from({ length: worldSize }, (_, rank) => ({ from: rank, to: rank, slot: 0, label: `R${rank} arrived`, action: "arrive" as const }));
  if (operation === "broadcast" || operation === "scatter") return Array.from({ length: worldSize }, (_, to) => ({ from: rootRank, to, slot: operation === "scatter" ? to : 0, label: operation === "scatter" ? `C${to}` : "root tensor", action: "copy" as const }));
  if (operation === "reduce" || operation === "gather") return Array.from({ length: worldSize }, (_, from) => ({ from, to: rootRank, slot: from, label: `from R${from}`, action: operation === "reduce" ? "reduce" as const : "copy" as const }));
  if (operation === "all-reduce" || operation === "all-gather") return Array.from({ length: worldSize }, (_, from) => Array.from({ length: worldSize }, (_, to) => ({ from, to, slot: from, label: `R${from} → R${to}`, action: operation === "all-reduce" ? "reduce" as const : "copy" as const }))).flat();
  if (operation === "reduce-scatter") return Array.from({ length: worldSize }, (_, from) => Array.from({ length: worldSize }, (_, to) => ({ from, to, slot: to, label: `C${to}`, action: "reduce" as const }))).flat();
  return Array.from({ length: worldSize }, (_, from) => Array.from({ length: worldSize }, (_, to) => ({ from, to, slot: to, label: `R${from}.C${to}`, action: "exchange" as const }))).flat();
}

function callName(operation: CollectiveOperation): string {
  return `dist.${operation.replace(/-/g, "_")}`;
}

function pythonCall(config: CollectiveConfig, view: CollectiveRankView): string {
  const expectedName = callName(config.operation);
  if (view.callName !== expectedName) return `${view.callName}(...)  # mismatched call`;
  const reduceOp = `dist.ReduceOp.${config.reduction.toUpperCase()}`;
  if (config.operation === "broadcast") return `dist.broadcast(tensor, src=${config.rootRank}, group=WORLD)`;
  if (config.operation === "reduce") return `dist.reduce(tensor, dst=${config.rootRank}, op=${reduceOp})`;
  if (config.operation === "all-reduce") return `dist.all_reduce(tensor, op=${reduceOp}, group=WORLD)`;
  if (config.operation === "scatter") return `dist.scatter(output, scatter_list, src=${config.rootRank})`;
  if (config.operation === "gather") return `dist.gather(tensor, gather_list, dst=${config.rootRank})`;
  if (config.operation === "all-gather") return "dist.all_gather(output_list, tensor, group=WORLD)";
  if (config.operation === "reduce-scatter") return `dist.reduce_scatter(output, input_list, op=${reduceOp})`;
  if (config.operation === "all-to-all") return "dist.all_to_all(output_list, input_list, group=WORLD)";
  return "dist.barrier(group=WORLD)";
}

function analyzeFault(config: CollectiveConfig): CollectiveFaultResult {
  if (config.fault === "none") return { fault: "none", failurePhase: null, affectedRanks: [], title: "四个 rank 的调用契约一致", explanation: "operation、count、dtype、root 与 collective 顺序一致，可以形成一次完整操作。" };
  if (config.fault === "missing-rank") return { fault: config.fault, failurePhase: "contract", affectedRanks: [3], title: "Rank 3 没有进入这次 Collective", explanation: "其他 rank 已经提交操作，但通信组缺少一个参与者，只能等待或最终超时。" };
  if (config.fault === "operation-mismatch") return { fault: config.fault, failurePhase: "contract", affectedRanks: [3], title: `Rank 3 调用了 ${config.operation === "broadcast" ? "AllReduce" : "Broadcast"}`, explanation: "同一个 communicator 上的 collective 顺序和类型必须匹配，否则各 rank 对同一批消息的解释不同。" };
  if (config.fault === "count-mismatch") return { fault: config.fault, failurePhase: "contract", affectedRanks: [3], title: "Rank 3 的 count 少一半", explanation: "Backend 得到的元素数量不同，发送和接收边界不一致，结果属于未定义行为，可能 hang、崩溃或数据损坏。" };
  const rooted = definitionFor(config.operation).rootRequired;
  return rooted
    ? { fault: config.fault, failurePhase: "contract", affectedRanks: [3], title: `Rank 3 声明 root=${(config.rootRank + 1) % worldSize}`, explanation: "Root 是 collective 契约的一部分。不同 rank 对数据源或最终接收者的理解不一致。" }
    : { fault: config.fault, failurePhase: null, affectedRanks: [], title: `${definitionFor(config.operation).label} 没有 root 参数`, explanation: "这个操作对所有 rank 对称，root mismatch 不适用，当前调用仍然有效。" };
}

export function simulateCollective(config: CollectiveConfig): CollectiveSimulation {
  validateConfig(config);
  const definition = definitionFor(config.operation);
  const ranks = Array.from({ length: worldSize }, (_, rank): CollectiveRankView => {
    const inputs = inputSlots(config.operation, rank, config.rootRank);
    return {
      rank,
      inputSlots: inputs,
      outputSlots: completedOutputSlots(config, rank),
      participates: !(config.fault === "missing-rank" && rank === 3),
      callName: config.fault === "operation-mismatch" && rank === 3 ? (config.operation === "broadcast" ? "dist.all_reduce" : "dist.broadcast") : callName(config.operation),
      count: config.fault === "count-mismatch" && rank === 3 ? 1 : config.operation === "barrier" ? 1 : 2,
    };
  });
  return { config, definition, worldSize, ranks, routes: routesFor(config), fault: analyzeFault(config) };
}

function pendingSlot(slot: CollectiveSlot): CollectiveSlot {
  if (slot.state === "not-applicable") return { ...slot, values: null };
  return { ...slot, values: null, state: slot.state === "coordination" ? "pending" : "pending" };
}

export function rankViewAtPhase(simulation: CollectiveSimulation, rank: number, phase: CollectivePhase): CollectiveRankView {
  if (!Number.isInteger(rank) || rank < 0 || rank >= simulation.worldSize) throw new Error("rank out of range");
  const complete = simulation.ranks[rank];
  if (phase === "complete" && simulation.fault.failurePhase) return { ...complete, outputSlots: complete.outputSlots.map(pendingSlot) };
  if (phase === "complete") return complete;
  if (phase === "inputs" || phase === "contract") return { ...complete, outputSlots: complete.outputSlots.map(pendingSlot) };

  const operation = simulation.config.operation;
  const outputSlots = complete.outputSlots.map((slot): CollectiveSlot => {
    if (slot.state === "not-applicable") return slot;
    if (operation === "barrier") return rank < 2 ? { ...slot, state: "partial", sourceRanks: [0, 1] } : pendingSlot(slot);
    if (operation === "broadcast") return rank === simulation.config.rootRank ? { ...slot, state: "resident" } : pendingSlot(slot);
    if (operation === "reduce" || operation === "all-reduce" || operation === "reduce-scatter") {
      const contributorVectors = operation === "reduce-scatter"
        ? [matrixChunk(0, rank), matrixChunk(1, rank)]
        : [rankVector(0), rankVector(1)];
      return { ...slot, values: reduceVectors(contributorVectors, simulation.config.reduction), state: "partial", sourceRanks: [0, 1] };
    }
    if (operation === "scatter") return rank === simulation.config.rootRank ? { ...slot, state: "resident" } : pendingSlot(slot);
    if (operation === "gather" || operation === "all-gather") return slot.sourceRanks[0] === rank ? { ...slot, state: "resident" } : pendingSlot(slot);
    return slot.sourceRanks[0] === rank ? { ...slot, state: "resident" } : pendingSlot(slot);
  });
  return { ...complete, outputSlots };
}

export function collectiveCallPacket(simulation: CollectiveSimulation, rank: number): CollectiveCallPacket {
  if (!Number.isInteger(rank) || rank < 0 || rank >= simulation.worldSize) throw new Error("rank out of range");
  const view = simulation.ranks[rank];
  const definition = simulation.definition;
  const suffix = rank.toString(16).padStart(2, "0");
  const backendPrefix = simulation.config.backend === "nccl" ? "nccl" : "gloo";
  const apiStem = simulation.config.operation === "all-to-all"
    ? "AlltoAll"
    : simulation.config.operation.split("-").map((part) => part[0].toUpperCase() + part.slice(1)).join("");
  const inPlace = ["broadcast", "all-reduce"].includes(simulation.config.operation);
  return {
    pythonApi: pythonCall(simulation.config, view),
    backendApi: simulation.config.operation === "barrier"
      ? "ProcessGroup::barrier"
      : simulation.config.backend === "nccl"
        ? `${backendPrefix}${apiStem}`
        : `ProcessGroupGloo::${simulation.config.operation.replace(/-/g, "")}`,
    sendPointer: `0x7f20a0${suffix}00`,
    receivePointer: inPlace ? `0x7f20a0${suffix}00` : `0x7f20b0${suffix}00`,
    count: view.count,
    dtype: simulation.config.backend === "nccl" ? "torch.float32 → ncclFloat32" : "torch.float32 → gloo::float",
    reduceOp: definition.reductionRequired
      ? simulation.config.backend === "nccl"
        ? `${backendPrefix}${simulation.config.reduction[0].toUpperCase()}${simulation.config.reduction.slice(1)}`
        : `ReduceOp::${simulation.config.reduction.toUpperCase()}`
      : "not used",
    root: definition.rootRequired ? `rank ${simulation.config.rootRank}` : "not used",
    communicator: simulation.config.backend === "nccl" ? `ncclComm(rank=${rank}, n=4)` : `gloo::Context(rank=${rank}, n=4)`,
    stream: simulation.config.backend === "nccl" ? "CUDA comm stream" : "CPU worker thread",
    inPlace,
  };
}
