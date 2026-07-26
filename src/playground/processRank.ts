export type ClusterLayout = "single-node" | "two-node";
export type ProcessBackend = "nccl" | "gloo";
export type ProcessInitFault = "none" | "duplicate-rank" | "world-size-mismatch" | "master-port-mismatch" | "device-collision";
export type ProcessInitPhaseId = "spawn" | "environment" | "rendezvous" | "communicator" | "ready";
export type ProcessState = "not-started" | "running" | "waiting" | "ready" | "error" | "blocked";

export interface ProcessRankConfig {
  layout: ClusterLayout;
  backend: ProcessBackend;
  fault: ProcessInitFault;
}

export interface ProcessRankEnvironment {
  MASTER_ADDR: string;
  MASTER_PORT: string;
  RANK: string;
  WORLD_SIZE: string;
  LOCAL_RANK: string;
  LOCAL_WORLD_SIZE: string;
}

export interface RankWorker {
  workerId: string;
  pid: number;
  hostname: string;
  nodeRank: number;
  globalRank: number;
  advertisedRank: number;
  localRank: number;
  device: string;
  environment: ProcessRankEnvironment;
  defaultGroupObject: string;
  communicatorObject: string;
}

export interface ProcessInitPhase {
  id: ProcessInitPhaseId;
  index: number;
  label: string;
  shortLabel: string;
  description: string;
  plane: "launcher" | "control" | "collective";
}

export interface ProcessFaultResult {
  fault: ProcessInitFault;
  failurePhase: ProcessInitPhaseId | null;
  affectedWorkerIds: string[];
  title: string;
  explanation: string;
}

export interface RankProcessState {
  workerId: string;
  state: ProcessState;
  detail: string;
}

export interface ProcessRankSimulation {
  config: ProcessRankConfig;
  workers: RankWorker[];
  phases: ProcessInitPhase[];
  fault: ProcessFaultResult;
  rendezvousEndpoint: string;
  logicalGroupId: string;
}

export interface CollectiveDataPath {
  sourceRank: number;
  targetRank: number;
  sourceNode: string;
  targetNode: string;
  sourceMemory: "GPU HBM" | "CPU DRAM";
  targetMemory: "GPU HBM" | "CPU DRAM";
  collectiveLayer: "ProcessGroupNCCL" | "ProcessGroupGloo";
  transport: "NVLink / PCIe P2P" | "InfiniBand / RoCE / IP network" | "TCP sockets";
  crossesNode: boolean;
  payload: string;
  usesMasterPortForPayload: false;
}

export const processInitPhases: ProcessInitPhase[] = [
  { id: "spawn", index: 0, label: "torchrun 启动 worker", shortLabel: "Spawn", description: "每个 worker 是独立的 OS Python 进程，拥有自己的地址空间。", plane: "launcher" },
  { id: "environment", index: 1, label: "注入身份环境变量", shortLabel: "Identity", description: "RANK、WORLD_SIZE、LOCAL_RANK 和 rendezvous endpoint 进入每个进程。", plane: "launcher" },
  { id: "rendezvous", index: 2, label: "连接 TCPStore", shortLabel: "Rendezvous", description: "进程通过共同 endpoint 交换初始化信息，并等待预期数量的成员。", plane: "control" },
  { id: "communicator", index: 3, label: "创建 Process Group", shortLabel: "Group", description: "每个进程创建自己的默认组对象，NCCL 后端还会为本 rank 与 GPU 创建设备 communicator。", plane: "collective" },
  { id: "ready", index: 4, label: "集合通信就绪", shortLabel: "Ready", description: "逻辑上属于同一组的 rank 可以按完全相同的顺序调用 collective。", plane: "collective" },
];

export const defaultProcessRankConfig: ProcessRankConfig = {
  layout: "two-node",
  backend: "nccl",
  fault: "none",
};

function validateConfig(config: ProcessRankConfig): void {
  if (config.layout !== "single-node" && config.layout !== "two-node") throw new Error("unsupported cluster layout");
  if (config.backend !== "nccl" && config.backend !== "gloo") throw new Error("unsupported process backend");
  const faults: ProcessInitFault[] = ["none", "duplicate-rank", "world-size-mismatch", "master-port-mismatch", "device-collision"];
  if (!faults.includes(config.fault)) throw new Error("unsupported init fault");
}

function buildWorkers(config: ProcessRankConfig): RankWorker[] {
  const worldSize = 4;
  const localWorldSize = config.layout === "single-node" ? 4 : 2;
  const workers = Array.from({ length: worldSize }, (_, globalRank): RankWorker => {
    const nodeRank = config.layout === "single-node" ? 0 : Math.floor(globalRank / localWorldSize);
    const localRank = globalRank % localWorldSize;
    const advertisedRank = config.fault === "duplicate-rank" && globalRank === 3 ? 2 : globalRank;
    const advertisedWorldSize = config.fault === "world-size-mismatch" && globalRank === 3 ? 3 : worldSize;
    const masterPort = config.fault === "master-port-mismatch" && globalRank === 3 ? "29600" : "29500";
    const effectiveLocalRank = config.fault === "device-collision" && globalRank === 1 ? 0 : localRank;
    return {
      workerId: `worker-${globalRank}`,
      pid: 42000 + globalRank,
      hostname: `node-${nodeRank}`,
      nodeRank,
      globalRank,
      advertisedRank,
      localRank: effectiveLocalRank,
      device: config.backend === "nccl" ? `cuda:${effectiveLocalRank}` : "cpu",
      environment: {
        MASTER_ADDR: "10.0.0.10",
        MASTER_PORT: masterPort,
        RANK: String(advertisedRank),
        WORLD_SIZE: String(advertisedWorldSize),
        LOCAL_RANK: String(effectiveLocalRank),
        LOCAL_WORLD_SIZE: String(localWorldSize),
      },
      defaultGroupObject: `0xPG${globalRank.toString(16).padStart(4, "0")}`,
      communicatorObject: config.backend === "nccl" ? `0xNCCL${globalRank.toString(16).padStart(4, "0")}` : `0xGLOO${globalRank.toString(16).padStart(4, "0")}`,
    };
  });
  return workers;
}

function analyzeFault(config: ProcessRankConfig, workers: RankWorker[]): ProcessFaultResult {
  if (config.fault === "none") return {
    fault: "none", failurePhase: null, affectedWorkerIds: [], title: "初始化契约一致", explanation: "四个进程连接同一 rendezvous endpoint，声明唯一 rank 与同一个 world size。",
  };
  if (config.fault === "duplicate-rank") return {
    fault: config.fault, failurePhase: "rendezvous", affectedWorkerIds: [workers[2].workerId, workers[3].workerId], title: "rank 2 被声明两次，rank 3 缺席", explanation: "rank 是通信组内的唯一身份。重复 rank 不能因为 PID 不同而被自动消歧，组无法收齐 0 到 world_size-1。",
  };
  if (config.fault === "world-size-mismatch") return {
    fault: config.fault, failurePhase: "rendezvous", affectedWorkerIds: [workers[3].workerId], title: "同一组出现两个 world size", explanation: "一个进程等待 3 个成员，其他进程等待 4 个成员，集合通信的参与者集合无法形成共同契约。",
  };
  if (config.fault === "master-port-mismatch") return {
    fault: config.fault, failurePhase: "rendezvous", affectedWorkerIds: [workers[3].workerId], title: "worker 3 连接了另一个 TCP endpoint", explanation: "MASTER_ADDR 与 MASTER_PORT 标识 rendezvous 控制面入口。端口不同意味着没有加入同一个 Store。",
  };
  return config.backend === "nccl" ? {
    fault: config.fault, failurePhase: "communicator", affectedWorkerIds: [workers[0].workerId, workers[1].workerId], title: "同一节点的两个 NCCL rank 绑定 cuda:0", explanation: "多进程 NCCL 要求每个进程独占自己使用的 GPU。共享同一 CUDA device 可能导致 invalid usage 或 hang。",
  } : {
    fault: config.fault, failurePhase: null, affectedWorkerIds: [], title: "Gloo 不消费 CUDA device 绑定", explanation: "当前教学场景的 Gloo collective 在 CPU tensor 上运行，因此重复 LOCAL_RANK 不影响这个 CPU 通信组；它仍会影响随后需要 GPU 的代码。",
  };
}

export function simulateProcessRanks(config: ProcessRankConfig): ProcessRankSimulation {
  validateConfig(config);
  const workers = buildWorkers(config);
  return {
    config,
    workers,
    phases: processInitPhases,
    fault: analyzeFault(config, workers),
    rendezvousEndpoint: "10.0.0.10:29500",
    logicalGroupId: "default_group / world",
  };
}

export function processStatesAtPhase(
  simulation: ProcessRankSimulation,
  phaseId: ProcessInitPhaseId,
): RankProcessState[] {
  const phaseIndex = processInitPhases.findIndex((phase) => phase.id === phaseId);
  if (phaseIndex < 0) throw new Error("unknown init phase");
  const failureIndex = simulation.fault.failurePhase
    ? processInitPhases.findIndex((phase) => phase.id === simulation.fault.failurePhase)
    : Number.POSITIVE_INFINITY;

  return simulation.workers.map((worker): RankProcessState => {
    if (phaseIndex < failureIndex) {
      if (phaseId === "spawn") return { workerId: worker.workerId, state: "running", detail: "OS process running" };
      if (phaseId === "environment") return { workerId: worker.workerId, state: "running", detail: `RANK=${worker.environment.RANK} loaded` };
      if (phaseId === "rendezvous") return { workerId: worker.workerId, state: "waiting", detail: `connect ${worker.environment.MASTER_ADDR}:${worker.environment.MASTER_PORT}` };
      if (phaseId === "communicator") return { workerId: worker.workerId, state: "running", detail: `create ${simulation.config.backend} group` };
    }

    if (phaseIndex === failureIndex) {
      const affected = simulation.fault.affectedWorkerIds.includes(worker.workerId);
      return affected
        ? { workerId: worker.workerId, state: "error", detail: simulation.fault.title }
        : { workerId: worker.workerId, state: "waiting", detail: "waiting for every declared rank" };
    }
    if (phaseIndex > failureIndex) return { workerId: worker.workerId, state: "blocked", detail: `blocked by ${simulation.fault.failurePhase}` };
    if (phaseId === "ready") return { workerId: worker.workerId, state: "ready", detail: `${simulation.config.backend} collective ready` };
    return { workerId: worker.workerId, state: "running", detail: processInitPhases[phaseIndex].description };
  });
}

export function collectiveDataPath(
  simulation: ProcessRankSimulation,
  sourceRank: number,
): CollectiveDataPath {
  if (!Number.isInteger(sourceRank) || sourceRank < 0 || sourceRank >= simulation.workers.length) throw new Error("sourceRank out of range");
  const source = simulation.workers[sourceRank];
  const targetRank = (sourceRank + 1) % simulation.workers.length;
  const target = simulation.workers[targetRank];
  const crossesNode = source.hostname !== target.hostname;
  if (simulation.config.backend === "gloo") return {
    sourceRank,
    targetRank,
    sourceNode: source.hostname,
    targetNode: target.hostname,
    sourceMemory: "CPU DRAM",
    targetMemory: "CPU DRAM",
    collectiveLayer: "ProcessGroupGloo",
    transport: "TCP sockets",
    crossesNode,
    payload: "float32 tensor bytes from CPU memory",
    usesMasterPortForPayload: false,
  };
  return {
    sourceRank,
    targetRank,
    sourceNode: source.hostname,
    targetNode: target.hostname,
    sourceMemory: "GPU HBM",
    targetMemory: "GPU HBM",
    collectiveLayer: "ProcessGroupNCCL",
    transport: crossesNode ? "InfiniBand / RoCE / IP network" : "NVLink / PCIe P2P",
    crossesNode,
    payload: "float32 gradient chunk bytes from GPU memory",
    usesMasterPortForPayload: false,
  };
}
