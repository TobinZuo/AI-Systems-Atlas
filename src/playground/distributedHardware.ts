import type { FSDPPhase } from "./fsdp";
import type { ShardedOptimizerPhase } from "./shardedOptimizer";
import type {
  ComparisonPhaseId,
  DistributedStrategyId,
} from "./distributedComparison";

export type ActiveCudaStream = "compute" | "comm" | "compute-then-comm" | "none";

export type DistributedHardwarePattern =
  | "idle"
  | "local-all"
  | "owner-compute"
  | "broadcast"
  | "all-gather"
  | "reduce-scatter"
  | "all-reduce"
  | "compute-then-broadcast"
  | "release";

export type HardwareRankRole =
  | "idle"
  | "compute"
  | "owner"
  | "source"
  | "receiver"
  | "exchange"
  | "reduce"
  | "release";

export interface HardwareRankView {
  rank: number;
  role: HardwareRankRole;
  roleLabel: string;
  stream: "compute" | "comm" | "compute-then-comm" | "none";
  memoryLabel: string;
  selected: boolean;
}

export interface DistributedHardwareSnapshot {
  operation: string;
  cpuCall: string;
  cpuDetail: string;
  activeStream: ActiveCudaStream;
  kernel: string;
  kernelDetail: string;
  hbmObject: string;
  hbmDetail: string;
  link: "NVLink / PCIe P2P" | "无链路传输";
  payload: string;
  explanation: string;
  pattern: DistributedHardwarePattern;
  sourceRank?: number;
  ownerHbmObject?: string;
  replicaHbmObject?: string;
}

export function distributedHardwareRankViews(
  snapshot: DistributedHardwareSnapshot,
  worldSize: number,
  selectedRank: number,
): HardwareRankView[] {
  if (!Number.isInteger(worldSize) || worldSize < 1) {
    throw new Error("worldSize must be a positive integer");
  }

  const sourceRank = Math.min(Math.max(snapshot.sourceRank ?? 0, 0), worldSize - 1);

  return Array.from({ length: worldSize }, (_, rank): HardwareRankView => {
    const selected = rank === selectedRank;

    if (snapshot.pattern === "local-all") {
      return { rank, role: "compute", roleLabel: "执行本地 kernel", stream: "compute", memoryLabel: snapshot.hbmObject, selected };
    }
    if (snapshot.pattern === "owner-compute") {
      const owner = rank === sourceRank;
      return {
        rank,
        role: owner ? "owner" : "idle",
        roleLabel: owner ? "Owner 执行 AdamW" : `等待 Rank ${sourceRank}`,
        stream: owner ? "compute" : "none",
        memoryLabel: owner
          ? snapshot.ownerHbmObject ?? snapshot.hbmObject
          : snapshot.replicaHbmObject ?? "parameter replica + grad",
        selected,
      };
    }
    if (snapshot.pattern === "broadcast") {
      const source = rank === sourceRank;
      return {
        rank,
        role: source ? "source" : "receiver",
        roleLabel: source ? "读取并发送新参数" : "接收并覆盖副本",
        stream: "comm",
        memoryLabel: source ? `source · ${snapshot.hbmObject}` : `destination · ${snapshot.hbmObject}`,
        selected,
      };
    }
    if (snapshot.pattern === "all-gather") {
      return { rank, role: "exchange", roleLabel: "发送 shard，接收完整 W", stream: "comm", memoryLabel: `local shard + ${snapshot.hbmObject}`, selected };
    }
    if (snapshot.pattern === "reduce-scatter") {
      return { rank, role: "reduce", roleLabel: "归约并保留 dW shard", stream: "comm", memoryLabel: `full local dW → ${snapshot.hbmObject}`, selected };
    }
    if (snapshot.pattern === "all-reduce") {
      return { rank, role: "reduce", roleLabel: "发送、接收并归约梯度", stream: "comm", memoryLabel: snapshot.hbmObject, selected };
    }
    if (snapshot.pattern === "compute-then-broadcast") {
      return { rank, role: "exchange", roleLabel: "更新 owner 参数并交换", stream: "compute-then-comm", memoryLabel: snapshot.hbmObject, selected };
    }
    if (snapshot.pattern === "release") {
      return { rank, role: "release", roleLabel: "释放临时完整 buffer", stream: "none", memoryLabel: snapshot.hbmObject, selected };
    }

    return { rank, role: "idle", roleLabel: "保持当前显存状态", stream: "none", memoryLabel: snapshot.hbmObject, selected };
  });
}

export function shardedOptimizerHardwareSnapshot(
  phase: ShardedOptimizerPhase,
  parameterName: string,
  owner: number,
  selectedRank: number,
): DistributedHardwareSnapshot {
  if (phase.kind === "owner-update") {
    const ownsParameter = owner === selectedRank;
    return {
      operation: ownsParameter ? "Owner 本地更新" : "等待其他 owner 更新",
      cpuCall: "local_optimizer.step()",
      cpuDetail: ownsParameter ? `Rank ${selectedRank} 的 optimizer 包含 ${parameterName}` : `Rank ${selectedRank} 的 optimizer 不包含 ${parameterName}`,
      activeStream: ownsParameter ? "compute" : "none",
      kernel: ownsParameter ? "AdamW CUDA kernel" : "没有该参数的更新 kernel",
      kernelDetail: ownsParameter ? "SM 读取 grad、m、v 并计算新参数" : "本 rank 会更新自己拥有的其他参数",
      hbmObject: ownsParameter ? `${parameterName} + grad + m + v` : `${parameterName} + grad`,
      hbmDetail: ownsParameter ? "新参数直接写回 owner 的显存地址" : "副本暂时保持旧值",
      link: "无链路传输",
      payload: ownsParameter ? `Rank ${selectedRank} 写回 ${parameterName}` : `等待 Rank ${owner} 的新值`,
      explanation: "Optimizer step 仍是 GPU 计算。只不过每个 rank 的本地 optimizer 只持有一部分参数。",
      pattern: "owner-compute",
      sourceRank: owner,
      ownerHbmObject: `${parameterName} + grad + m + v`,
      replicaHbmObject: `${parameterName} replica + grad`,
    };
  }

  if (phase.kind === "broadcast") {
    const isSource = owner === selectedRank;
    return {
      operation: `Broadcast ${parameterName}`,
      cpuCall: `dist.broadcast(..., src=${owner})`,
      cpuDetail: "所有 rank 以相同顺序进入同一个 collective",
      activeStream: "comm",
      kernel: "NCCL broadcast kernel",
      kernelDetail: isSource ? "SM 从 owner HBM 读取参数并发送" : "SM 接收参数并写入目标地址",
      hbmObject: parameterName,
      hbmDetail: isSource ? "source buffer 提供新参数" : "destination buffer 被原地覆盖",
      link: "NVLink / PCIe P2P",
      payload: `${parameterName}：Rank ${owner} → 其余 ranks`,
      explanation: "CPU 只负责入队。真正的数据搬运由 comm stream 上的 NCCL kernel 和 GPU 互连完成。",
      pattern: "broadcast",
      sourceRank: owner,
    };
  }

  return {
    operation: phase.kind === "ready" ? "等待本地更新" : "所有模型副本已一致",
    cpuCall: phase.kind === "ready" ? "DDP gradient sync complete" : "optimizer.step() complete",
    cpuDetail: "当前没有新的 GPU 工作需要入队",
    activeStream: "none",
    kernel: "没有活动 kernel",
    kernelDetail: "Compute stream 与 Comm stream 都可以推进后续任务",
    hbmObject: `${parameterName} replica`,
    hbmDetail: phase.kind === "ready" ? "完整梯度已经同步" : "每张卡上的参数值相同",
    link: "无链路传输",
    payload: phase.kind === "ready" ? "完整 parameter + grad" : "一致的 parameter replicas",
    explanation: phase.explanation,
    pattern: "idle",
  };
}

export function fsdpHardwareSnapshot(
  phase: FSDPPhase,
  layerName: string,
  computeDtype: "fp32" | "fp16",
): DistributedHardwareSnapshot {
  if (phase.stage === "all-gather") {
    return {
      operation: phase.pass === "forward" ? "Forward 前重建权重" : "Backward 前重建权重",
      cpuCall: phase.pass === "forward" ? "forward_pre_hook → _unshard()" : "backward hook → _unshard()",
      cpuDetail: "Framework hook 在当前层计算前把 collective 入队",
      activeStream: "comm",
      kernel: "NCCL All-Gather kernel",
      kernelDetail: "SM 读取本地 shard，接收其他 rank 的 shard 并按顺序拼接",
      hbmObject: `${layerName} full W (${computeDtype})`,
      hbmDetail: "在 HBM 中分配当前层的临时完整权重 buffer",
      link: "NVLink / PCIe P2P",
      payload: `${layerName} weight shards`,
      explanation: "只有当前层被展开。其他层仍然只保留各自 shard。",
      pattern: "all-gather",
    };
  }

  if (phase.stage === "compute") {
    const backward = phase.pass === "backward";
    return {
      operation: backward ? "计算 dX 与完整 local dW" : "执行当前层 Forward",
      cpuCall: backward ? "Autograd Engine schedules backward" : "module.forward()",
      cpuDetail: "CPU 线程把算子提交到当前 GPU 的 compute stream",
      activeStream: "compute",
      kernel: backward ? "GEMM backward kernels" : "GEMM forward kernel",
      kernelDetail: backward ? "SM / Tensor Core 读取 W、activation、dY 并产生 dW" : "SM / Tensor Core 读取完整 W 与 activation",
      hbmObject: backward ? `${layerName} full W + local dW` : `${layerName} full W + activation`,
      hbmDetail: backward ? "完整 local dW 暂时存在，随后进入 Reduce-Scatter" : "Kernel 读写当前层临时 buffer",
      link: "无链路传输",
      payload: backward ? "activation、dY、W → dW" : "activation、W → output",
      explanation: "All-Gather 完成后才允许 compute stream 使用完整权重，CUDA event 负责建立依赖。",
      pattern: "local-all",
    };
  }

  if (phase.stage === "reduce-scatter") {
    return {
      operation: "梯度归约并切回 shard",
      cpuCall: "gradient hook → reduce_scatter_tensor()",
      cpuDetail: "Autograd 产出完整 dW 后，hook 把梯度 collective 入队",
      activeStream: "comm",
      kernel: "NCCL Reduce-Scatter kernel",
      kernelDetail: "SM 对来自不同 rank 的梯度片段求和，同时只保留目标 shard",
      hbmObject: `${layerName} dW shard (fp32)`,
      hbmDetail: "临时完整 dW 被释放，平均梯度 shard 写入 parameter.grad",
      link: "NVLink / PCIe P2P",
      payload: `${layerName} full local dW → averaged dW shards`,
      explanation: "归约和切分合成一个 collective，避免先得到每卡完整 global dW。",
      pattern: "reduce-scatter",
    };
  }

  if (phase.stage === "optimizer") {
    return {
      operation: "更新本地 FP32 shard",
      cpuCall: "optimizer.step()",
      cpuDetail: "Optimizer 只遍历本 rank 持有的参数 shard",
      activeStream: "compute",
      kernel: "AdamW CUDA kernel",
      kernelDetail: "SM 读取本地 W shard、dW shard、m、v 并写回新值",
      hbmObject: `${layerName} FP32 master shard + m + v`,
      hbmDetail: "参数和 optimizer state 都不需要在更新后复制",
      link: "无链路传输",
      payload: "local W shard + dW shard → updated W shard",
      explanation: "下一次计算需要该层时，再用 All-Gather 临时重建完整权重。",
      pattern: "local-all",
    };
  }

  if (phase.stage === "reshard") {
    return {
      operation: "释放完整权重并切回 shard",
      cpuCall: "forward hook → _reshard()",
      cpuDetail: "Framework 恢复 parameter.data 指针并回收临时 buffer",
      activeStream: "none",
      kernel: "没有新的计算 kernel",
      kernelDetail: "等待已入队的 forward kernel 完成后再释放 buffer",
      hbmObject: `${layerName} FP32 master shard`,
      hbmDetail: "临时完整 W 已释放，只保留长期 shard",
      link: "无链路传输",
      payload: "full W buffer released",
      explanation: "FSDP 的峰值显存不仅由分片比例决定，也由完整 buffer 在 HBM 中存活多久决定。",
      pattern: "release",
    };
  }

  return {
    operation: "长期分片状态",
    cpuCall: "等待当前层进入执行窗口",
    cpuDetail: "当前没有针对这层的新任务入队",
    activeStream: "none",
    kernel: "没有活动 kernel",
    kernelDetail: "GPU 可以执行其他层或其他 stream 上的任务",
    hbmObject: `${layerName} FP32 master shard`,
    hbmDetail: "每个 rank 只保存约 1/N 的参数行",
    link: "无链路传输",
    payload: "local weight shard",
    explanation: "完整参数此刻不存在于任何单张 GPU。",
    pattern: "idle",
  };
}

export function comparisonHardwareSnapshot(
  strategy: DistributedStrategyId,
  phase: ComparisonPhaseId,
): DistributedHardwareSnapshot {
  if (strategy === "ddp") {
    if (phase === "gradient-sync") {
      return {
        operation: "DDP 梯度 All-Reduce",
        cpuCall: "Autograd hook → process_group.allreduce()",
        cpuDetail: "Bucket ready 后，框架把 NCCL collective 入队",
        activeStream: "comm",
        kernel: "NCCL Reduce-Scatter / All-Gather kernels",
        kernelDetail: "SM 对梯度 chunk 求和，再传播完整结果",
        hbmObject: "gradient bucket",
        hbmDetail: "local dW 被原地替换为 global averaged dW",
        link: "NVLink / PCIe P2P",
        payload: "gradient chunks",
        explanation: "参数始终在本地，DDP 的模型状态通信主要发生在梯度路径。",
        pattern: "all-reduce",
      };
    }
    if (phase === "optimizer-step") {
      return {
        operation: "每个 rank 执行相同 AdamW",
        cpuCall: "optimizer.step()",
        cpuDetail: "所有 rank 都遍历完整参数列表",
        activeStream: "compute",
        kernel: "AdamW CUDA kernel",
        kernelDetail: "每卡独立计算相同更新",
        hbmObject: "full W + dW + m + v",
        hbmDetail: "完整状态全部驻留在每张 GPU",
        link: "无链路传输",
        payload: "full local optimizer state",
        explanation: "梯度同步后输入相同，所以无需再广播参数。",
        pattern: "local-all",
      };
    }
  }

  if (strategy === "zero-1") {
    if (phase === "optimizer-step") {
      return {
        operation: "Owner 更新并广播参数",
        cpuCall: "local_optimizer.step() → broadcast()",
        cpuDetail: "每个 rank 更新本地 owner 参数，再按全局顺序进入 collective",
        activeStream: "compute-then-comm",
        kernel: "AdamW kernel → NCCL broadcast kernel",
        kernelDetail: "先在 compute stream 更新，再由 comm stream 同步新参数",
        hbmObject: "full W + dW，local m + v",
        hbmDetail: "非 owner 的参数显存地址被新值覆盖",
        link: "NVLink / PCIe P2P",
        payload: "updated parameter shards",
        explanation: "同一个进程提交计算和通信，CUDA event 与 stream 顺序保证先更新再发送。",
        pattern: "compute-then-broadcast",
      };
    }
    if (phase === "gradient-sync") {
      return comparisonHardwareSnapshot("ddp", phase);
    }
    if (phase === "next-forward") {
      return {
        operation: "使用已同步的完整参数执行 Forward",
        cpuCall: "module.forward()",
        cpuDetail: "所有 owner 的参数广播完成后，CPU 提交下一轮算子",
        activeStream: "compute",
        kernel: "GEMM forward kernel",
        kernelDetail: "SM / Tensor Core 直接读取本卡上的完整参数副本",
        hbmObject: "full synchronized W + activation",
        hbmDetail: "下一轮 forward 不需要再次通信参数",
        link: "无链路传输",
        payload: "activation + full W → output",
        explanation: "ZeRO-1 在 optimizer step 末尾恢复了完整参数副本，所以 forward 路径仍像 DDP。",
        pattern: "local-all",
      };
    }
  }

  if (strategy === "fsdp") {
    if (phase === "gradient-sync") {
      return {
        operation: "FSDP 梯度 Reduce-Scatter",
        cpuCall: "gradient hook → reduce_scatter_tensor()",
        cpuDetail: "完整 local dW 产生后立即提交 collective",
        activeStream: "comm",
        kernel: "NCCL Reduce-Scatter kernel",
        kernelDetail: "跨 rank 求和，同时输出本地 dW shard",
        hbmObject: "full local dW → dW shard",
        hbmDetail: "完整梯度 buffer 在 collective 后释放",
        link: "NVLink / PCIe P2P",
        payload: "gradient shards",
        explanation: "通信结果直接匹配本地参数 shard，不需要完整 global dW。",
        pattern: "reduce-scatter",
      };
    }
    if (phase === "optimizer-step") {
      return {
        operation: "FSDP 本地 shard 更新",
        cpuCall: "optimizer.step()",
        cpuDetail: "每个 rank 只遍历本地参数 shard",
        activeStream: "compute",
        kernel: "AdamW CUDA kernel",
        kernelDetail: "SM 只读写本地 W、dW、m、v shard",
        hbmObject: "local W + dW + m + v shards",
        hbmDetail: "更新后仍保持长期分片",
        link: "无链路传输",
        payload: "local optimizer shards",
        explanation: "参数在下一层计算前才会临时 All-Gather。",
        pattern: "local-all",
      };
    }
    if (phase === "next-forward") {
      return {
        operation: "当前层权重 All-Gather",
        cpuCall: "forward_pre_hook → _unshard()",
        cpuDetail: "当前层进入执行窗口时才提交参数 collective",
        activeStream: "comm",
        kernel: "NCCL All-Gather kernel",
        kernelDetail: "SM 交换 weight shard 并写入完整 W buffer",
        hbmObject: "local W shard + transient full W",
        hbmDetail: "完整权重只在当前层计算窗口内存活",
        link: "NVLink / PCIe P2P",
        payload: "weight shards",
        explanation: "FSDP 用额外参数通信换取长期显存下降。",
        pattern: "all-gather",
      };
    }
  }

  if (strategy === "ddp" && phase === "next-forward") {
    return {
      operation: "使用本地完整参数执行 Forward",
      cpuCall: "module.forward()",
      cpuDetail: "CPU 进程把下一轮算子提交到 compute stream",
      activeStream: "compute",
      kernel: "GEMM forward kernel",
      kernelDetail: "SM / Tensor Core 直接读取本卡上的完整参数副本",
      hbmObject: "full W + activation",
      hbmDetail: "参数在整个训练过程中始终完整驻留",
      link: "无链路传输",
      payload: "activation + full W → output",
      explanation: "DDP 不需要为 forward 重建参数，通信主要隐藏在 backward 的梯度同步里。",
      pattern: "local-all",
    };
  }

  return {
    operation: "查看长期驻留状态",
    cpuCall: "没有新的 API 调用",
    cpuDetail: "页面只展示此策略稳定状态下的 HBM 布局",
    activeStream: "none",
    kernel: "没有活动 kernel",
    kernelDetail: "Compute stream 与 Comm stream 当前空闲",
    hbmObject: strategy === "ddp" ? "full W + dW + m + v" : strategy === "zero-1" ? "full W + dW，local m + v" : "local W + dW + m + v shards",
    hbmDetail: "切分策略决定每张卡需要长期保存哪些状态",
    link: "无链路传输",
    payload: "persistent model state",
    explanation: "训练性能还取决于 activation、临时 buffer、分桶和通信重叠，这里只聚焦模型状态。",
    pattern: "idle",
  };
}
