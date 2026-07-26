import { ArrowDown } from "@phosphor-icons/react/ArrowDown";
import { Code } from "@phosphor-icons/react/Code";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { scrollToSection } from "./scrollToSection";

export type DistributedSystemStackTopic =
  | "process-rank"
  | "collective"
  | "ring-allreduce"
  | "ddp"
  | "zero-1"
  | "fsdp";

type LayerState = "active" | "ready" | "deferred";

interface SystemLayerProfile {
  action: string;
  object: string;
  state: LayerState;
}

interface SystemStackProfile {
  question: string;
  answer: string;
  cta: string;
  targetId: string;
  layers: Record<SystemLayerId, SystemLayerProfile>;
}

type SystemLayerId = "framework" | "process" | "stream" | "kernel" | "memory" | "fabric";

const systemLayers: Array<{
  id: SystemLayerId;
  label: string;
  Icon: typeof Code;
}> = [
  { id: "framework", label: "训练框架", Icon: Code },
  { id: "process", label: "CPU / Rank", Icon: Cpu },
  { id: "stream", label: "CUDA Stream", Icon: Lightning },
  { id: "kernel", label: "GPU / SM", Icon: Stack },
  { id: "memory", label: "HBM Buffer", Icon: Database },
  { id: "fabric", label: "GPU 互连", Icon: Network },
];

export const distributedSystemStackProfiles: Record<DistributedSystemStackTopic, SystemStackProfile> = {
  "process-rank": {
    question: "任务身份建立时，GPU 已经开始通信了吗？",
    answer: "还没有。先完成进程、设备和通信组绑定，后续 Collective 才会提交 GPU 工作。",
    cta: "查看任务建立现场",
    targetId: "process-rank-lab",
    layers: {
      framework: { action: "torchrun + init_process_group", object: "启动参数与 group 配置", state: "active" },
      process: { action: "创建 OS 进程与全局 Rank", object: "RANK / LOCAL_RANK / WORLD_SIZE", state: "active" },
      stream: { action: "尚未提交 CUDA 工作", object: "等待第一次 GPU operation", state: "deferred" },
      kernel: { action: "尚无训练或通信 kernel", object: "GPU 已绑定给 local rank", state: "deferred" },
      memory: { action: "设备可见，buffer 尚未分配", object: "cuda:{LOCAL_RANK}", state: "ready" },
      fabric: { action: "TCPStore 只做控制面会合", object: "数据面将在 Collective 建立", state: "ready" },
    },
  },
  collective: {
    question: "一个 Collective 调用怎样落到真实硬件？",
    answer: "框架把地址、count、dtype、operation、communicator 和 stream 一起交给 Backend。",
    cta: "查看 Backend 调用包",
    targetId: "collective-packet-stage",
    layers: {
      framework: { action: "dist.all_reduce(tensor)", object: "Tensor 语义与 Collective 契约", state: "active" },
      process: { action: "ProcessGroup 对齐所有 Rank", object: "operation / count / dtype / root", state: "active" },
      stream: { action: "提交到 NCCL Comm stream", object: "用 event 等待输入 buffer", state: "active" },
      kernel: { action: "NCCL kernel 读取与归约", object: "send pointer / receive pointer", state: "active" },
      memory: { action: "直接读写 Tensor bytes", object: "GPU HBM 中的 send / recv buffer", state: "active" },
      fabric: { action: "按拓扑搬运分块", object: "NVLink / PCIe / RDMA / Socket", state: "active" },
    },
  },
  "ring-allreduce": {
    question: "每轮 Ring 调度在软硬件各层做了什么？",
    answer: "框架提交一个 AllReduce，NCCL kernel 在 Comm stream 中持续读取、归约并转发 M/N 大小的分块。",
    cta: "查看逐轮 Ring 调度",
    targetId: "ring-round-stage",
    layers: {
      framework: { action: "dist.all_reduce(bucket)", object: "输入输出仍是同一块梯度 buffer", state: "active" },
      process: { action: "每个 Rank 加入同一 communicator", object: "共享 Ring 顺序、count、dtype 与 reduce op", state: "active" },
      stream: { action: "Collective 排入 Comm stream", object: "用 event 等待 bucket 生产完成", state: "active" },
      kernel: { action: "NCCL kernel 边收边归约", object: "Reduce、Copy 与 ReduceAndCopy primitives", state: "active" },
      memory: { action: "按 chunk 读写 HBM buffer", object: "每轮每 Rank 处理约 M/N bytes", state: "active" },
      fabric: { action: "逻辑 next 边并行传输", object: "Ring 映射到 NVLink / PCIe / RDMA", state: "active" },
    },
  },
  ddp: {
    question: "DDP 为什么能把多卡算力变成一次一致更新？",
    answer: "每个 Rank 独立反向计算，Autograd hook 把就绪梯度交给 Comm stream 做 AllReduce。",
    cta: "查看梯度硬件路径",
    targetId: "ddp-system-stage",
    layers: {
      framework: { action: "Autograd hook 标记 bucket ready", object: "parameter.grad -> gradient bucket", state: "active" },
      process: { action: "每个 Rank 计算一个数据分片", object: "完整模型副本 + 本地 batch", state: "active" },
      stream: { action: "Backward 与 AllReduce 重叠", object: "Compute stream + Comm stream", state: "active" },
      kernel: { action: "Backward 与 NCCL kernel", object: "SM 产生梯度并执行归约", state: "active" },
      memory: { action: "bucket 原地写回平均梯度", object: "parameter / grad / bucket / m / v", state: "active" },
      fabric: { action: "交换并归约 gradient chunks", object: "Ring / Tree over NVLink or RDMA", state: "active" },
    },
  },
  "zero-1": {
    question: "ZeRO-1 省下的状态究竟存在哪里？",
    answer: "参数与梯度仍在每张 GPU，只有 owner 保存 m、v 并更新参数，随后广播新参数。",
    cta: "查看 Owner 硬件路径",
    targetId: "zero-hardware-stage",
    layers: {
      framework: { action: "ShardedOptimizer.step()", object: "owner = parameter_index % world_size", state: "active" },
      process: { action: "每个 Rank 只更新 owned params", object: "相同参数顺序决定相同 owner", state: "active" },
      stream: { action: "先本地 AdamW，再广播", object: "Compute stream -> Comm stream", state: "active" },
      kernel: { action: "Owner AdamW + NCCL Broadcast", object: "非 owner 不运行该参数的更新", state: "active" },
      memory: { action: "W、dW 复制，m、v 分片", object: "owner HBM 多保存 optimizer state", state: "active" },
      fabric: { action: "Owner 发送更新后的 W", object: "每个参数一次 Broadcast", state: "active" },
    },
  },
  fsdp: {
    question: "FSDP 为什么能训练单卡放不下的模型？",
    answer: "长期只保存 shard，计算当前层前临时拼出完整权重，用完立即释放。",
    cta: "查看分片硬件路径",
    targetId: "fsdp-hardware-stage",
    layers: {
      framework: { action: "逐层 pre / post hook", object: "控制 unshard、compute 与 reshard", state: "active" },
      process: { action: "每个 Rank 管理一份 state shard", object: "参数、梯度、m、v 都按 Rank 切分", state: "active" },
      stream: { action: "通信、计算与预取建立依赖", object: "Comm stream + Compute stream", state: "active" },
      kernel: { action: "NCCL + forward / backward kernel", object: "SM 消费当前层完整权重", state: "active" },
      memory: { action: "shard 长驻，full W 临时存在", object: "HBM 峰值由 wrap 与预取决定", state: "active" },
      fabric: { action: "AllGather W，ReduceScatter dW", object: "按层交换参数与梯度 shards", state: "active" },
    },
  },
};

const stateLabels: Record<LayerState, string> = {
  active: "正在参与",
  ready: "已经就绪",
  deferred: "此刻未使用",
};

export function DistributedSystemStack({
  topic,
  onDrillDown,
}: {
  topic: DistributedSystemStackTopic;
  onDrillDown?: () => void;
}) {
  const profile = distributedSystemStackProfiles[topic];
  const drillDown = onDrillDown ?? (() => scrollToSection(profile.targetId));

  return (
    <section className={`distributed-system-stack stack-topic-${topic}`} aria-label="当前专题的跨层系统路径">
      <header>
        <div>
          <span>跨层系统视角</span>
          <h2>{profile.question}</h2>
          <p>{profile.answer}</p>
        </div>
        <button type="button" onClick={drillDown}>
          <HardDrives size={17} weight="duotone" aria-hidden="true" />
          {profile.cta}
          <ArrowDown size={14} aria-hidden="true" />
        </button>
      </header>

      <div className="distributed-system-stack-scroll">
        <div className="distributed-system-layer-track">
          {systemLayers.map(({ id, label, Icon }, index) => {
            const layer = profile.layers[id];
            return (
              <article className={`system-layer-${id} state-${layer.state}`} key={id}>
                <div>
                  <span><Icon size={16} weight="duotone" aria-hidden="true" />{label}</span>
                  <i>{stateLabels[layer.state]}</i>
                </div>
                <strong>{layer.action}</strong>
                <small>{layer.object}</small>
                {index < systemLayers.length - 1 && <b aria-hidden="true" />}
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
