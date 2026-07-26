export type DomainId =
  | "representation"
  | "model"
  | "training"
  | "gpu"
  | "distributed"
  | "inference"
  | "generation";

export type DomainIcon =
  | "database"
  | "circuitry"
  | "graph"
  | "cpu"
  | "network"
  | "lightning"
  | "atom";

export type TopicStatus = "available" | "next" | "mapped";

export interface TopicDefinition {
  id: string;
  title: string;
  description: string;
  status: TopicStatus;
  route?: string;
  prerequisites?: string[];
}

export interface KnowledgeDomain {
  id: DomainId;
  index: string;
  title: string;
  description: string;
  question: string;
  icon: DomainIcon;
  topics: TopicDefinition[];
}

export interface LearningJourney {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  status: "active" | "planned";
  topicIds: string[];
}

export const knowledgeDomains: KnowledgeDomain[] = [
  {
    id: "representation",
    index: "01",
    title: "数据与表示",
    description: "文本如何变成模型可以读取、存储和计算的数字。",
    question: "一句话怎样进入模型？",
    icon: "database",
    topics: [
      { id: "unicode-byte", title: "Unicode 与 Byte", description: "字符最终如何落到字节序列。", status: "mapped" },
      { id: "bpe", title: "BPE", description: "从字节统计出可复用 token。", status: "mapped", prerequisites: ["unicode-byte"] },
      { id: "tokenizer", title: "Tokenizer", description: "编码、解码与特殊 token。", status: "mapped", prerequisites: ["bpe"] },
      { id: "embedding", title: "Embedding", description: "token id 如何变成连续向量。", status: "mapped", prerequisites: ["tokenizer"] },
    ],
  },
  {
    id: "model",
    index: "02",
    title: "模型结构",
    description: "Transformer 内部的算子如何把表示逐层变换。",
    question: "一个 token 如何理解上下文？",
    icon: "circuitry",
    topics: [
      { id: "linear", title: "Linear", description: "矩阵乘法如何改变特征。", status: "mapped", prerequisites: ["embedding"] },
      { id: "attention", title: "Attention", description: "Q、K、V 如何建立 token 关系。", status: "next", prerequisites: ["linear"] },
      { id: "rope", title: "RoPE", description: "位置信息如何进入注意力。", status: "mapped", prerequisites: ["attention"] },
      { id: "rmsnorm", title: "RMSNorm", description: "为什么需要稳定激活尺度。", status: "mapped", prerequisites: ["linear"] },
      { id: "swiglu", title: "SwiGLU", description: "门控 MLP 如何扩展表达能力。", status: "mapped", prerequisites: ["linear"] },
      { id: "transformer", title: "Transformer Block", description: "把注意力、MLP、残差串成一层。", status: "mapped", prerequisites: ["attention", "rope", "rmsnorm", "swiglu"] },
    ],
  },
  {
    id: "training",
    index: "03",
    title: "训练机制",
    description: "损失如何沿计算图返回，并持续改变模型参数。",
    question: "模型为什么会越训越好？",
    icon: "graph",
    topics: [
      { id: "autograd", title: "Autograd", description: "计算图如何记录并反向传播。", status: "next", prerequisites: ["linear"] },
      { id: "loss", title: "Loss 与 Cross Entropy", description: "预测误差如何变成标量目标。", status: "mapped", prerequisites: ["transformer"] },
      { id: "gradient", title: "Gradient", description: "参数变化方向从哪里来。", status: "next", prerequisites: ["autograd", "loss"] },
      { id: "adamw", title: "AdamW", description: "一阶、二阶动量与权重衰减。", status: "next", prerequisites: ["gradient"] },
      { id: "lr-schedule", title: "Learning Rate", description: "学习率为何需要 warmup 与衰减。", status: "mapped", prerequisites: ["adamw"] },
      { id: "gradient-clipping", title: "Gradient Clipping", description: "如何限制异常大的更新。", status: "mapped", prerequisites: ["gradient"] },
      { id: "checkpoint", title: "Checkpoint", description: "训练状态怎样保存和恢复。", status: "mapped", prerequisites: ["adamw"] },
    ],
  },
  {
    id: "gpu",
    index: "04",
    title: "GPU 与性能",
    description: "同一段计算如何落到 Kernel、SM、Warp 与存储层级。",
    question: "一行 PyTorch 最终在哪里执行？",
    icon: "cpu",
    topics: [
      { id: "gpu-architecture", title: "GPU、SM 与 Warp", description: "线程怎样分组并占用执行资源。", status: "next" },
      { id: "cuda-kernel", title: "CUDA Kernel", description: "算子如何变成 GPU 上的任务。", status: "next", prerequisites: ["gpu-architecture"] },
      { id: "cuda-stream", title: "CUDA Stream", description: "计算与通信怎样排队和重叠。", status: "next", prerequisites: ["cuda-kernel"] },
      { id: "memory-hierarchy", title: "HBM、SRAM 与 Cache", description: "数据在哪一层以及搬运代价。", status: "next", prerequisites: ["gpu-architecture"] },
      { id: "mixed-precision", title: "Mixed Precision", description: "精度、吞吐与稳定性的权衡。", status: "mapped", prerequisites: ["cuda-kernel"] },
      { id: "flash-attention", title: "FlashAttention", description: "用分块减少 HBM 读写。", status: "next", prerequisites: ["attention", "memory-hierarchy"] },
      { id: "roofline", title: "Roofline", description: "判断算子受计算还是带宽限制。", status: "mapped", prerequisites: ["memory-hierarchy"] },
    ],
  },
  {
    id: "distributed",
    index: "05",
    title: "分布式训练",
    description: "多进程、多 GPU 如何分工计算并保持模型状态一致。",
    question: "一张卡放不下、算不快怎么办？",
    icon: "network",
    topics: [
      { id: "process-rank", title: "Process 与 Rank", description: "任务身份和通信组如何建立。", status: "next" },
      { id: "collective", title: "集合通信", description: "Broadcast、Reduce、Gather 的共同抽象。", status: "next", prerequisites: ["process-rank"] },
      { id: "ring-allreduce", title: "Ring AllReduce", description: "Reduce-Scatter 与 All-Gather 如何拼成归约。", status: "next", prerequisites: ["collective"] },
      { id: "ddp", title: "DDP", description: "复制模型、同步梯度、并行计算。", status: "available", route: "/distributed/ddp", prerequisites: ["gradient", "cuda-stream", "ring-allreduce"] },
      { id: "zero-1", title: "ZeRO-1", description: "只分片 optimizer state。", status: "next", prerequisites: ["ddp", "adamw"] },
      { id: "fsdp", title: "FSDP", description: "参数按需 All-Gather，梯度 Reduce-Scatter。", status: "next", prerequisites: ["zero-1", "collective"] },
      { id: "tensor-parallel", title: "Tensor Parallel", description: "把一个算子的矩阵切到多卡。", status: "mapped", prerequisites: ["collective", "linear"] },
      { id: "pipeline-parallel", title: "Pipeline Parallel", description: "把模型层切成流水线阶段。", status: "mapped", prerequisites: ["transformer", "process-rank"] },
      { id: "expert-parallel", title: "Expert Parallel", description: "MoE expert 如何分布与路由。", status: "mapped", prerequisites: ["collective"] },
    ],
  },
  {
    id: "inference",
    index: "06",
    title: "推理系统",
    description: "模型训练完成后，怎样以更低延迟和更高吞吐生成。",
    question: "模型如何服务大量请求？",
    icon: "lightning",
    topics: [
      { id: "prefill-decode", title: "Prefill 与 Decode", description: "推理为何分成两个计算阶段。", status: "mapped", prerequisites: ["transformer"] },
      { id: "kv-cache", title: "KV Cache", description: "用显存保存历史注意力状态。", status: "next", prerequisites: ["attention", "prefill-decode"] },
      { id: "continuous-batching", title: "Continuous Batching", description: "不同长度请求如何动态合批。", status: "mapped", prerequisites: ["prefill-decode"] },
      { id: "paged-attention", title: "Paged Attention", description: "离散管理 KV Cache，减少碎片。", status: "mapped", prerequisites: ["kv-cache", "memory-hierarchy"] },
      { id: "quantization", title: "Quantization", description: "用更少位数存储和计算权重。", status: "mapped", prerequisites: ["mixed-precision"] },
      { id: "speculative-decoding", title: "Speculative Decoding", description: "让小模型帮助大模型并行猜 token。", status: "mapped", prerequisites: ["prefill-decode"] },
    ],
  },
  {
    id: "generation",
    index: "07",
    title: "生成与多模态",
    description: "图像、视频和跨模态模型如何表示、加噪与生成内容。",
    question: "文本之外的内容怎样被生成？",
    icon: "atom",
    topics: [
      { id: "vae", title: "VAE", description: "在紧凑潜空间里编码与重建。", status: "mapped" },
      { id: "diffusion", title: "Diffusion", description: "从噪声逐步恢复样本。", status: "next", prerequisites: ["vae"] },
      { id: "dit", title: "DiT", description: "用 Transformer 预测扩散噪声。", status: "mapped", prerequisites: ["diffusion", "transformer"] },
      { id: "clip", title: "CLIP", description: "把文本和图像对齐到同一空间。", status: "mapped", prerequisites: ["embedding"] },
      { id: "vit", title: "ViT", description: "把图像 patch 当作 token。", status: "mapped", prerequisites: ["transformer"] },
      { id: "multimodal", title: "多模态生成", description: "连接视觉编码器、LLM 与生成器。", status: "mapped", prerequisites: ["clip", "vit", "diffusion"] },
    ],
  },
];

export const topicIndex = new Map(
  knowledgeDomains.flatMap((domain) =>
    domain.topics.map((topic) => [topic.id, { ...topic, domainId: domain.id }] as const),
  ),
);

export const learningJourneys: LearningJourney[] = [
  {
    id: "gradient-to-cluster",
    eyebrow: "当前主线",
    title: "跟着一个梯度，走到多机多卡",
    description: "从反向传播产生梯度开始，依次理解 GPU 执行、集合通信、DDP、ZeRO 与 FSDP。",
    status: "active",
    topicIds: ["gradient", "gpu-architecture", "cuda-stream", "collective", "ring-allreduce", "ddp", "zero-1", "fsdp"],
  },
  {
    id: "text-to-context",
    eyebrow: "模型主线",
    title: "从一句文本，到 Transformer 上下文",
    description: "沿 token 的数据形态理解 BPE、Embedding、Attention、RoPE 与完整 Transformer Block。",
    status: "planned",
    topicIds: ["unicode-byte", "bpe", "tokenizer", "embedding", "attention", "rope", "transformer"],
  },
  {
    id: "prompt-to-serving",
    eyebrow: "推理主线",
    title: "从一个请求，到流式输出 token",
    description: "拆解 Prefill、Decode、KV Cache、动态批处理与显存管理如何共同决定推理性能。",
    status: "planned",
    topicIds: ["prefill-decode", "kv-cache", "continuous-batching", "paged-attention", "quantization"]
  },
];

export const topicCount = knowledgeDomains.reduce(
  (count, domain) => count + domain.topics.length,
  0,
);

export function getTopic(topicId: string): TopicDefinition | undefined {
  return topicIndex.get(topicId);
}
