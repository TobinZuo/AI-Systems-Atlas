import { simulateGradientScaleKernel, type GPUKernelSimulation } from "./gpuArchitecture";

export type KernelDevice = "cpu" | "cuda";
export type KernelFault = "none" | "invalid-launch" | "device-memory";
export type KernelStepStatus = "executed" | "bypassed" | "fault" | "blocked";
export type KernelStepId =
  | "python-call"
  | "operator-schema"
  | "dispatcher"
  | "backend-kernel"
  | "runtime-launch"
  | "stream-enqueue"
  | "device-execute"
  | "error-observation"
  | "result-visible";

export interface KernelJourneyConfig {
  device: KernelDevice;
  vectorLength: number;
  threadsPerBlock: number;
  scale: number;
  fault: KernelFault;
}

export interface KernelJourneyStep {
  id: KernelStepId;
  label: string;
  compactLabel: string;
  layer: "python" | "pytorch" | "backend" | "runtime" | "queue" | "device" | "observation";
  call: string;
  explanation: string;
  reads: string[];
  writes: string[];
  status: KernelStepStatus;
}

export interface KernelArgument {
  name: string;
  hostValue: string;
  meaning: string;
  kind: "pointer" | "scalar" | "configuration";
}

export interface KernelJourneySimulation {
  config: KernelJourneyConfig;
  steps: KernelJourneyStep[];
  gpu: GPUKernelSimulation | null;
  selectedBackend: "CPU kernel" | "CUDA kernel";
  effectiveThreadsPerBlock: number;
  gridDim: number | null;
  launchedThreadCount: number;
  usefulThreadCount: number;
  maskedThreadCount: number;
  arguments: KernelArgument[];
  hostReturnAt: number;
  deviceCompleteAt: number;
  hostReturnsBeforeCompletion: boolean;
  launchAccepted: boolean;
  resultValid: boolean;
  errorCode: string | null;
  errorObservedAt: "immediate launch check" | "later synchronization" | null;
}

const GRAD_POINTER = "0x7f40_1000";
const OUTPUT_POINTER = "0x7f40_2000";

function assertConfig(config: KernelJourneyConfig): void {
  if (!Number.isInteger(config.vectorLength) || config.vectorLength <= 0 || config.vectorLength > 4096) {
    throw new RangeError("vectorLength must be an integer from 1 to 4096");
  }
  if (!Number.isInteger(config.threadsPerBlock) || config.threadsPerBlock <= 0 || config.threadsPerBlock > 1024) {
    throw new RangeError("threadsPerBlock must be an integer from 1 to 1024");
  }
  if (!Number.isFinite(config.scale)) throw new RangeError("scale must be finite");
  if (config.device === "cpu" && config.fault !== "none") {
    throw new RangeError("CUDA faults require a CUDA tensor");
  }
}

function step(
  id: KernelStepId,
  label: string,
  compactLabel: string,
  layer: KernelJourneyStep["layer"],
  call: string,
  explanation: string,
  reads: string[],
  writes: string[],
  status: KernelStepStatus = "executed",
): KernelJourneyStep {
  return { id, label, compactLabel, layer, call, explanation, reads, writes, status };
}

export function simulateKernelJourney(config: KernelJourneyConfig): KernelJourneySimulation {
  assertConfig(config);
  const isCuda = config.device === "cuda";
  const invalidLaunch = isCuda && config.fault === "invalid-launch";
  const deviceMemoryFault = isCuda && config.fault === "device-memory";
  const effectiveThreadsPerBlock = invalidLaunch ? 2048 : config.threadsPerBlock;
  const gridDim = isCuda ? Math.ceil(config.vectorLength / effectiveThreadsPerBlock) : null;
  const launchedThreadCount = gridDim === null ? config.vectorLength : gridDim * effectiveThreadsPerBlock;
  const usefulThreadCount = invalidLaunch ? 0 : config.vectorLength;
  const maskedThreadCount = invalidLaunch ? 0 : Math.max(0, launchedThreadCount - usefulThreadCount);
  const gpu = isCuda && !invalidLaunch
    ? simulateGradientScaleKernel({
        vectorLength: config.vectorLength,
        threadsPerBlock: config.threadsPerBlock,
        smCount: 2,
        scale: config.scale,
      })
    : null;

  const cudaBlockedStatus: KernelStepStatus = invalidLaunch ? "blocked" : "executed";
  const resultValid = !invalidLaunch && !deviceMemoryFault;
  const steps: KernelJourneyStep[] = [
    step(
      "python-call",
      "Python 发起 Tensor 运算",
      "Python",
      "python",
      `scaled = grad.mul(${config.scale})`,
      "Python 只表达要做什么。真正逐元素计算由输入 Tensor 的设备实现决定。",
      [`grad: ${config.device}:${config.vectorLength}`],
      ["operator call"],
    ),
    step(
      "operator-schema",
      "解析统一的 ATen operator schema",
      "ATen schema",
      "pytorch",
      "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
      "Schema 规定参数和返回值，不绑定 CPU 或 CUDA 实现。不同后端共享同一个算子语义。",
      ["operator name", "Tensor metadata"],
      ["typed operator handle"],
    ),
    step(
      "dispatcher",
      `Dispatcher 选择 ${isCuda ? "CUDA" : "CPU"} backend`,
      "Dispatcher",
      "pytorch",
      `DispatchKeySet = {${isCuda ? "CUDA" : "CPU"}}`,
      "Dispatcher 检查 Tensor device 与线程局部状态，把统一算子路由到已注册的后端 kernel。",
      ["Tensor device", "dtype", "dispatch keys"],
      [isCuda ? "CUDA implementation" : "CPU implementation"],
    ),
    step(
      "backend-kernel",
      isCuda ? "CUDA backend 准备 pointwise kernel" : "CPU backend 执行向量循环",
      isCuda ? "CUDA impl" : "CPU loop",
      "backend",
      isCuda ? "prepare_pointwise_kernel(grad, scale)" : "parallel_for(0, n, grad[i] * scale)",
      isCuda
        ? "后端读取 shape、stride、dtype 与设备指针，准备 execution configuration 和 kernel 参数。具体内部 helper 会随 PyTorch 版本变化。"
        : "CPU kernel 在主机线程池上读取内存并完成计算，不经过 CUDA Runtime、Stream 或 GPU。",
      ["shape", "stride", "dtype", isCuda ? "device pointer" : "host pointer"],
      [isCuda ? "launch description" : "scaled host tensor"],
    ),
    step(
      "runtime-launch",
      invalidLaunch ? "CUDA Runtime 拒绝非法配置" : "CUDA Runtime 发起 kernel launch",
      "Runtime launch",
      "runtime",
      `pointwise_kernel<<<${gridDim ?? 0}, ${effectiveThreadsPerBlock}, 0, compute_stream>>>(...)`,
      invalidLaunch
        ? "一个 Block 请求 2048 个线程，超过教学设备的 maxThreadsPerBlock=1024。错误属于 launch 配置错误，GPU 不会执行 kernel。"
        : isCuda
          ? "Runtime 把函数入口、execution configuration、标量和显存地址编码成一次 launch。它不会把整块梯度复制进调用参数。"
          : "CPU 路径没有 CUDA launch，这一层被旁路。",
      isCuda ? ["kernel entry", "gridDim", "blockDim", "stream", "arguments"] : [],
      isCuda && !invalidLaunch ? ["launch command"] : [],
      !isCuda ? "bypassed" : invalidLaunch ? "fault" : "executed",
    ),
    step(
      "stream-enqueue",
      "Launch 进入 Compute Stream",
      "Stream queue",
      "queue",
      "enqueue(command, compute_stream)",
      isCuda
        ? "Stream 保存 GPU 工作的先后关系。Host 完成入队后通常可以继续运行，不必等待每个 Lane 算完。"
        : "CPU 路径没有 CUDA Stream。",
      isCuda ? ["launch command"] : [],
      isCuda && !invalidLaunch ? ["ordered GPU work"] : [],
      !isCuda ? "bypassed" : cudaBlockedStatus,
    ),
    step(
      "device-execute",
      deviceMemoryFault ? "GPU 执行时访问非法显存" : "GPU 调度 Block 并执行指令",
      "GPU execute",
      "device",
      deviceMemoryFault ? "grad[n + lane] = value  // invalid" : "if (i < n) out[i] = grad[i] * scale",
      !isCuda
        ? "CPU backend 已经完成结果，这一层被旁路。"
        : deviceMemoryFault
          ? "Launch 本身格式正确，但 kernel 真正运行后才触发非法地址。这个错误可能到后续同步点才被 Host 观察到。"
          : "GPU 从 HBM 读取地址指向的数据。Grid 中每个有效线程处理一个元素，尾部线程被边界判断屏蔽。",
      isCuda && !invalidLaunch ? [GRAD_POINTER, "scale", "n"] : [],
      isCuda && !invalidLaunch ? [deviceMemoryFault ? "partial or undefined HBM state" : OUTPUT_POINTER] : [],
      !isCuda ? "bypassed" : invalidLaunch ? "blocked" : deviceMemoryFault ? "fault" : "executed",
    ),
    step(
      "error-observation",
      invalidLaunch
        ? "Host 立即检查到 launch error"
        : deviceMemoryFault
          ? "同步点暴露异步执行错误"
          : isCuda
            ? "同步点确认 kernel 已完成"
            : "CPU 调用返回时结果已就绪",
      "Observe",
      "observation",
      invalidLaunch
        ? "cudaGetLastError()"
        : isCuda
          ? "torch.cuda.synchronize()"
          : "return scaled",
      invalidLaunch
        ? "Runtime 已经知道 execution configuration 非法，因此 launch 检查可以立即报告。"
        : deviceMemoryFault
          ? "设备执行与 Host 异步。显存访问错误通常需要在同步或后续 CUDA API 边界才能可靠归因。"
          : isCuda
            ? "教学页显式同步便于观察完成点。真实训练通常依赖 Stream/Event 保持顺序，不会每个算子都全设备同步。"
            : "普通 CPU 计算在函数返回前完成。",
      isCuda ? ["CUDA error state", "stream progress"] : ["CPU result"],
      [invalidLaunch || deviceMemoryFault ? "raised RuntimeError" : "completion observed"],
    ),
    step(
      "result-visible",
      resultValid ? "输出 Tensor 可以安全使用" : "本次输出不能作为训练结果",
      "Result",
      "observation",
      resultValid ? "scaled[i] = grad[i] * scale" : "abort current training step",
      resultValid
        ? isCuda
          ? "同一 Stream 的后续 kernel 可以直接消费输出。CPU 若要读取数值，需要等待相关 GPU 工作完成。"
          : "输出位于主机内存，CPU 可以直接读取。"
        : "系统应停止这一步并报告错误，不能把部分写入或未写入的数据继续传给 optimizer。",
      resultValid ? [isCuda ? OUTPUT_POINTER : "host output buffer"] : [],
      resultValid ? ["valid scaled gradient"] : [],
      resultValid ? "executed" : "blocked",
    ),
  ];

  const hostReturnAt = isCuda ? 0.18 : 1.12;
  const deviceCompleteAt = invalidLaunch ? 0.18 : isCuda ? 1.46 : 1.12;

  return {
    config: { ...config },
    steps,
    gpu,
    selectedBackend: isCuda ? "CUDA kernel" : "CPU kernel",
    effectiveThreadsPerBlock,
    gridDim,
    launchedThreadCount,
    usefulThreadCount,
    maskedThreadCount,
    arguments: isCuda
      ? [
          { name: "grad", hostValue: GRAD_POINTER, meaning: "HBM 输入首地址", kind: "pointer" },
          { name: "out", hostValue: OUTPUT_POINTER, meaning: "HBM 输出首地址", kind: "pointer" },
          { name: "n", hostValue: String(config.vectorLength), meaning: "元素数量", kind: "scalar" },
          { name: "scale", hostValue: String(config.scale), meaning: "乘法标量", kind: "scalar" },
          { name: "<<< >>>", hostValue: `${gridDim ?? 0} × ${effectiveThreadsPerBlock}`, meaning: "Grid × Block", kind: "configuration" },
        ]
      : [
          { name: "grad", hostValue: "host buffer", meaning: "CPU 内存地址", kind: "pointer" },
          { name: "n", hostValue: String(config.vectorLength), meaning: "元素数量", kind: "scalar" },
          { name: "scale", hostValue: String(config.scale), meaning: "乘法标量", kind: "scalar" },
        ],
    hostReturnAt,
    deviceCompleteAt,
    hostReturnsBeforeCompletion: hostReturnAt < deviceCompleteAt,
    launchAccepted: !invalidLaunch,
    resultValid,
    errorCode: invalidLaunch
      ? "cudaErrorInvalidConfiguration"
      : deviceMemoryFault
        ? "cudaErrorIllegalAddress"
        : null,
    errorObservedAt: invalidLaunch
      ? "immediate launch check"
      : deviceMemoryFault
        ? "later synchronization"
        : null,
  };
}

export function firstFaultIndex(simulation: KernelJourneySimulation): number | null {
  const index = simulation.steps.findIndex((item) => item.status === "fault");
  return index === -1 ? null : index;
}
