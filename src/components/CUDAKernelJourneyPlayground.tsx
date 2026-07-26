import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { BracketsCurly } from "@phosphor-icons/react/BracketsCurly";
import { Bug } from "@phosphor-icons/react/Bug";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Circuitry } from "@phosphor-icons/react/Circuitry";
import { Code } from "@phosphor-icons/react/Code";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { FlowArrow } from "@phosphor-icons/react/FlowArrow";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Pause } from "@phosphor-icons/react/Pause";
import { Play } from "@phosphor-icons/react/Play";
import { Queue } from "@phosphor-icons/react/Queue";
import { Warning } from "@phosphor-icons/react/Warning";
import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import {
  firstFaultIndex,
  simulateKernelJourney,
  type KernelDevice,
  type KernelFault,
  type KernelJourneyStep,
} from "../playground/cudaKernelJourney";

const vectorLengths = [70, 1000, 4096];
const blockSizes = [32, 64, 128, 256];
const scales = [0.5, 1, 2];

const layerLabels: Record<KernelJourneyStep["layer"], string> = {
  python: "Python front end",
  pytorch: "PyTorch core",
  backend: "Backend implementation",
  runtime: "CUDA Runtime",
  queue: "CUDA Stream",
  device: "GPU device",
  observation: "Completion boundary",
};

const statusLabels: Record<KernelJourneyStep["status"], string> = {
  executed: "执行",
  bypassed: "旁路",
  fault: "报错",
  blocked: "未执行",
};

function LayerIcon({ layer }: { layer: KernelJourneyStep["layer"] }) {
  const icons: Record<KernelJourneyStep["layer"], ReactNode> = {
    python: <Code size={18} weight="duotone" />,
    pytorch: <FlowArrow size={18} weight="duotone" />,
    backend: <FunctionIcon size={18} weight="duotone" />,
    runtime: <BracketsCurly size={18} weight="duotone" />,
    queue: <Queue size={18} weight="duotone" />,
    device: <Circuitry size={18} weight="duotone" />,
    observation: <CheckCircle size={18} weight="duotone" />,
  };
  return icons[layer];
}

export function CUDAKernelJourneyPlayground() {
  const [device, setDevice] = useState<KernelDevice>("cuda");
  const [vectorLength, setVectorLength] = useState(70);
  const [threadsPerBlock, setThreadsPerBlock] = useState(32);
  const [scale, setScale] = useState(0.5);
  const [fault, setFault] = useState<KernelFault>("none");
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const simulation = useMemo(
    () => simulateKernelJourney({ device, vectorLength, threadsPerBlock, scale, fault }),
    [device, vectorLength, threadsPerBlock, scale, fault],
  );
  const currentStep = simulation.steps[stepIndex];
  const faultIndex = firstFaultIndex(simulation);
  const activeGpuStep = currentStep.layer === "device" || currentStep.layer === "queue";
  const timelineStyle = {
    "--host-return": `${(simulation.hostReturnAt / 1.6) * 100}%`,
    "--device-complete": `${(simulation.deviceCompleteAt / 1.6) * 100}%`,
  } as CSSProperties;

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((current) => {
        if (current === simulation.steps.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1100);
    return () => window.clearInterval(timer);
  }, [isPlaying, simulation.steps.length]);

  const selectDevice = (nextDevice: KernelDevice) => {
    setDevice(nextDevice);
    setFault("none");
    setStepIndex(0);
    setIsPlaying(false);
  };

  const selectFault = (nextFault: KernelFault) => {
    setFault(nextFault);
    setIsPlaying(false);
    if (nextFault === "invalid-launch") setStepIndex(4);
    else if (nextFault === "device-memory") setStepIndex(6);
    else setStepIndex(0);
  };

  const selectStep = (index: number) => {
    setStepIndex(index);
    setIsPlaying(false);
  };

  const reset = () => {
    setDevice("cuda");
    setVectorLength(70);
    setThreadsPerBlock(32);
    setScale(0.5);
    setFault("none");
    setStepIndex(0);
    setIsPlaying(false);
  };

  return (
    <section className="cuda-kernel-journey" id="cuda-kernel-journey" aria-labelledby="kernel-journey-title">
      <header className="kernel-journey-header">
        <div>
          <span>Operator dispatch lab</span>
          <h2 id="kernel-journey-title">同一行代码，为什么可能走 CPU，也可能 launch CUDA Kernel</h2>
          <p>切换 Tensor device、线程配置和错误类型。调用链、参数包、GPU 线程与错误暴露位置会一起变化。</p>
        </div>
        <pre aria-label="本页追踪的 Tensor 运算"><code>{`grad = torch.tensor(..., device="${device}"); scaled = grad.mul(${scale})`}</code></pre>
      </header>

      <div className="kernel-config-bar" aria-label="Kernel journey 配置">
        <div className="kernel-device-switch" role="tablist" aria-label="Tensor device">
          <button type="button" role="tab" aria-selected={device === "cuda"} className={device === "cuda" ? "is-active" : ""} onClick={() => selectDevice("cuda")}><Lightning size={16} weight="duotone" />CUDA Tensor</button>
          <button type="button" role="tab" aria-selected={device === "cpu"} className={device === "cpu" ? "is-active" : ""} onClick={() => selectDevice("cpu")}><Cpu size={16} weight="duotone" />CPU Tensor</button>
        </div>
        <label><span>元素数量 n</span><select value={vectorLength} onChange={(event) => setVectorLength(Number(event.target.value))}>{vectorLengths.map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        <label><span>blockDim.x</span><select value={threadsPerBlock} disabled={device === "cpu"} onChange={(event) => setThreadsPerBlock(Number(event.target.value))}>{blockSizes.map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
        <label><span>scale</span><select value={scale} onChange={(event) => setScale(Number(event.target.value))}>{scales.map((value) => <option value={value} key={value}>× {value}</option>)}</select></label>
        <div className="kernel-play-actions">
          <button type="button" className="kernel-play-button" onClick={() => {
            if (!isPlaying && stepIndex === simulation.steps.length - 1) setStepIndex(0);
            setIsPlaying((current) => !current);
          }}>{isPlaying ? <Pause size={16} weight="fill" /> : <Play size={16} weight="fill" />}{isPlaying ? "暂停" : "逐层播放"}</button>
          <button type="button" onClick={reset} aria-label="重置 Kernel 调用链"><ArrowCounterClockwise size={17} /></button>
        </div>
      </div>

      <div className="kernel-fault-selector" aria-label="错误注入">
        <span><Bug size={16} weight="duotone" />错误注入</span>
        <button type="button" className={fault === "none" ? "is-active" : ""} onClick={() => selectFault("none")}>正常执行</button>
        <button type="button" disabled={device === "cpu"} className={fault === "invalid-launch" ? "is-active is-error" : ""} onClick={() => selectFault("invalid-launch")}>非法 launch 配置</button>
        <button type="button" disabled={device === "cpu"} className={fault === "device-memory" ? "is-active is-error" : ""} onClick={() => selectFault("device-memory")}>GPU 非法地址</button>
        <p>{device === "cpu" ? "CPU 路径不经过 CUDA，错误注入已关闭。" : fault === "none" ? "保持正确边界判断，观察正常异步 launch。" : simulation.errorObservedAt === "immediate launch check" ? "配置错误在 GPU 执行前就能检查到。" : "执行错误会在后续同步边界暴露。"}</p>
      </div>

      <div className="kernel-summary-metrics" aria-label="Kernel 调度摘要">
        <div><span>Dispatcher 选择</span><strong>{simulation.selectedBackend}</strong><small>由 Tensor device 决定后端</small></div>
        <div><span>Execution config</span><strong>{simulation.gridDim === null ? "CPU parallel loop" : `${simulation.gridDim} Blocks × ${simulation.effectiveThreadsPerBlock}`}</strong><small>{simulation.gridDim === null ? `${vectorLength} host elements` : `${simulation.launchedThreadCount} CUDA threads`}</small></div>
        <div><span>{device === "cuda" ? "有效线程" : "处理元素"}</span><strong>{simulation.usefulThreadCount} / {simulation.launchedThreadCount}</strong><small>{device === "cuda" ? `${simulation.maskedThreadCount} 个尾部线程被 guard 屏蔽` : "CPU backend 处理全部元素"}</small></div>
        <div className={simulation.resultValid ? "is-valid" : "is-invalid"}><span>最终结果</span><strong>{simulation.resultValid ? "scaled gradient 有效" : simulation.errorCode}</strong><small>{simulation.resultValid ? "可以被后续算子消费" : "训练步骤必须中止"}</small></div>
      </div>

      <div className="kernel-step-rail" role="tablist" aria-label="调用链阶段">
        {simulation.steps.map((item, index) => (
          <button type="button" role="tab" aria-selected={index === stepIndex} className={`${index === stepIndex ? "is-active" : ""}${index < stepIndex ? " is-reached" : ""} status-${item.status}`} onClick={() => selectStep(index)} key={item.id}>
            <span>{index + 1}</span><strong>{item.compactLabel}</strong><small>{statusLabels[item.status]}</small>
          </button>
        ))}
      </div>

      <div className="kernel-journey-layout">
        <nav className="kernel-layer-stack" aria-label="软件到硬件调用栈">
          {simulation.steps.map((item, index) => (
            <button type="button" className={`${index === stepIndex ? "is-active" : ""} status-${item.status}`} onClick={() => selectStep(index)} key={item.id}>
              <i><LayerIcon layer={item.layer} /></i>
              <span><small>{layerLabels[item.layer]}</small><strong>{item.label}</strong></span>
              <code>{statusLabels[item.status]}</code>
            </button>
          ))}
        </nav>

        <section className={`kernel-step-inspector layer-${currentStep.layer} status-${currentStep.status}`} aria-live="polite">
          <header>
            <i><LayerIcon layer={currentStep.layer} /></i>
            <div><span>{layerLabels[currentStep.layer]}</span><h3>{currentStep.label}</h3></div>
            <code>{stepIndex + 1} / {simulation.steps.length}</code>
          </header>
          <pre><code>{currentStep.call}</code></pre>
          <p>{currentStep.explanation}</p>
          <div className="kernel-step-io">
            <div><span>读取</span>{currentStep.reads.length ? currentStep.reads.map((value) => <code key={value}>{value}</code>) : <small>这一层没有读取 CUDA 状态</small>}</div>
            <ArrowRight size={18} aria-hidden="true" />
            <div><span>产生</span>{currentStep.writes.length ? currentStep.writes.map((value) => <code key={value}>{value}</code>) : <small>错误或旁路，没有新状态</small>}</div>
          </div>
          <footer>
            <button type="button" onClick={() => selectStep(Math.max(0, stepIndex - 1))} disabled={stepIndex === 0}><CaretLeft size={16} />上一层</button>
            <strong>{currentStep.status === "fault" ? <><Warning size={16} weight="fill" />错误发生在这一层</> : currentStep.status === "blocked" ? "上游失败，所以没有执行" : currentStep.status === "bypassed" ? "当前 device 不经过这一层" : "本层已满足依赖"}</strong>
            <button type="button" onClick={() => selectStep(Math.min(simulation.steps.length - 1, stepIndex + 1))} disabled={stepIndex === simulation.steps.length - 1}>下一层<CaretRight size={16} /></button>
          </footer>
        </section>
      </div>

      <section className="kernel-dispatch-board" aria-labelledby="kernel-dispatch-title">
        <header><div><span>One schema, many backends</span><h3 id="kernel-dispatch-title">Dispatcher 不是计算单元，它负责找到正确实现</h3></div><code>aten::mul.Scalar</code></header>
        <div className="kernel-dispatch-flow">
          <article className="schema-node"><FunctionIcon size={19} weight="duotone" /><span>Operator schema</span><strong>统一参数与返回值语义</strong><small>不绑定硬件</small></article>
          <ArrowRight size={18} className="dispatch-arrow" aria-hidden="true" />
          <article className="dispatcher-node"><FlowArrow size={20} weight="duotone" /><span>Dispatcher</span><strong>读取 device 与 dispatch keys</strong><small>选择已注册 kernel</small></article>
          <ArrowRight size={18} className="dispatch-arrow" aria-hidden="true" />
          <div className="backend-branches">
            <article className={device === "cpu" ? "is-selected" : ""}><Cpu size={18} weight="duotone" /><span>CPU dispatch key</span><strong>CPU vector kernel</strong></article>
            <article className={device === "cuda" ? "is-selected" : ""}><Lightning size={18} weight="duotone" /><span>CUDA dispatch key</span><strong>CUDA pointwise kernel</strong></article>
          </div>
        </div>
        <p className="kernel-implementation-note"><strong>边界说明：</strong>Dispatcher 的后端选择是稳定概念；TensorIterator、具体 kernel 名称和生成方式属于 PyTorch 实现细节，可能随版本、dtype 和编译模式变化。</p>
      </section>

      <section className="kernel-launch-packet" aria-labelledby="kernel-packet-title">
        <header><div><span>Launch packet</span><h3 id="kernel-packet-title">Host 传的是地址与配置，Kernel 搬运的是地址指向的数据</h3></div><BracketsCurly size={22} weight="duotone" /></header>
        <div className="kernel-packet-layout">
          <div className="kernel-argument-grid">
            {simulation.arguments.map((argument) => <article className={`kind-${argument.kind}`} key={argument.name}><span>{argument.kind}</span><strong>{argument.name}</strong><code>{argument.hostValue}</code><small>{argument.meaning}</small></article>)}
          </div>
          <div className={`kernel-data-reality${device === "cpu" ? " is-cpu" : ""}`}>
            <article><Cpu size={18} weight="duotone" /><span>Host API</span><strong>{device === "cuda" ? "编码 launch command" : "调用 CPU kernel"}</strong></article>
            <ArrowRight size={16} aria-hidden="true" />
            <article><Queue size={18} weight="duotone" /><span>{device === "cuda" ? "Compute Stream" : "Host thread pool"}</span><strong>{device === "cuda" ? "保存命令和依赖" : "分配 CPU 工作"}</strong></article>
            <ArrowRight size={16} aria-hidden="true" />
            <article className={activeGpuStep ? "is-active" : ""}><Database size={18} weight="duotone" /><span>{device === "cuda" ? "GPU HBM" : "Host memory"}</span><strong>读取真实 gradient bytes</strong></article>
          </div>
        </div>
      </section>

      <section className="kernel-thread-map" aria-labelledby="kernel-thread-title">
        <header><div><span>Execution configuration</span><h3 id="kernel-thread-title">Grid 决定要创建多少线程，不决定 GPU 有多少核心</h3></div><code>{simulation.gridDim === null ? "CPU path" : `ceil(${vectorLength} / ${simulation.effectiveThreadsPerBlock}) = ${simulation.gridDim} Blocks`}</code></header>
        {simulation.gpu ? (
          <div className="kernel-block-map">
            {simulation.gpu.blocks.map((block) => <article className={currentStep.layer === "device" ? "is-executing" : ""} key={block.blockId}><span>Block {block.blockId}</span><strong>{block.activeThreadCount}/{simulation.gpu!.blockDim} active</strong><small>SM {block.smId}, {block.warps.length} warp{block.warps.length > 1 ? "s" : ""}</small></article>)}
          </div>
        ) : (
          <div className={`kernel-map-empty${fault === "invalid-launch" ? " is-error" : ""}`}>
            {fault === "invalid-launch" ? <Warning size={22} weight="duotone" /> : <Cpu size={22} weight="duotone" />}
            <strong>{fault === "invalid-launch" ? "Grid 没有进入 GPU" : "CPU 路径不创建 CUDA Grid"}</strong>
            <p>{fault === "invalid-launch" ? "Runtime 在入队前拒绝了 2048 threads/block。" : "同一个 operator schema 被路由到 CPU backend，在主机上完成向量循环。"}</p>
          </div>
        )}
        <footer><span>想观察 Block 怎样驻留到 SM、Warp 怎样发射指令？</span><a href="#/gpu/architecture">下钻 GPU、SM 与 Warp<ArrowRight size={14} /></a></footer>
      </section>

      <section className="kernel-async-boundary" aria-labelledby="kernel-async-title">
        <header><div><span>Host time is not device time</span><h3 id="kernel-async-title">Python 返回，不代表 GPU 已经算完</h3></div><strong>{simulation.hostReturnsBeforeCompletion ? "异步边界存在" : "同步 CPU 路径"}</strong></header>
        <div className={`kernel-clock device-${device}`} style={timelineStyle}>
          <div className="kernel-clock-ruler"><span>0 ms</span><span>0.4</span><span>0.8</span><span>1.2</span><span>1.6 ms</span></div>
          <div className="kernel-clock-lane host-lane"><span>Host</span><div><i className="host-work" /><b className="host-return-marker">API return</b></div></div>
          <div className="kernel-clock-lane device-lane"><span>{device === "cuda" ? "GPU" : "CPU"}</span><div><i className={faultIndex === 6 ? "device-work is-error" : "device-work"} /><b className="device-complete-marker">{simulation.errorObservedAt === "immediate launch check" ? "launch rejected" : simulation.errorObservedAt === "later synchronization" ? "fault surfaces" : "complete"}</b></div></div>
        </div>
        <p className="kernel-clock-note">时间刻度只表达 Host 返回与设备完成的先后关系，不是 profiler 实测延迟。</p>
        <div className={`kernel-error-verdict${simulation.errorCode ? " is-error" : ""}`}>
          {simulation.errorCode ? <Warning size={19} weight="fill" /> : <CheckCircle size={19} weight="fill" />}
          <div><strong>{simulation.errorCode ?? "No CUDA error"}</strong><p>{simulation.errorObservedAt === "immediate launch check" ? "Launch configuration 在入队时就不合法，立即检查可定位到 launch。" : simulation.errorObservedAt === "later synchronization" ? "Launch 成功不等于执行成功。非法地址在 GPU 真正访问显存后，于后续同步边界暴露。" : simulation.hostReturnsBeforeCompletion ? "Host 已经继续执行，Compute Stream 仍负责保持后续 GPU 工作的顺序。" : "CPU kernel 在调用返回前完成，Host 与计算完成点重合。"}</p></div>
        </div>
      </section>

      <section className="kernel-contracts" aria-labelledby="kernel-contracts-title">
        <header><h3 id="kernel-contracts-title">把调用链压缩成四个系统概念</h3></header>
        <div>
          <article><span>Operator</span><strong>先定义统一语义</strong><p>Schema 说明输入输出是什么，让调用方不需要硬编码每一种设备实现。</p></article>
          <article><span>Dispatch</span><strong>运行时选择后端</strong><p>系统根据 Tensor device 以及 autograd、autocast 等上下文，把同一个算子路由到合适 kernel。</p></article>
          <article><span>Launch</span><strong>传命令、地址与配置</strong><p>Host 不逐元素发送梯度。GPU kernel 根据指针访问 HBM 中的真实数据。</p></article>
          <article><span>Async</span><strong>提交与完成分离</strong><p>异步提高流水化机会，也要求 Stream、Event 和同步边界承担正确性与排障。</p></article>
        </div>
      </section>

      <footer className="kernel-reference-footer">
        <strong>官方依据</strong>
        <a href="https://docs.pytorch.org/tutorials/advanced/dispatcher.html" target="_blank" rel="noreferrer">PyTorch Dispatcher 与 backend kernels</a>
        <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html" target="_blank" rel="noreferrer">CUDA Kernel 与 execution configuration</a>
        <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html" target="_blank" rel="noreferrer">CUDA Runtime 同步与异步行为</a>
        <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html" target="_blank" rel="noreferrer">CUDA error reporting</a>
      </footer>
    </section>
  );
}
