import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Gauge } from "@phosphor-icons/react/Gauge";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { useMemo, useState } from "react";
import {
  adamWGradientSchedule,
  adamWParameters,
  defaultAdamWConfig,
  getAdamWMemoryBreakdown,
  simulateAdamW,
  type AdamWConfig,
  type AdamWMemoryStrategy,
  type AdamWMode,
  type AdamWParameterId,
} from "../playground/adamw";

type HardwarePhase = "ready" | "read" | "moments" | "update" | "write";

const beta1Values = [0, 0.9, 0.99];
const beta2Values = [0, 0.9, 0.999];
const learningRates = [0.001, 0.01, 0.1];
const weightDecays = [0, 0.01, 0.1];
const epsilonValues = [1e-8, 1e-4];
const hardwarePhases: Array<{ id: HardwarePhase; label: string; detail: string }> = [
  { id: "ready", label: "State ready", detail: "参数、梯度、m、v 已经在 HBM" },
  { id: "read", label: "Kernel read", detail: "SM 读取四组显存地址" },
  { id: "moments", label: "Update m, v", detail: "寄存器中计算移动平均和偏差修正" },
  { id: "update", label: "Update θ", detail: "合并自适应方向和独立衰减" },
  { id: "write", label: "Write back", detail: "把 θ、m、v 写回原显存" },
];

function formatNumber(value: number, digits = 6): string {
  if (Math.abs(value) < 1e-10) return "0";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 1e-5)) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}

function formatGB(bytes: number): string {
  return `${formatNumber(bytes / 1_000_000_000, 2)} GB`;
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

export function AdamWPlayground() {
  const [config, setConfig] = useState<AdamWConfig>(defaultAdamWConfig);
  const [selectedStep, setSelectedStep] = useState(1);
  const [selectedParameter, setSelectedParameter] = useState<AdamWParameterId>("weight-0");
  const [hardwarePhase, setHardwarePhase] = useState<HardwarePhase>("ready");
  const [memoryStrategy, setMemoryStrategy] = useState<AdamWMemoryStrategy>("ddp");

  const simulation = useMemo(() => simulateAdamW(config), [config]);
  const adamwComparison = useMemo(() => simulateAdamW({ ...config, mode: "adamw" }), [config]);
  const coupledComparison = useMemo(() => simulateAdamW({ ...config, mode: "coupled-l2" }), [config]);
  const memory = useMemo(() => getAdamWMemoryBreakdown(memoryStrategy, 1_000_000_000, 4), [memoryStrategy]);
  const trace = simulation.steps[selectedStep - 1].parameters.find((item) => item.id === selectedParameter)!;
  const adamwTrace = adamwComparison.steps[selectedStep - 1].parameters.find((item) => item.id === selectedParameter)!;
  const coupledTrace = coupledComparison.steps[selectedStep - 1].parameters.find((item) => item.id === selectedParameter)!;
  const phaseIndex = hardwarePhases.findIndex((phase) => phase.id === hardwarePhase);
  const selectedHistory = simulation.steps.map((step) => step.parameters.find((item) => item.id === selectedParameter)!);

  const updateConfig = <K extends keyof AdamWConfig>(key: K, value: AdamWConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  const reset = () => {
    setConfig(defaultAdamWConfig);
    setSelectedStep(1);
    setSelectedParameter("weight-0");
    setHardwarePhase("ready");
    setMemoryStrategy("ddp");
  };

  return (
    <section className="adamw-playground" id="adamw-lab" aria-label="AdamW 交互实验台">
      <header className="adamw-playground-header">
        <div>
          <span>Executable optimizer state machine</span>
          <h2>一个 step 不是 `parameter -= lr × grad`</h2>
          <p>先选 step 和参数，再沿公式与硬件路径观察同一组实时数值。</p>
        </div>
        <div className="adamw-live-state" aria-label="当前 AdamW 状态">
          <span><small>step</small><strong>{selectedStep} / 5</strong></span>
          <span><small>parameter</small><strong>{trace.label}</strong></span>
          <span><small>current grad</small><strong>{formatNumber(trace.gradient)}</strong></span>
          <span><small>θ after</small><strong>{formatNumber(trace.parameterAfter)}</strong></span>
        </div>
        <button type="button" className="adamw-reset" onClick={reset}><ArrowCounterClockwise size={15} aria-hidden="true" />重置</button>
      </header>

      <div className="adamw-control-deck">
        <fieldset><legend>Optimizer</legend><div>{(["adamw", "coupled-l2"] as AdamWMode[]).map((mode) => <button type="button" className={config.mode === mode ? "is-active" : ""} aria-pressed={config.mode === mode} onClick={() => updateConfig("mode", mode)} key={mode}>{mode === "adamw" ? "AdamW" : "Adam + L2"}</button>)}</div></fieldset>
        <fieldset><legend>β₁</legend><div>{beta1Values.map((value) => <button type="button" className={config.beta1 === value ? "is-active" : ""} aria-pressed={config.beta1 === value} onClick={() => updateConfig("beta1", value)} key={value}>{value}</button>)}</div></fieldset>
        <fieldset><legend>β₂</legend><div>{beta2Values.map((value) => <button type="button" className={config.beta2 === value ? "is-active" : ""} aria-pressed={config.beta2 === value} onClick={() => updateConfig("beta2", value)} key={value}>{value}</button>)}</div></fieldset>
        <fieldset><legend>Learning rate</legend><div>{learningRates.map((value) => <button type="button" className={config.learningRate === value ? "is-active" : ""} aria-pressed={config.learningRate === value} onClick={() => updateConfig("learningRate", value)} key={value}>{value}</button>)}</div></fieldset>
        <fieldset><legend>Weight decay</legend><div>{weightDecays.map((value) => <button type="button" className={config.weightDecay === value ? "is-active" : ""} aria-pressed={config.weightDecay === value} onClick={() => updateConfig("weightDecay", value)} key={value}>{value}</button>)}</div></fieldset>
        <fieldset><legend>Epsilon</legend><div>{epsilonValues.map((value) => <button type="button" className={config.epsilon === value ? "is-active" : ""} aria-pressed={config.epsilon === value} onClick={() => updateConfig("epsilon", value)} key={value}>{value === 1e-8 ? "1e-8" : "1e-4"}</button>)}</div></fieldset>
      </div>

      <div className="adamw-selection-bar">
        <div><span>选择训练 step</span><div role="group" aria-label="选择训练 step">{adamWGradientSchedule.map((_, index) => <button type="button" className={selectedStep === index + 1 ? "is-active" : ""} aria-pressed={selectedStep === index + 1} onClick={() => setSelectedStep(index + 1)} key={index}>t{index + 1}</button>)}</div></div>
        <div><span>聚焦参数</span><div role="group" aria-label="选择参数">{adamWParameters.map((parameter) => <button type="button" className={`${selectedParameter === parameter.id ? "is-active" : ""} ${parameter.decayApplied ? "is-decay" : "is-no-decay"}`} aria-pressed={selectedParameter === parameter.id} onClick={() => setSelectedParameter(parameter.id)} key={parameter.id}>{parameter.label}<small>{parameter.decayApplied ? "decay" : "no decay"}</small></button>)}</div></div>
      </div>

      <section className="adamw-formula-stage" aria-labelledby="adamw-formula-title">
        <header><div><span>One scalar, exact arithmetic</span><h3 id="adamw-formula-title">先只看 {trace.label} 在第 {selectedStep} 步发生了什么</h3></div><code>θ{selectedStep - 1} = {formatNumber(trace.parameterBefore)}，g{selectedStep} = {formatNumber(trace.gradient)}</code></header>
        <div className="adamw-formula-flow">
          <article className="is-gradient"><span>本轮输入</span><strong>g = {formatNumber(trace.gradient)}</strong><code>{config.mode === "coupled-l2" && trace.decayApplied ? `g + λθ = ${formatNumber(trace.momentInput)}` : "moment input = raw grad"}</code></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article className="is-moment"><span>一阶动量</span><strong>m = {formatNumber(trace.expAvg)}</strong><code>{config.beta1} × {formatNumber(trace.expAvgBefore)} + {formatNumber(1 - config.beta1)} × {formatNumber(trace.momentInput)}</code></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article className="is-variance"><span>二阶动量</span><strong>v = {formatNumber(trace.expAvgSq)}</strong><code>{config.beta2} × {formatNumber(trace.expAvgSqBefore)} + {formatNumber(1 - config.beta2)} × input²</code></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article className="is-corrected"><span>偏差修正</span><strong>m̂ / (√v̂ + ε)</strong><code>{formatNumber(trace.adaptiveDirection)}</code></article>
        </div>
        <div className="adamw-update-composer">
          <article><span>旧参数</span><strong>{formatNumber(trace.parameterBefore)}</strong></article>
          <i>+</i>
          <article className="is-adaptive"><span>自适应更新</span><strong>{signed(trace.gradientUpdate)}</strong><small>-lr × normalized grad</small></article>
          <i>+</i>
          <article className="is-decay"><span>独立权重衰减</span><strong>{signed(trace.decayUpdate)}</strong><small>{trace.decayApplied ? (config.mode === "adamw" ? "-lr × λ × θ" : "已混入 moment") : "本参数组禁用 decay"}</small></article>
          <i>=</i>
          <article className="is-result"><span>新参数</span><strong>{formatNumber(trace.parameterAfter)}</strong></article>
        </div>
        <footer><CheckCircle size={18} weight="fill" aria-hidden="true" /><p><strong>偏差修正解决冷启动。</strong>m、v 从 0 开始，早期会系统性偏小；除以 <code>1-β₁ᵗ</code> 与 <code>1-β₂ᵗ</code> 后，第一步就恢复正确尺度。</p></footer>
      </section>

      <section className="adamw-history-stage" aria-labelledby="adamw-history-title">
        <header><div><span>Persistent state across steps</span><h3 id="adamw-history-title">`.grad` 每步替换，m 与 v 跨 step 留在 optimizer state</h3></div><Gauge size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="adamw-history-grid">
          {selectedHistory.map((item) => {
            const maxMagnitude = Math.max(1, ...selectedHistory.flatMap((history) => [Math.abs(history.gradient), Math.abs(history.expAvg), Math.sqrt(history.expAvgSq)]));
            return <button type="button" className={selectedStep === item.step ? "is-active" : ""} onClick={() => setSelectedStep(item.step)} key={item.step}>
              <header><strong>step {item.step}</strong><span>θ {formatNumber(item.parameterBefore)} → {formatNumber(item.parameterAfter)}</span></header>
              <div><span>grad</span><i className={item.gradient < 0 ? "is-negative" : ""} style={{ "--bar": `${Math.abs(item.gradient) / maxMagnitude * 100}%` } as React.CSSProperties} /><code>{signed(item.gradient)}</code></div>
              <div><span>m</span><i className={item.expAvg < 0 ? "is-negative" : ""} style={{ "--bar": `${Math.abs(item.expAvg) / maxMagnitude * 100}%` } as React.CSSProperties} /><code>{signed(item.expAvg)}</code></div>
              <div><span>√v</span><i style={{ "--bar": `${Math.sqrt(item.expAvgSq) / maxMagnitude * 100}%` } as React.CSSProperties} /><code>{formatNumber(Math.sqrt(item.expAvgSq))}</code></div>
            </button>;
          })}
        </div>
        <aside><strong>为什么要两个历史量？</strong><span><b>m</b> 平滑方向，减少一轮噪声带来的摇摆。</span><span><b>v</b> 记录近期梯度平方尺度，让长期大梯度方向走得更谨慎。</span></aside>
      </section>

      <section className="adamw-decoupling-stage" aria-labelledby="adamw-decoupling-title">
        <header><div><span>Decoupled weight decay</span><h3 id="adamw-decoupling-title">AdamW 与 Adam + L2 的差别发生在 m、v 之前</h3></div><FunctionIcon size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="adamw-compare-grid">
          <article className="is-adamw"><header><span>AdamW</span><strong>衰减绕过 m 与 v</strong></header><div><code>moment input</code><strong>{formatNumber(adamwTrace.momentInput)}</strong><small>始终是 loss gradient</small></div><div><code>m</code><strong>{formatNumber(adamwTrace.expAvg)}</strong></div><div><code>v</code><strong>{formatNumber(adamwTrace.expAvgSq)}</strong></div><footer><span>θ after</span><strong>{formatNumber(adamwTrace.parameterAfter)}</strong></footer></article>
          <div className="adamw-compare-divider"><span>同一初值与 gradient schedule</span><i /><strong>路径逐步分叉</strong></div>
          <article className="is-l2"><header><span>Adam + L2</span><strong>λθ 先混入梯度</strong></header><div><code>moment input</code><strong>{formatNumber(coupledTrace.momentInput)}</strong><small>{trace.decayApplied ? "g + λθ" : "no-decay 组仍是 raw grad"}</small></div><div><code>m</code><strong>{formatNumber(coupledTrace.expAvg)}</strong></div><div><code>v</code><strong>{formatNumber(coupledTrace.expAvgSq)}</strong></div><footer><span>θ after</span><strong>{formatNumber(coupledTrace.parameterAfter)}</strong></footer></article>
        </div>
        <p className="adamw-decoupling-note">在普通 SGD 中，L2 penalty 与 weight decay 可以等价；进入 Adam 的逐元素自适应缩放后，<code>λθ</code> 一旦混入 m、v，就会被历史梯度尺度改变。AdamW 把收缩参数与优化 loss 两件事分开。</p>
      </section>

      <section className="adamw-param-group-stage" aria-labelledby="adamw-group-title">
        <header><div><span>Optimizer parameter groups</span><h3 id="adamw-group-title">同一个 optimizer，可以给不同参数不同超参数</h3></div><Stack size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="adamw-group-code"><code><span>optimizer = AdamW([</span><br />&nbsp;&nbsp;{'{"params": [weight], "weight_decay": '}{config.weightDecay}{'}'},<br />&nbsp;&nbsp;{'{"params": [bias], "weight_decay": 0.0}'},<br /><span>], lr={config.learningRate})</span></code></div>
        <div className="adamw-group-cards">{adamWParameters.map((parameter) => {
          const item = simulation.steps[selectedStep - 1].parameters.find((candidate) => candidate.id === parameter.id)!;
          return <button type="button" className={`${selectedParameter === parameter.id ? "is-active" : ""} ${parameter.decayApplied ? "is-decay" : "is-no-decay"}`} onClick={() => setSelectedParameter(parameter.id)} key={parameter.id}><header><strong>{parameter.label}</strong><span>{parameter.tensorName}</span></header><div><small>group</small><strong>{parameter.group}</strong></div><div><small>effective decay</small><strong>{parameter.decayApplied ? config.weightDecay : 0}</strong></div><footer>Δθ decay = {signed(item.decayUpdate)}</footer></button>;
        })}</div>
        <p>这是常见教学配置，用来明确 parameter group 的语义，不代表 bias 与 normalization 参数在所有模型中都必须禁用 decay。</p>
      </section>

      <section className="adamw-hardware-stage" aria-labelledby="adamw-hardware-title">
        <header><div><span>CUDA execution and HBM residency</span><h3 id="adamw-hardware-title">optimizer.step() 最终是一次读写多组显存的 GPU 工作</h3></div><HardDrives size={22} weight="duotone" aria-hidden="true" /></header>
        <nav className="adamw-hardware-rail" aria-label="AdamW 硬件执行阶段">{hardwarePhases.map((phase, index) => <button type="button" className={`${hardwarePhase === phase.id ? "is-active" : ""}${index < phaseIndex ? " is-complete" : ""}`} aria-pressed={hardwarePhase === phase.id} onClick={() => setHardwarePhase(phase.id)} key={phase.id}><small>0{index + 1}</small><span><strong>{phase.label}</strong><i>{phase.detail}</i></span></button>)}</nav>
        <div className={`adamw-hardware-flow phase-${hardwarePhase}`}>
          <article className="adamw-host-node"><Cpu size={20} weight="duotone" /><span>Rank 0 CPU process</span><strong>optimizer.step()</strong><small>准备 tensor 地址与超参数，向 CUDA stream 提交工作</small></article>
          <ArrowRight size={19} aria-hidden="true" />
          <article className="adamw-stream-node"><Lightning size={20} weight="duotone" /><span>CUDA compute stream</span><strong>AdamW kernel 排队</strong><small>等待 backward 写完 grad 的 event 依赖</small></article>
          <ArrowRight size={19} aria-hidden="true" />
          <article className="adamw-sm-node"><Stack size={20} weight="duotone" /><span>GPU SM</span><strong>load → arithmetic → store</strong><small>{hardwarePhase === "moments" ? "寄存器正在计算 m、v 与 bias correction" : hardwarePhase === "update" ? "寄存器正在合并 adaptive update 与 decay" : "线程按元素处理 parameter state"}</small></article>
          <ArrowRight size={19} aria-hidden="true" />
          <div className="adamw-hbm-node"><header><Database size={20} weight="duotone" /><span>GPU HBM</span><strong>{trace.label} 对应地址</strong></header><div>
            <article className={`${phaseIndex >= 1 ? "is-read" : ""}${phaseIndex === 4 ? " is-written" : ""}`}><span>parameter θ</span><strong>{phaseIndex === 4 ? formatNumber(trace.parameterAfter) : formatNumber(trace.parameterBefore)}</strong><small>{phaseIndex === 4 ? "new value" : "persistent tensor"}</small></article>
            <article className={phaseIndex >= 1 ? "is-read" : ""}><span>parameter.grad</span><strong>{formatNumber(trace.gradient)}</strong><small>本轮输入，随后 zero_grad</small></article>
            <article className={`${phaseIndex >= 1 ? "is-read" : ""}${phaseIndex === 4 ? " is-written" : ""}`}><span>state.exp_avg</span><strong>{phaseIndex === 4 ? formatNumber(trace.expAvg) : formatNumber(trace.expAvgBefore)}</strong><small>一阶动量 m</small></article>
            <article className={`${phaseIndex >= 1 ? "is-read" : ""}${phaseIndex === 4 ? " is-written" : ""}`}><span>state.exp_avg_sq</span><strong>{phaseIndex === 4 ? formatNumber(trace.expAvgSq) : formatNumber(trace.expAvgSqBefore)}</strong><small>二阶动量 v</small></article>
          </div></div>
        </div>
        <footer><strong>{hardwarePhases[phaseIndex].label}</strong><span>{hardwarePhases[phaseIndex].detail}。教学图按逐元素 kernel 画出数据契约，PyTorch 还可选择 foreach 或 fused 实现。</span><code>{hardwarePhase === "write" ? "θ, m, v persist for next step" : "Python holds Tensor metadata; bytes stay in HBM"}</code></footer>
      </section>

      <section className="adamw-memory-stage" aria-labelledby="adamw-memory-title">
        <header><div><span>One billion FP32 parameters, four ranks</span><h3 id="adamw-memory-title">AdamW 的 m、v 正是 ZeRO 最先要分片的内存</h3></div><Network size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="adamw-memory-switch" role="group" aria-label="选择分布式内存策略">{(["ddp", "zero-1", "fsdp"] as AdamWMemoryStrategy[]).map((strategy) => <button type="button" className={memoryStrategy === strategy ? "is-active" : ""} aria-pressed={memoryStrategy === strategy} onClick={() => setMemoryStrategy(strategy)} key={strategy}>{strategy === "ddp" ? "DDP" : strategy === "zero-1" ? "ZeRO-1" : "FSDP"}</button>)}</div>
        <div className="adamw-memory-layout">
          <div className="adamw-memory-stack">
            {[
              ["parameter θ", memory.parameterBytes, "parameter"],
              ["gradient", memory.gradientBytes, "gradient"],
              ["first moment m", memory.firstMomentBytes, "m"],
              ["second moment v", memory.secondMomentBytes, "v"],
            ].map(([label, bytes, key]) => <article className={memory.shardedComponents.includes(String(key)) ? "is-sharded" : "is-replicated"} key={String(label)}><span>{String(label)}</span><strong>{formatGB(Number(bytes))}</strong><small>{memory.shardedComponents.includes(String(key)) ? `shard ÷ ${memory.worldSize}` : "full replica"}</small></article>)}
          </div>
          <aside><span>每个 rank 的持久训练状态</span><strong>{formatGB(memory.persistentBytesPerRank)}</strong><div><small>复制</small><code>{memory.replicatedComponents.join(" + ") || "none"}</code></div><div><small>分片</small><code>{memory.shardedComponents.join(" + ") || "none"}</code></div><p>{memoryStrategy === "ddp" ? "每张卡都有完整 θ、grad、m、v，最简单但重复最多。" : memoryStrategy === "zero-1" ? "θ 与 grad 仍完整，只让每个 rank 保存并更新一部分 m、v。" : "四类持久状态都按 rank 分片，需要时临时 All-Gather 完整参数。"}</p></aside>
        </div>
      </section>

      <section className="adamw-sources" aria-labelledby="adamw-sources-title">
        <div><span>Primary references</span><h3 id="adamw-sources-title">公式和工程语义来自论文与 PyTorch 官方文档</h3></div>
        <nav aria-label="AdamW 参考资料">
          <a href="https://arxiv.org/abs/1412.6980" target="_blank" rel="noreferrer">Adam 原始论文</a>
          <a href="https://arxiv.org/abs/1711.05101" target="_blank" rel="noreferrer">Decoupled Weight Decay 论文</a>
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html" target="_blank" rel="noreferrer">torch.optim.AdamW</a>
          <a href="https://docs.pytorch.org/docs/stable/optim.html" target="_blank" rel="noreferrer">Optimizer 与 parameter groups</a>
        </nav>
      </section>

      <nav className="adamw-bridge-nav" aria-label="AdamW 相邻专题">
        <a href="#/training/gradient"><FunctionIcon size={17} weight="duotone" /><span><small>上游</small><strong>`.grad` 怎样形成</strong></span></a>
        <a href="#/gpu/cuda-kernel"><Cpu size={17} weight="duotone" /><span><small>下钻</small><strong>更新怎样成为 GPU Kernel</strong></span></a>
        <a href="#/distributed/zero-1"><Network size={17} weight="duotone" /><span><small>扩展</small><strong>m、v 怎样跨 rank 分片</strong></span></a>
      </nav>
    </section>
  );
}
