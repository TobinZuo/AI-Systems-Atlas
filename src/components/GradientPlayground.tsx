import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { MathOperations } from "@phosphor-icons/react/MathOperations";
import { Network } from "@phosphor-icons/react/Network";
import { Sigma } from "@phosphor-icons/react/Sigma";
import { Stack } from "@phosphor-icons/react/Stack";
import { Target } from "@phosphor-icons/react/Target";
import { WarningCircle } from "@phosphor-icons/react/WarningCircle";
import { useMemo, useState } from "react";
import {
  defaultGradientConfig,
  gradientDataset,
  simulateGradient,
  simulateGradientDDP,
  type GradientBatchSize,
  type GradientConfig,
  type GradientDevice,
  type GradientDirection,
  type GradientRankSplit,
  type GradientReduction,
  type GradientVector,
} from "../playground/gradient";

const gradientPhases = [
  { label: "Forward", title: "先得到 prediction、residual 与标量 loss", detail: "Backward 使用 Forward 保存的局部信息，不会重新猜测输入。" },
  { label: "样本贡献", title: "每个样本产生一条参数梯度向量", detail: "同一个参数会从 batch 中的多个样本收到贡献。" },
  { label: "Batch reduction", title: "sum 或 mean 决定最终梯度尺度", detail: "Loss 的 reduction 会沿链式法则原样缩放梯度。" },
  { label: ".grad buffer", title: "Autograd 把结果累加到参数旁边的 buffer", detail: "Gradient 具有相同 shape、dtype 与 device，并占用真实内存。" },
  { label: "Parameter update", title: "Optimizer 读取梯度，再决定实际更新", detail: "负梯度只是局部方向，学习率过大仍然可能越过低点。" },
] as const;

const parameterOptions = {
  w0: [-1, 0, 0.5, 1, 2],
  w1: [-2, -1, 0, 1],
  b: [0, 0.5, 1],
};

const learningRates = [0.05, 0.1, 0.5, 1];

function formatNumber(value: number, digits = 4): string {
  if (Math.abs(value) < 1e-10) return "0";
  if (Number.isInteger(value)) return String(value);
  return Number(value.toFixed(digits)).toString();
}

function formatVector(vector: GradientVector): string {
  return `[${formatNumber(vector.w0)}, ${formatNumber(vector.w1)}, ${formatNumber(vector.b)}]`;
}

function formatArray(values: number[]): string {
  return `[${values.map((value) => formatNumber(value)).join(", ")}]`;
}

export function GradientPlayground() {
  const [config, setConfig] = useState<GradientConfig>(defaultGradientConfig);
  const [selectedSampleId, setSelectedSampleId] = useState(0);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [rankSplit, setRankSplit] = useState<GradientRankSplit>("2+2");

  const simulation = useMemo(() => simulateGradient(config), [config]);
  const ddp = useMemo(
    () => simulateGradientDDP({ w0: config.w0, w1: config.w1, b: config.b }, rankSplit),
    [config.w0, config.w1, config.b, rankSplit],
  );
  const selectedSample = simulation.samples.find((sample) => sample.id === selectedSampleId) ?? simulation.samples[0];
  const phase = gradientPhases[phaseIndex];
  const gradientSum = config.reduction === "sum"
    ? simulation.aggregateGradient
    : {
        w0: simulation.aggregateGradient.w0 * config.batchSize,
        w1: simulation.aggregateGradient.w1 * config.batchSize,
        b: simulation.aggregateGradient.b * config.batchSize,
      };
  const stepImprovesLoss = simulation.step.lossChange < 0;

  const updateConfig = <K extends keyof GradientConfig>(key: K, value: GradientConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  const changeBatchSize = (batchSize: GradientBatchSize) => {
    updateConfig("batchSize", batchSize);
    setSelectedSampleId((sampleId) => Math.min(sampleId, batchSize - 1));
  };

  const reset = () => {
    setConfig(defaultGradientConfig);
    setSelectedSampleId(0);
    setPhaseIndex(0);
    setRankSplit("2+2");
  };

  return (
    <section className="gradient-playground" id="gradient-lab" aria-label="Gradient 交互实验台">
      <header className="gradient-playground-header">
        <div>
          <span>Executable gradient model</span>
          <h2>同一条梯度，从样本贡献一直追到 HBM 和 DDP</h2>
          <p>所有数值都来自当前参数与四条固定样本，没有预先写死的结果。</p>
        </div>
        <div className="gradient-live-facts" aria-label="当前模型状态">
          <span><strong>ŷ = w₀x₀ + w₁x₁ + b</strong>线性模型</span>
          <span><strong>0.5 × residual²</strong>单样本 loss</span>
          <span><strong>{formatNumber(simulation.aggregateLoss)}</strong>当前 batch loss</span>
          <span><strong>{formatNumber(simulation.gradientNorm)}</strong>gradient L2 norm</span>
        </div>
        <button type="button" className="gradient-reset" onClick={reset}><ArrowCounterClockwise size={15} aria-hidden="true" />重置</button>
      </header>

      <div className="gradient-control-deck">
        <fieldset className="gradient-parameter-controls">
          <legend>模型参数</legend>
          {(Object.keys(parameterOptions) as Array<keyof typeof parameterOptions>).map((parameter) => (
            <label key={parameter}><span>{parameter}</span><select value={config[parameter]} onChange={(event) => updateConfig(parameter, Number(event.target.value))}>{parameterOptions[parameter].map((value) => <option value={value} key={value}>{value}</option>)}</select></label>
          ))}
        </fieldset>

        <fieldset>
          <legend>Batch size</legend>
          <div role="group" aria-label="选择 batch size">{([1, 2, 4] as GradientBatchSize[]).map((size) => <button type="button" className={config.batchSize === size ? "is-active" : ""} aria-pressed={config.batchSize === size} onClick={() => changeBatchSize(size)} key={size}>{size}</button>)}</div>
        </fieldset>

        <fieldset>
          <legend>Loss reduction</legend>
          <div role="group" aria-label="选择 loss reduction">{(["mean", "sum"] as GradientReduction[]).map((reduction) => <button type="button" className={config.reduction === reduction ? "is-active" : ""} aria-pressed={config.reduction === reduction} onClick={() => updateConfig("reduction", reduction)} key={reduction}>{reduction}</button>)}</div>
        </fieldset>

        <fieldset>
          <legend>Tensor device</legend>
          <div role="group" aria-label="选择 tensor device">{(["cuda", "cpu"] as GradientDevice[]).map((device) => <button type="button" className={config.device === device ? "is-active" : ""} aria-pressed={config.device === device} onClick={() => updateConfig("device", device)} key={device}>{device.toUpperCase()}</button>)}</div>
        </fieldset>

        <fieldset>
          <legend>Learning rate</legend>
          <div role="group" aria-label="选择 learning rate">{learningRates.map((learningRate) => <button type="button" className={config.learningRate === learningRate ? "is-active" : ""} aria-pressed={config.learningRate === learningRate} onClick={() => updateConfig("learningRate", learningRate)} key={learningRate}>{learningRate}</button>)}</div>
        </fieldset>

        <fieldset>
          <legend>Update direction</legend>
          <div role="group" aria-label="选择更新方向">{(["descent", "ascent"] as GradientDirection[]).map((direction) => <button type="button" className={`${config.direction === direction ? "is-active" : ""}${direction === "ascent" ? " is-danger" : ""}`} aria-pressed={config.direction === direction} onClick={() => updateConfig("direction", direction)} key={direction}>{direction}</button>)}</div>
        </fieldset>
      </div>

      <nav className="gradient-phase-rail" role="tablist" aria-label="梯度形成与消费阶段">
        {gradientPhases.map((item, index) => <button type="button" role="tab" aria-selected={phaseIndex === index} className={phaseIndex === index ? "is-active" : index < phaseIndex ? "is-complete" : ""} onClick={() => setPhaseIndex(index)} key={item.label}><span>{item.label}</span><small>{index + 1}</small></button>)}
      </nav>

      <div className="gradient-operation-band" aria-live="polite">
        <div><Target size={21} weight="duotone" aria-hidden="true" /></div>
        <span><strong>{phase.title}</strong><small>{phase.detail}</small></span>
        <code>∇L = {formatVector(simulation.aggregateGradient)}</code>
      </div>

      <section className={`gradient-sample-section${phaseIndex <= 1 ? " is-focused" : ""}`} aria-labelledby="gradient-sample-title">
        <header><div><span>Forward values and local derivatives</span><h3 id="gradient-sample-title">一条 batch 不是一个梯度黑盒，它是多条样本贡献</h3></div><code>rᵢ = ŷᵢ - yᵢ</code></header>
        <div className="gradient-sample-grid">
          {gradientDataset.map((definition) => {
            const sample = simulation.samples.find((item) => item.id === definition.id);
            const active = Boolean(sample);
            return (
              <button type="button" disabled={!active} className={`${active ? "is-in-batch" : "is-excluded"}${selectedSample.id === definition.id ? " is-selected" : ""}`} onClick={() => setSelectedSampleId(definition.id)} key={definition.id}>
                <header><strong>Sample {definition.id}</strong><span>{active ? "参与 reduction" : "不在当前 batch"}</span></header>
                <dl>
                  <div><dt>x</dt><dd>[{definition.x.join(", ")}]</dd></div>
                  <div><dt>target</dt><dd>{definition.target}</dd></div>
                  <div><dt>prediction</dt><dd>{sample ? formatNumber(sample.prediction) : "-"}</dd></div>
                  <div><dt>residual</dt><dd>{sample ? formatNumber(sample.residual) : "-"}</dd></div>
                  <div><dt>loss</dt><dd>{sample ? formatNumber(sample.loss) : "-"}</dd></div>
                </dl>
                <footer><span>sample gradient</span><code>{sample ? formatVector(sample.gradient) : "not computed"}</code></footer>
              </button>
            );
          })}
        </div>

        <div className="gradient-chain-rule" aria-label={`Sample ${selectedSample.id} 的链式法则`}>
          <div><span>选中样本</span><strong>Sample {selectedSample.id}</strong><code>x={formatArray(selectedSample.x)}，y={selectedSample.target}</code></div>
          <ArrowRight size={17} aria-hidden="true" />
          <div><span>Forward</span><strong>ŷ={formatNumber(selectedSample.prediction)}</strong><code>r={formatNumber(selectedSample.residual)}</code></div>
          <ArrowRight size={17} aria-hidden="true" />
          <div><span>局部导数</span><strong>∂L/∂ŷ = r</strong><code>{formatNumber(selectedSample.residual)}</code></div>
          <ArrowRight size={17} aria-hidden="true" />
          <div className="is-result"><span>参数贡献</span><strong>[r×x₀, r×x₁, r]</strong><code>{formatVector(selectedSample.gradient)}</code></div>
        </div>
        <p className="gradient-loss-assumption"><strong>教学假设：</strong>这里使用 <code>0.5 × residual²</code>，因此 <code>∂L/∂ŷ = residual</code>。PyTorch <code>MSELoss</code> 使用 <code>residual²</code>，局部导数会多一个系数 2；后续 reduction、buffer 与 DDP 逻辑不变。</p>
      </section>

      <section className={`gradient-reduction-section${phaseIndex === 2 ? " is-focused" : ""}`} aria-labelledby="gradient-reduction-title">
        <header><div><span>Batch aggregation</span><h3 id="gradient-reduction-title">先对样本梯度求和，再按 loss reduction 缩放</h3></div><Sigma size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="gradient-reduction-flow">
          <div className="sample-gradient-stack">{simulation.samples.map((sample) => <span key={sample.id}><b>g{sample.id}</b><code>{formatVector(sample.gradient)}</code></span>)}</div>
          <ArrowRight size={18} aria-hidden="true" />
          <div className="gradient-sum-node"><span>SUM</span><strong>{formatVector(gradientSum)}</strong><small>{simulation.samples.length} 条 contribution 逐元素相加</small></div>
          <ArrowRight size={18} aria-hidden="true" />
          <div className="gradient-reduction-node"><span>{config.reduction.toUpperCase()}</span><strong>{config.reduction === "mean" ? `SUM ÷ ${config.batchSize}` : "保持 SUM"}</strong><code>{formatVector(simulation.aggregateGradient)}</code></div>
        </div>
        <footer><code>L_batch = {config.reduction}([{simulation.samples.map((sample) => formatNumber(sample.loss)).join(", ")}]) = {formatNumber(simulation.aggregateLoss)}</code><p>切换 <code>mean</code> 与 <code>sum</code> 不改变每条样本导数，但会改变最终 <code>.grad</code> 的尺度，因此也会改变同一 learning rate 下的更新幅度。</p></footer>
      </section>

      <section className={`gradient-buffer-section${phaseIndex === 3 ? " is-focused" : ""}`} aria-labelledby="gradient-buffer-title">
        <header><div><span>Runtime and storage</span><h3 id="gradient-buffer-title">梯度不是一个 Python 数字，而是参数旁边的真实 Tensor buffer</h3></div><Database size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="gradient-runtime-path">
          <article className="runtime-host"><Cpu size={19} weight="duotone" /><span>Python 进程</span><strong>loss.backward()</strong><small>启动 Autograd Engine</small></article>
          <ArrowRight size={16} aria-hidden="true" />
          <article className="runtime-engine"><FunctionIcon size={19} weight="duotone" /><span>Autograd</span><strong>调度 backward Node</strong><small>计算并累加 contribution</small></article>
          <ArrowRight size={16} aria-hidden="true" />
          <article className="runtime-queue"><Lightning size={19} weight="duotone" /><span>{config.device === "cuda" ? "CUDA Compute Stream" : "CPU worker"}</span><strong>{config.device === "cuda" ? "enqueue backward kernels" : "run CPU kernels"}</strong><small>保持依赖顺序</small></article>
          <ArrowRight size={16} aria-hidden="true" />
          <article className="runtime-device"><Stack size={19} weight="duotone" /><span>{config.device === "cuda" ? "GPU SM" : "CPU cores"}</span><strong>读取输入并计算 dW</strong><small>{config.device === "cuda" ? "Kernel 直接读写 HBM 地址" : "Kernel 直接读写 DRAM 地址"}</small></article>
        </div>

        <div className="gradient-buffer-layout">
          <div className="gradient-buffer-list">
            {simulation.buffers.map((buffer) => <article key={buffer.name}><header><span>{buffer.memory}</span><strong>{buffer.name}</strong></header><dl><div><dt>shape</dt><dd>{buffer.shape}</dd></div><div><dt>dtype</dt><dd>{buffer.dtype}</dd></div><div><dt>device</dt><dd>{buffer.device}</dd></div><div><dt>bytes</dt><dd>{buffer.bytes} B</dd></div></dl><code>{formatArray(buffer.values)}</code></article>)}
          </div>
          <aside className="gradient-bucket-view"><span>DDP bucket flatten view</span><strong>[weight.grad | bias.grad]</strong><div>{simulation.flattenedBucket.map((value, index) => <i key={index}><b>{index}</b><code>{formatNumber(value)}</code></i>)}</div><p>通信框架知道 bucket 的 dtype、元素数量和显存地址。真正传输的是地址指向的梯度字节，不是只有地址本身。</p></aside>
        </div>
      </section>

      <section className={`gradient-update-section${phaseIndex === 4 ? " is-focused" : ""}`} aria-labelledby="gradient-update-title">
        <header><div><span>Local direction versus actual step</span><h3 id="gradient-update-title">梯度给方向，optimizer 与 learning rate 决定走多远</h3></div><MathOperations size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="gradient-update-equation">
          <article><span>当前参数 θ</span><strong>{formatVector({ w0: config.w0, w1: config.w1, b: config.b })}</strong><small>L(θ) = {formatNumber(simulation.aggregateLoss)}</small></article>
          <div><strong>{config.direction === "descent" ? "-" : "+"} lr × ∇L</strong><code>{formatVector(simulation.step.delta)}</code></div>
          <article><span>新参数 θ′</span><strong>{formatVector(simulation.step.nextParameters)}</strong><small>L(θ′) = {formatNumber(simulation.step.nextLoss)}</small></article>
          <aside className={stepImprovesLoss ? "is-success" : "is-warning"}>{stepImprovesLoss ? <CheckCircle size={22} weight="fill" /> : <WarningCircle size={22} weight="fill" />}<span><strong>{stepImprovesLoss ? "实际 loss 下降" : "实际 loss 上升"}</strong><small>ΔL = {formatNumber(simulation.step.lossChange)}</small></span></aside>
        </div>
        <div className="gradient-local-warning"><Target size={18} weight="duotone" /><p>一阶近似预测 <code>∇L · Δθ = {formatNumber(simulation.step.predictedFirstOrderChange)}</code>。它只在足够小的邻域内可靠，所以即使沿负梯度，learning rate 过大也可能 overshoot。</p></div>
        <div className="finite-difference-checks"><header><span>Central finite difference check</span><code>[L(θ+ε) - L(θ-ε)] / 2ε，ε=1e-4</code></header>{simulation.finiteDifferences.map((check) => <article key={check.parameter}><strong>{check.parameter}</strong><span><small>Autograd</small><code>{formatNumber(check.analytic, 6)}</code></span><span><small>数值近似</small><code>{formatNumber(check.numeric, 6)}</code></span><span><small>absolute error</small><code>{check.absoluteError.toExponential(1)}</code></span></article>)}</div>
      </section>

      <section className="gradient-ddp-section" aria-labelledby="gradient-ddp-title">
        <header><div><span>Single GPU gradient to DDP</span><h3 id="gradient-ddp-title">Rank-local mean 什么时候等于 global batch mean</h3></div><Network size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="gradient-ddp-controls"><span>每个 rank 的 local batch</span><div role="group" aria-label="选择 DDP local batch 切分">{(["2+2", "1+3"] as GradientRankSplit[]).map((split) => <button type="button" className={rankSplit === split ? "is-active" : ""} aria-pressed={rankSplit === split} onClick={() => setRankSplit(split)} key={split}>{split === "2+2" ? "R0=2, R1=2" : "R0=1, R1=3"}</button>)}</div></div>
        <div className="gradient-ddp-flow">
          {ddp.ranks.map((rank) => <article className="gradient-rank-card" key={rank.rank}><header><span>Rank {rank.rank} 进程</span><strong>GPU {rank.rank} HBM</strong></header><div><small>local samples</small><strong>{rank.sampleIds.map((id) => `S${id}`).join(" + ")}</strong></div><div><small>local loss mean 的梯度</small><code>{formatVector(rank.localMean)}</code></div><footer>local batch size = {rank.localBatchSize}</footer></article>)}
          <div className="gradient-allreduce-node"><Network size={19} weight="duotone" /><span>DDP AllReduce</span><strong>(g_rank0 + g_rank1) ÷ 2</strong><code>{formatVector(ddp.rankMean)}</code></div>
          <article className={`gradient-global-result${ddp.rankMeanMatchesGlobal ? " is-success" : " is-warning"}`}><header>{ddp.rankMeanMatchesGlobal ? <CheckCircle size={18} weight="fill" /> : <WarningCircle size={18} weight="fill" />}<strong>{ddp.rankMeanMatchesGlobal ? "等于 global sample mean" : "不等于 global sample mean"}</strong></header><span>真实 4 样本 mean</span><code>{formatVector(ddp.globalBatchMean)}</code>{!ddp.rankMeanMatchesGlobal && <p>按 local batch size 加权后才得到 {formatVector(ddp.sampleWeightedMean)}</p>}</article>
        </div>
        <footer><p>{ddp.rankMeanMatchesGlobal ? "两个 rank 的 local batch 一样大，所以先各自求 mean、再平均 rank，与直接对全部样本求 mean 完全一致。" : "两个 rank 的 local batch 大小不同。DDP 默认按 rank 等权平均，因此样本权重不再相同，需要额外加权才能匹配 global sample mean。"}</p><a href="#/distributed/ddp">进入 DDP 硬件现场<ArrowRight size={14} aria-hidden="true" /></a></footer>
      </section>

      <section className="gradient-sources" aria-labelledby="gradient-sources-title">
        <div><span>Official semantics</span><h3 id="gradient-sources-title">页面中的关键契约来自 PyTorch 官方文档</h3></div>
        <nav aria-label="Gradient 官方参考资料">
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html" target="_blank" rel="noreferrer">backward 与 leaf gradient accumulation</a>
          <a href="https://docs.pytorch.org/docs/stable/notes/autograd.html" target="_blank" rel="noreferrer">Autograd mechanics</a>
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html" target="_blank" rel="noreferrer">MSELoss sum 与 mean reduction</a>
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html" target="_blank" rel="noreferrer">DistributedDataParallel gradient semantics</a>
        </nav>
      </section>

      <nav className="gradient-bridge-nav" aria-label="Gradient 相邻专题">
        <a href="#/training/autograd"><FunctionIcon size={17} weight="duotone" /><span><small>上游</small><strong>Autograd 如何算出梯度</strong></span></a>
        <a href="#/training/adamw"><MathOperations size={17} weight="duotone" /><span><small>下游</small><strong>AdamW 怎样消费梯度</strong></span></a>
        <a href="#/gpu/architecture"><Cpu size={17} weight="duotone" /><span><small>下钻</small><strong>GPU 怎样执行梯度 kernel</strong></span></a>
        <a href="#/distributed/ddp"><Network size={17} weight="duotone" /><span><small>扩展</small><strong>多 GPU 怎样同步梯度</strong></span></a>
      </nav>
    </section>
  );
}
