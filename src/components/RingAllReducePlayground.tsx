import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { ArrowsClockwise } from "@phosphor-icons/react/ArrowsClockwise";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { useMemo, useState } from "react";
import {
  createRingTeachingSimulation,
  ringChunkJourney,
  ringCommunicationCost,
  ringRanksForTeachingStep,
  ringRoundForTeachingStep,
  ringTeachingSteps,
  ringTransferForRank,
  type RingInputPresetId,
  type RingTeachingPhase,
} from "../playground/ringAllReduce";

const inputPresets: Array<{ id: RingInputPresetId; label: string; detail: string }> = [
  { id: "rank-pattern", label: "Rank 位值", detail: "R2 的 C1 是 22，容易追踪来源" },
  { id: "same", label: "各 Rank 相同", detail: "只观察调度，不让数值分散注意力" },
  { id: "mixed-sign", label: "正负抵消", detail: "观察部分和怎样逐跳变化" },
];

const messagePresets = [
  { bytes: 4 * 1024, label: "4 KiB", detail: "小消息" },
  { bytes: 4 * 1024 * 1024, label: "4 MiB", detail: "中等消息" },
  { bytes: 1024 * 1024 * 1024, label: "1 GiB", detail: "大梯度桶" },
];

const phaseCopy: Record<RingTeachingPhase, { action: string; state: string }> = {
  initial: {
    action: "每个 Rank 把自己的 M 个元素按相同边界切成 N 个 chunk。",
    state: "每张 GPU 仍持有自己的完整本地 Tensor，每个 chunk 只有一个贡献者。",
  },
  "reduce-scatter": {
    action: "每个 Rank 同时向 next rank 发送一个不同 chunk，接收后与同编号本地 chunk 求和。",
    state: "部分和沿 Ring 移动。N-1 轮后，每张 GPU 只负责一个完整归约 chunk。",
  },
  "all-gather": {
    action: "完整归约 chunk 继续沿 Ring 传播，接收端只复制，不再做求和。",
    state: "每过一轮，每张 GPU 多得到一个完整 chunk，空槽逐步被填满。",
  },
  complete: {
    action: "把 N 个完整 chunk 按原顺序拼回 Tensor。",
    state: "每个 Rank 得到相同的 SUM；训练框架可以再除以 N 得到平均梯度。",
  },
};

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
}

function formatVector(values: number[]): string {
  return values.length > 0 ? `[${values.map(formatNumber).join(", ")}]` : "空";
}

function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || Number.isInteger(value) ? 0 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function RingAllReducePlayground() {
  const [worldSize, setWorldSize] = useState(4);
  const [presetId, setPresetId] = useState<RingInputPresetId>("rank-pattern");
  const [stepIndex, setStepIndex] = useState(1);
  const [selectedRank, setSelectedRank] = useState(0);
  const [selectedChunk, setSelectedChunk] = useState(0);
  const [messageBytes, setMessageBytes] = useState(4 * 1024 * 1024);

  const simulation = useMemo(
    () => createRingTeachingSimulation(worldSize, presetId),
    [worldSize, presetId],
  );
  const steps = useMemo(() => ringTeachingSteps(worldSize), [worldSize]);
  const step = steps[stepIndex];
  const round = ringRoundForTeachingStep(simulation, step);
  const ranks = ringRanksForTeachingStep(simulation, step);
  const selectedTransfers = round ? ringTransferForRank(round, selectedRank) : null;
  const selectedRankState = ranks[selectedRank];
  const journey = ringChunkJourney(simulation, selectedChunk);
  const cost = ringCommunicationCost(worldSize, messageBytes);
  const comparisonCosts = [2, 4, 8].map((size) => ringCommunicationCost(size, messageBytes));
  const currentJourneyIndex = step.phase === "reduce-scatter"
    ? step.round ?? 0
    : step.phase === "all-gather"
      ? worldSize - 1 + (step.round ?? 0)
      : step.phase === "complete"
        ? journey.length
        : -1;

  const changeWorldSize = (nextWorldSize: number) => {
    setWorldSize(nextWorldSize);
    setStepIndex(1);
    setSelectedRank((rank) => Math.min(rank, nextWorldSize - 1));
    setSelectedChunk((chunk) => Math.min(chunk, nextWorldSize - 1));
  };

  const selectPreset = (nextPreset: RingInputPresetId) => {
    setPresetId(nextPreset);
    setStepIndex(1);
  };

  const reset = () => {
    setWorldSize(4);
    setPresetId("rank-pattern");
    setStepIndex(1);
    setSelectedRank(0);
    setSelectedChunk(0);
    setMessageBytes(4 * 1024 * 1024);
  };

  return (
    <section className="ring-playground" id="ring-allreduce-lab" aria-label="Ring AllReduce 交互实验台">
      <header className="ring-playground-header">
        <div>
          <span>Executable Ring schedule</span>
          <h2>选一轮、一个 Rank 和一个 chunk，把并行调度逐项代入公式</h2>
          <p>这里采用单向逻辑 Ring。每个教学 chunk 只有一个标量，方便把注意力放在路由、归约和所有权变化上。</p>
        </div>
        <div className="ring-live-facts" aria-label="当前 Ring 配置">
          <span><small>world size</small><strong>{worldSize}</strong></span>
          <span><small>chunks / rank</small><strong>{worldSize}</strong></span>
          <span><small>rounds / phase</small><strong>{worldSize - 1}</strong></span>
          <span><small>allreduce rounds</small><strong>{2 * (worldSize - 1)}</strong></span>
        </div>
        <button type="button" className="ring-reset" onClick={reset}><ArrowCounterClockwise size={15} aria-hidden="true" />重置</button>
      </header>

      <div className="ring-control-deck">
        <fieldset>
          <legend>World size</legend>
          <div>{[2, 3, 4, 8].map((size) => <button type="button" className={worldSize === size ? "is-active" : ""} aria-pressed={worldSize === size} onClick={() => changeWorldSize(size)} key={size}>{size} ranks</button>)}</div>
        </fieldset>
        <fieldset className="ring-preset-control">
          <legend>本地 Tensor 数值</legend>
          <div>{inputPresets.map((preset) => <button type="button" className={presetId === preset.id ? "is-active" : ""} aria-pressed={presetId === preset.id} title={preset.detail} onClick={() => selectPreset(preset.id)} key={preset.id}><strong>{preset.label}</strong><small>{preset.detail}</small></button>)}</div>
        </fieldset>
        <fieldset>
          <legend>追踪 Rank</legend>
          <div>{Array.from({ length: worldSize }, (_, rank) => <button type="button" className={selectedRank === rank ? "is-active" : ""} aria-pressed={selectedRank === rank} onClick={() => setSelectedRank(rank)} key={rank}>R{rank}</button>)}</div>
        </fieldset>
        <fieldset>
          <legend>追踪 Chunk</legend>
          <div>{Array.from({ length: worldSize }, (_, chunk) => <button type="button" className={`chunk-tone-${chunk % 4}${selectedChunk === chunk ? " is-active" : ""}`} aria-pressed={selectedChunk === chunk} onClick={() => setSelectedChunk(chunk)} key={chunk}>C{chunk}</button>)}</div>
        </fieldset>
      </div>

      <nav className="ring-phase-rail" aria-label="Ring AllReduce 执行阶段">
        <button type="button" className="ring-step-arrow" disabled={stepIndex === 0} onClick={() => setStepIndex((index) => Math.max(0, index - 1))} aria-label="上一状态"><CaretLeft size={16} /></button>
        <div>
          {steps.map((item, index) => <button type="button" className={`${item.phase}${index === stepIndex ? " is-active" : ""}${index < stepIndex ? " is-complete" : ""}`} aria-current={index === stepIndex ? "step" : undefined} onClick={() => setStepIndex(index)} key={item.id}><strong>{item.compactLabel}</strong><small>{item.label}</small></button>)}
        </div>
        <button type="button" className="ring-step-arrow" disabled={stepIndex === steps.length - 1} onClick={() => setStepIndex((index) => Math.min(steps.length - 1, index + 1))} aria-label="下一状态"><CaretRight size={16} /></button>
      </nav>

      <section className="ring-logical-order" aria-labelledby="ring-order-title">
        <header><div><span>Logical ring order</span><h3 id="ring-order-title">每个 Rank 只认一个 next 和一个 previous</h3></div><code>next(r) = (r + 1) mod {worldSize}</code></header>
        <div>
          {Array.from({ length: worldSize }, (_, rank) => (
            <span key={rank}>
              <button type="button" className={selectedRank === rank ? "is-selected" : ""} onClick={() => setSelectedRank(rank)}><Cpu size={15} weight="duotone" aria-hidden="true" />Rank {rank}</button>
              <ArrowRight size={15} aria-hidden="true" />
            </span>
          ))}
          <button type="button" className={selectedRank === 0 ? "is-selected is-return" : "is-return"} onClick={() => setSelectedRank(0)}>Rank 0</button>
        </div>
        <p>这是逻辑顺序，不要求机箱里真的摆成圆。Backend 会把逻辑边映射到 NVLink、PCIe 或跨机网络路径。</p>
      </section>

      <section className={`ring-round-stage phase-${step.phase}`} id="ring-round-stage" aria-labelledby="ring-round-title">
        <header>
          <div><span>{step.label}</span><h3 id="ring-round-title">{phaseCopy[step.phase].action}</h3><p>{phaseCopy[step.phase].state}</p></div>
          <code>{step.phase === "reduce-scatter" ? "send_chunk = (rank - round) mod N" : step.phase === "all-gather" ? "send_chunk = (rank + 1 - round) mod N" : step.phase === "initial" ? "chunks = split(tensor, N)" : "AllReduce output ready"}</code>
        </header>

        {round ? (
          <div className="ring-round-layout">
            <div className="ring-transfer-lanes" style={{ gridTemplateColumns: `repeat(${worldSize}, minmax(220px, 1fr))` }}>
              {round.transfers.map((transfer) => (
                <button type="button" className={`${transfer.from === selectedRank ? "is-selected" : ""}${transfer.chunk === selectedChunk ? " is-followed" : ""}`} onClick={() => { setSelectedRank(transfer.from); setSelectedChunk(transfer.chunk); }} key={transfer.from}>
                  <span>R{transfer.from}</span><ArrowRight size={13} aria-hidden="true" />
                  <strong className={`chunk-tone-${transfer.chunk % 4}`}>C{transfer.chunk}<code>{formatVector(transfer.sent)}</code></strong>
                  <ArrowRight size={13} aria-hidden="true" /><span>R{transfer.to}</span>
                  <small>{round.phase === "reduce-scatter" ? "接收后逐元素求和" : "接收后写入固定槽位"}</small>
                </button>
              ))}
            </div>

            <aside className="ring-rank-inspector" aria-live="polite">
              <header><span>只看 Rank {selectedRank}</span><strong>同一轮既发送，也接收</strong></header>
              <div className="ring-formula-substitution">
                <span>发送公式</span>
                <code>{round.phase === "reduce-scatter" ? `(${selectedRank} - ${round.round}) mod ${worldSize}` : `(${selectedRank} + 1 - ${round.round}) mod ${worldSize}`} = C{selectedTransfers!.outgoing.chunk}</code>
                <small>R{selectedRank} 把 C{selectedTransfers!.outgoing.chunk} 发给 R{selectedTransfers!.outgoing.to}</small>
              </div>
              <div className="ring-receive-equation">
                <span>接收动作</span>
                <strong>R{selectedTransfers!.incoming.from} 发来 C{selectedTransfers!.incoming.chunk}</strong>
                {round.phase === "reduce-scatter" ? (
                  <code>{formatVector(selectedTransfers!.incoming.before ?? [])} + {formatVector(selectedTransfers!.incoming.sent)} = {formatVector(selectedTransfers!.incoming.after)}</code>
                ) : (
                  <code>空槽 ← {formatVector(selectedTransfers!.incoming.sent)}</code>
                )}
                <small>{round.phase === "reduce-scatter" ? `现在包含 ${selectedTransfers!.incoming.contributors.map((rank) => `R${rank}`).join(" + ")} 的贡献` : "归约已经完成，这一阶段不再改变数值"}</small>
              </div>
            </aside>
          </div>
        ) : (
          <div className="ring-boundary-state">
            {ranks.map((rankState) => <button type="button" className={selectedRank === rankState.rank ? "is-selected" : ""} onClick={() => setSelectedRank(rankState.rank)} key={rankState.rank}><span>Rank {rankState.rank}</span><strong>{formatVector(rankState.chunks.flatMap((chunk) => chunk.values))}</strong><small>{step.phase === "initial" ? `${worldSize} 个本地 chunk，尚未交换` : "完整 SUM Tensor 已写回"}</small></button>)}
          </div>
        )}
      </section>

      <section className="ring-state-matrix" aria-labelledby="ring-matrix-title">
        <header><div><span>Exact state after this step</span><h3 id="ring-matrix-title">数值矩阵显示每个槽位此刻究竟有什么</h3></div><code>{step.label}</code></header>
        <div className="ring-state-matrix-scroll">
          <div className="ring-state-table" style={{ gridTemplateColumns: `112px repeat(${worldSize}, minmax(118px, 1fr))` }}>
            <span className="ring-matrix-corner">HBM slots</span>
            {Array.from({ length: worldSize }, (_, chunk) => <button type="button" className={`chunk-tone-${chunk % 4}${selectedChunk === chunk ? " is-selected" : ""}`} onClick={() => setSelectedChunk(chunk)} key={`h-${chunk}`}><strong>C{chunk}</strong><small>element {chunk}</small></button>)}
            {ranks.flatMap((rankState) => [
              <button type="button" className={`ring-matrix-rank${selectedRank === rankState.rank ? " is-selected" : ""}`} onClick={() => setSelectedRank(rankState.rank)} key={`r-${rankState.rank}`}><strong>Rank {rankState.rank}</strong><small>GPU {rankState.rank}</small></button>,
              ...rankState.chunks.map((chunk) => <button type="button" className={`ring-matrix-cell chunk-tone-${chunk.chunk % 4}${chunk.values.length === 0 ? " is-empty" : ""}${chunk.complete ? " is-complete" : ""}${selectedRank === rankState.rank && selectedChunk === chunk.chunk ? " is-selected" : ""}`} onClick={() => { setSelectedRank(rankState.rank); setSelectedChunk(chunk.chunk); }} key={`r-${rankState.rank}-c-${chunk.chunk}`}><code>{formatVector(chunk.values)}</code><span>{chunk.contributors.length > 0 ? chunk.contributors.map((rank) => `R${rank}`).join(" + ") : "等待传入"}</span><small>{chunk.complete ? "完整归约" : `${chunk.contributors.length}/${worldSize} contributors`}</small></button>),
            ])}
          </div>
        </div>
        <footer><Database size={17} weight="duotone" aria-hidden="true" /><p>Reduce-Scatter 期间，教学模型保留旧的部分和槽位便于比较；只有标记“完整归约”的 chunk 才是该阶段的逻辑输出。All-Gather 开始时，非 owner 槽位会显示为空，随后被远端完整 chunk 覆盖。</p></footer>
      </section>

      <section className="ring-chunk-journey" aria-labelledby="ring-journey-title">
        <header><div><span>Follow one data identity</span><h3 id="ring-journey-title">C{selectedChunk} 在 {2 * (worldSize - 1)} 轮中走过的完整路线</h3></div><code>chunk identity never changes</code></header>
        <div className="ring-journey-track">
          <div className={`ring-journey-origin chunk-tone-${selectedChunk % 4}`}><strong>C{selectedChunk}</strong><small>逻辑数据身份</small></div>
          {journey.map((hop, index) => <button type="button" className={`${hop.phase}${index === currentJourneyIndex ? " is-active" : ""}${index < currentJourneyIndex ? " is-past" : ""}`} onClick={() => setStepIndex(hop.phase === "reduce-scatter" ? 1 + hop.round : worldSize + hop.round)} key={`${hop.phase}-${hop.round}`}><small>{hop.phase === "reduce-scatter" ? `RS ${hop.round + 1}` : `AG ${hop.round + 1}`}</small><strong>R{hop.from}<ArrowRight size={12} aria-hidden="true" />R{hop.to}</strong><span>{hop.phase === "reduce-scatter" ? "移动部分和" : "复制完整结果"}</span></button>)}
        </div>
      </section>

      <section className="ring-cost-stage" aria-labelledby="ring-cost-title">
        <header><div><span>Bandwidth and latency model</span><h3 id="ring-cost-title">Ring 省的不是轮数，而是每轮只搬 M/N</h3><p>选择一个每 Rank 输入 Tensor 的大小。这里只计算算法 payload，不包含协议头、对齐、分片流水和链路重传。</p></div><code>M = {formatBytes(messageBytes)}</code></header>
        <div className="ring-message-presets" role="group" aria-label="选择输入 Tensor 大小">{messagePresets.map((preset) => <button type="button" className={messageBytes === preset.bytes ? "is-active" : ""} aria-pressed={messageBytes === preset.bytes} onClick={() => setMessageBytes(preset.bytes)} key={preset.bytes}><strong>{preset.label}</strong><small>{preset.detail}</small></button>)}</div>
        <div className="ring-cost-equation">
          <article><span>每轮每 Rank</span><strong>M / N</strong><code>{formatBytes(cost.chunkBytes)}</code></article>
          <ArrowRight size={16} aria-hidden="true" />
          <article><span>每个阶段</span><strong>(N-1) × M/N</strong><code>{formatBytes(cost.bytesPerRankPerPhase)}</code></article>
          <ArrowRight size={16} aria-hidden="true" />
          <article className="is-result"><span>完整 AllReduce</span><strong>2(N-1) × M/N</strong><code>{formatBytes(cost.bytesPerRankTotal)} / rank</code></article>
        </div>
        <div className="ring-cost-tapes" aria-label="Reduce-Scatter 与 All-Gather 通信量分段">
          {(["Reduce-Scatter", "All-Gather"] as const).map((phase) => <div key={phase}><span>{phase}</span><div>{Array.from({ length: worldSize - 1 }, (_, roundIndex) => <i key={roundIndex}>M/{worldSize}</i>)}</div><strong>{formatBytes(cost.bytesPerRankPerPhase)}</strong></div>)}
        </div>
        <div className="ring-scale-comparison">
          {comparisonCosts.map((item) => <button type="button" className={item.worldSize === worldSize ? "is-active" : ""} onClick={() => changeWorldSize(item.worldSize)} key={item.worldSize}><span>{item.worldSize} ranks</span><strong>{item.totalRounds} 轮</strong><code>{item.messageMultiplesPerRank.toFixed(3)} × M / rank</code><small>大消息通信量趋近 2M，轮数随 N 线性增长</small></button>)}
        </div>
      </section>

      <section className="ring-layer-separation" aria-labelledby="ring-separation-title">
        <header><div><span>Do not mix these layers</span><h3 id="ring-separation-title">AllReduce、Ring 和 NVLink 不是同一个层级</h3></div><Network size={22} weight="duotone" aria-hidden="true" /></header>
        <div>
          <article><FunctionIcon size={20} weight="duotone" aria-hidden="true" /><span>Collective 契约</span><strong>AllReduce</strong><p>规定所有 Rank 最终得到相同归约结果，不规定内部路线。</p></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article><ArrowsClockwise size={20} weight="duotone" aria-hidden="true" /><span>通信算法</span><strong>Ring</strong><p>用 Reduce-Scatter 加 All-Gather 实现契约，强调大消息带宽利用率。</p></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article><Lightning size={20} weight="duotone" aria-hidden="true" /><span>执行机制</span><strong>NCCL kernel + CUDA stream</strong><p>GPU kernel 同时处理搬运与归约，并通过 Stream 建立异步顺序。</p></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article><Stack size={20} weight="duotone" aria-hidden="true" /><span>物理传输</span><strong>NVLink / PCIe / RDMA</strong><p>逻辑 Ring 边最终映射到真实链路；跨机路径还会经过 NIC 和交换网络。</p></article>
        </div>
        <footer><p><strong>Ring 不是永远最快。</strong>它的带宽项适合大消息，但需要 2(N-1) 个顺序步骤。Tree 可提供对数级延迟，Backend 会结合规模、消息大小与拓扑选择实现。</p></footer>
      </section>

      <section className="ring-result-stage" aria-labelledby="ring-result-title">
        <header><div><span>Collective result</span><h3 id="ring-result-title">调度可以旋转，最终语义不能变</h3></div><code>SUM then optional ÷ {worldSize}</code></header>
        <div><article><span>Elementwise SUM</span><strong>{formatVector(simulation.reduced)}</strong></article><article><span>DDP 常用平均梯度</span><strong>{formatVector(simulation.averaged)}</strong></article><article><span>选中 Rank {selectedRank} 当前状态</span><strong>{formatVector(selectedRankState.chunks.flatMap((chunk) => chunk.values))}</strong></article></div>
        <p>本页采用 RS: (rank - round) mod N 的等价旋转，因此 Reduce-Scatter 后 Rank r 拥有 C(r+1)。换一个整体偏移仍然是正确 Ring，只要所有 Rank 使用同一调度并覆盖所有 chunk。</p>
      </section>

      <section className="ring-sources" aria-labelledby="ring-sources-title">
        <div><span>Primary references</span><h3 id="ring-sources-title">语义、带宽下界和 NCCL 执行来自官方文档与原始论文</h3></div>
        <nav aria-label="Ring AllReduce 资料">
          <a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html" target="_blank" rel="noreferrer">NCCL overview and execution model</a>
          <a href="https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/" target="_blank" rel="noreferrer">NVIDIA ring-style collectives</a>
          <a href="https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/" target="_blank" rel="noreferrer">NVIDIA Ring and Tree tradeoff</a>
          <a href="https://doi.org/10.1016/j.jpdc.2008.09.002" target="_blank" rel="noreferrer">Bandwidth-optimal Ring AllReduce paper</a>
        </nav>
      </section>
    </section>
  );
}
