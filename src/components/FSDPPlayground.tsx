import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowsOut } from "@phosphor-icons/react/ArrowsOut";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Code } from "@phosphor-icons/react/Code";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Memory } from "@phosphor-icons/react/Memory";
import { Stack } from "@phosphor-icons/react/Stack";
import { useMemo, useState } from "react";
import {
  allGatherPayloadBytes,
  averagedGradient,
  fsdpLayers,
  fsdpMemorySnapshot,
  fsdpPhases,
  fsdpRankStates,
  paddedRows,
  rowsPerShard,
  type ComputeDtype,
} from "../playground/fsdp";
import { fsdpHardwareSnapshot } from "../playground/distributedHardware";
import { formatBytes } from "../playground/shardedOptimizer";
import { DistributedHardwarePath } from "./DistributedHardwarePath";

export function FSDPPlayground() {
  const [worldSize, setWorldSize] = useState(2);
  const [computeDtype, setComputeDtype] = useState<ComputeDtype>("fp32");
  const [layerId, setLayerId] = useState("embedding");
  const [selectedRank, setSelectedRank] = useState(0);
  const [selectedRow, setSelectedRow] = useState(0);
  const [phaseIndex, setPhaseIndex] = useState(0);

  const layer = fsdpLayers.find((item) => item.id === layerId)!;
  const phase = fsdpPhases[phaseIndex];
  const rankStates = useMemo(
    () => fsdpRankStates(layer, worldSize, phase, computeDtype),
    [layer, worldSize, phase, computeDtype],
  );
  const shardRowCount = rowsPerShard(layer, worldSize);
  const selectedRowOwner = Math.floor(selectedRow / shardRowCount);
  const memory = fsdpMemorySnapshot(layer, worldSize, phase, computeDtype);
  const layerElements = layer.rows * layer.columns;
  const ddpBytes = layerElements * 4 * 4;
  const zeroAverageBytes = layerElements * 4 * 2 + (layerElements * 4 * 2) / worldSize;
  const fsdpPersistentAdamBytes = shardRowCount * layer.columns * 4 * 4;
  const hardwareSnapshot = fsdpHardwareSnapshot(phase, layer.name, computeDtype);

  const changeWorldSize = (nextWorldSize: number) => {
    setWorldSize(nextWorldSize);
    setSelectedRank((rank) => Math.min(rank, nextWorldSize - 1));
    setPhaseIndex(0);
  };

  const changeLayer = (nextLayerId: string) => {
    const nextLayer = fsdpLayers.find((item) => item.id === nextLayerId)!;
    setLayerId(nextLayerId);
    setSelectedRow((row) => Math.min(row, nextLayer.rows - 1));
    setPhaseIndex(0);
  };

  const movePhase = (delta: number) => {
    setPhaseIndex((index) => Math.max(0, Math.min(fsdpPhases.length - 1, index + delta)));
  };

  const operationName = phase.stage === "all-gather"
    ? "All-Gather 权重分片"
    : phase.stage === "reduce-scatter"
      ? "Reduce-Scatter 完整梯度"
      : phase.label;

  return (
    <section className="distributed-playground fsdp-playground" id="fsdp-playground" aria-label="FSDP 参数生命周期交互实验台">
      <header className="distributed-playground-header">
        <div>
          <span>FSDP executable model</span>
          <h2>选一层、一个权重行，观察它何时完整、何时只剩分片</h2>
          <p>这里模拟你实现里的逐层 hook、行分片、padding、通信 dtype 和梯度 Reduce-Scatter。</p>
        </div>
        <div className="distributed-facts" aria-label="模拟配置">
          <span><strong>{worldSize}</strong> ranks</span>
          <span><strong>{layer.rows} × {layer.columns}</strong> weight</span>
          <span><strong>{computeDtype}</strong> compute</span>
          <span><strong>FP32</strong> master shard</span>
        </div>
      </header>

      <div className="distributed-control-bar fsdp-controls">
        <div className="compact-control">
          <span>World size</span>
          <div role="group" aria-label="选择 rank 数量">
            {[2, 4].map((size) => (
              <button type="button" className={worldSize === size ? "is-active" : ""} onClick={() => changeWorldSize(size)} key={size}>{size} ranks</button>
            ))}
          </div>
        </div>

        <div className="compact-control">
          <span>通信与计算 dtype</span>
          <div role="group" aria-label="选择计算精度">
            {(["fp32", "fp16"] as ComputeDtype[]).map((dtype) => (
              <button type="button" className={computeDtype === dtype ? "is-active" : ""} onClick={() => setComputeDtype(dtype)} key={dtype}>{dtype}</button>
            ))}
          </div>
        </div>

        <div className="compact-control rank-control">
          <span>检查 rank</span>
          <div role="group" aria-label="选择要检查的 rank">
            {Array.from({ length: worldSize }, (_, rank) => (
              <button type="button" className={selectedRank === rank ? "is-active" : ""} onClick={() => setSelectedRank(rank)} key={rank}>R{rank}</button>
            ))}
          </div>
        </div>

        <div className="phase-stepper">
          <button type="button" onClick={() => setPhaseIndex(0)} aria-label="重置参数生命周期"><ArrowCounterClockwise size={16} /></button>
          <button type="button" onClick={() => movePhase(-1)} disabled={phaseIndex === 0} aria-label="上一步"><CaretLeft size={16} /></button>
          <span><strong>{phase.label}</strong><small>{phaseIndex + 1} / {fsdpPhases.length}</small></span>
          <button type="button" onClick={() => movePhase(1)} disabled={phaseIndex === fsdpPhases.length - 1} aria-label="下一步"><CaretRight size={16} /></button>
        </div>
      </div>

      <div className="layer-selector" aria-label="选择模型层">
        <span>当前计算层</span>
        <div>
          {fsdpLayers.map((item) => (
            <button type="button" className={layer.id === item.id ? "is-active" : ""} onClick={() => changeLayer(item.id)} key={item.id}>
              <strong>{item.shortName}</strong><small>{item.rows} × {item.columns}</small>
            </button>
          ))}
        </div>
        <p>{layer.use}</p>
      </div>

      <div className="fsdp-layer-schedule" aria-label="逐层执行顺序">
        <span>Forward</span>
        {fsdpLayers.map((item) => <i className={phase.pass === "forward" && item.id === layer.id ? "is-active" : ""} key={`f-${item.id}`}>{item.shortName}</i>)}
        <b>Loss</b>
        <span>Backward</span>
        {[...fsdpLayers].reverse().map((item) => <i className={phase.pass === "backward" && item.id === layer.id ? "is-active" : ""} key={`b-${item.id}`}>{item.shortName}</i>)}
      </div>

      <div className="distributed-phase-rail fsdp-phase-rail" role="tablist" aria-label="FSDP 参数生命周期">
        {fsdpPhases.map((item, index) => (
          <button
            type="button"
            role="tab"
            aria-selected={phaseIndex === index}
            className={phaseIndex === index ? "is-active" : index < phaseIndex ? "is-complete" : ""}
            onClick={() => setPhaseIndex(index)}
            key={item.id}
          ><span>{item.compactLabel}</span><small>{item.pass}</small></button>
        ))}
      </div>

      <div className={`collective-operation-band phase-${phase.stage}`}>
        <div className="operation-icon" aria-hidden="true">
          {phase.stage === "all-gather" || phase.stage === "reduce-scatter"
            ? <ArrowsOut size={23} weight="duotone" />
            : <Stack size={23} weight="duotone" />}
        </div>
        <div><strong>{operationName}</strong><p>{phase.explanation}</p></div>
        <code>
          {phase.stage === "all-gather"
            ? `${formatBytes(allGatherPayloadBytes(layer, worldSize, computeDtype))} contribution / rank`
            : phase.stage === "reduce-scatter"
              ? `SUM ÷ ${worldSize} -> ${shardRowCount} rows / rank`
              : `parameter.data = ${phase.fullParameter ? "full_weight" : "local_shard"}`}
        </code>
      </div>

      <DistributedHardwarePath
        snapshot={hardwareSnapshot}
        title={`${layer.shortName}：当前阶段的软件与硬件路径`}
      />

      <div className="row-tracker">
        <div><span>追踪权重行</span><strong>Row {selectedRow}</strong><small>长期 owner: Rank {selectedRowOwner}</small></div>
        <div className="row-buttons" role="group" aria-label="选择要追踪的权重行">
          {Array.from({ length: layer.rows }, (_, row) => (
            <button type="button" className={selectedRow === row ? "is-active" : ""} onClick={() => setSelectedRow(row)} key={row}>{row}</button>
          ))}
        </div>
        <div className="row-presence" aria-label={`Row ${selectedRow} 当前所在位置`}>
          {rankStates.map((rankState) => {
            const hasWeight = rankState.visibleWeightRows.includes(selectedRow);
            const gradient = rankState.gradientRows.find((row) => row.rowIndex === selectedRow);
            return (
              <button type="button" className={`${hasWeight ? "has-weight" : ""}${gradient ? " has-gradient" : ""}${selectedRank === rankState.rank ? " is-selected" : ""}`} onClick={() => setSelectedRank(rankState.rank)} key={rankState.rank}>
                <strong>R{rankState.rank}</strong>
                <span>{hasWeight ? `W=${layer.baseValue + selectedRow}` : "W=空"}</span>
                <small>{gradient ? `dW=${gradient.value}` : "dW=空"}</small>
              </button>
            );
          })}
        </div>
      </div>

      <div className="fsdp-main-grid">
        <div className={`fsdp-rank-grid ranks-${worldSize}`}>
          {rankStates.map((rankState) => (
            <article className={`fsdp-rank-card${selectedRank === rankState.rank ? " is-selected" : ""}`} key={rankState.rank}>
              <header>
                <button type="button" onClick={() => setSelectedRank(rankState.rank)}><span>Rank {rankState.rank}</span><strong>GPU {rankState.rank} HBM</strong></button>
                <span>{rankState.fullParameter ? "unsharded" : "sharded"}</span>
              </header>

              <section className="hbm-region persistent-region">
                <div><span>长期保存</span><strong>FP32 master shard</strong></div>
                <div className="shard-row-grid">
                  {rankState.localRows.map((row) => (
                    <i className={`${row.isPadding ? "is-padding" : ""}${row.rowIndex === selectedRow ? " is-tracked" : ""}`} key={row.paddedIndex}>
                      {row.isPadding ? "pad 0" : `R${row.rowIndex}: ${row.value}`}
                    </i>
                  ))}
                </div>
              </section>

              <section className={`hbm-region transient-region${rankState.fullParameter ? " is-present" : ""}`}>
                <div><span>临时 buffer</span><strong>完整 W · {rankState.weightDtype}</strong></div>
                {rankState.fullParameter ? (
                  <div className="full-row-grid">
                    {rankState.visibleWeightRows.map((row) => <i className={row === selectedRow ? "is-tracked" : ""} key={row}>{row}</i>)}
                  </div>
                ) : <p>未分配。当前层完整权重已释放。</p>}
              </section>

              <section className={`hbm-region gradient-region${rankState.gradientRows.length > 0 ? " is-present" : ""}`}>
                <div><span>梯度 buffer</span><strong>{phase.fullGradient ? "完整 local dW" : phase.shardedGradient ? "平均 dW shard" : "尚未产生"}</strong></div>
                {rankState.gradientRows.length > 0 ? (
                  <div className="gradient-row-list">
                    {rankState.gradientRows.slice(0, 5).map((row) => <i className={row.rowIndex === selectedRow ? "is-tracked" : ""} key={row.rowIndex}>R{row.rowIndex}={row.value}</i>)}
                    {rankState.gradientRows.length > 5 && <i>+{rankState.gradientRows.length - 5} rows</i>}
                  </div>
                ) : <p>等待 backward。</p>}
              </section>
            </article>
          ))}
        </div>

        <aside className="distributed-inspector fsdp-inspector">
          <div className="inspector-section-title"><span>当前层</span><strong>{layer.name}</strong></div>

          <dl className="owner-equation shard-equation">
            <div><dt>原始 shape</dt><dd>[{layer.rows}, {layer.columns}]</dd></div>
            <div><dt>每 rank 行数</dt><dd>ceil({layer.rows} / {worldSize}) = {shardRowCount}</dd></div>
            <div><dt>padding 后</dt><dd>{paddedRows(layer, worldSize)} rows</dd></div>
            <div className="owner-result"><dt>Row {selectedRow} owner</dt><dd>Rank {selectedRowOwner}</dd></div>
          </dl>

          <section className="inspector-data-block">
            <span>Rank {selectedRank} 此刻 parameter.data</span>
            <div className="pointer-switch">
              <code>parameter.data</code><strong>→</strong><code>{phase.fullParameter ? `full_weight (${computeDtype})` : "local_shard (fp32)"}</code>
            </div>
          </section>

          <section className="inspector-data-block memory-snapshot">
            <span>当前层在每个 rank 的教学内存快照</span>
            <div><i className="memory-param" /><code>master shard</code><strong>{formatBytes(memory.persistentWeightBytes)}</strong></div>
            <div><i className="memory-state" /><code>Adam m、v shard</code><strong>{formatBytes(memory.optimizerStateBytes)}</strong></div>
            <div><i className="memory-grad" /><code>{phase.fullGradient ? "临时完整 dW" : "gradient shard"}</code><strong>{formatBytes(memory.persistentGradientBytes + memory.transientFullGradientBytes)}</strong></div>
            <div><i className="memory-transient" /><code>临时完整 W</code><strong>{formatBytes(memory.transientFullWeightBytes)}</strong></div>
          </section>

          {phase.fullGradient && (
            <p className="inspector-callout">各 rank 的 local batch 不同，所以此刻完整 dW 不同。Row {selectedRow} 分别贡献 {rankStates.map((state) => state.gradientRows.find((row) => row.rowIndex === selectedRow)?.value).join("、")}。</p>
          )}
          {phase.shardedGradient && (
            <p className="inspector-callout">Reduce-Scatter 后，Row {selectedRow} 的全局平均梯度是 {averagedGradient(worldSize, selectedRow)}，只保留在 Rank {selectedRowOwner}。</p>
          )}
        </aside>
      </div>

      <section className="memory-ledger fsdp-memory-ledger" aria-labelledby="fsdp-memory-title">
        <div className="memory-ledger-heading"><HardDrives size={22} weight="duotone" /><div><h3 id="fsdp-memory-title">同一层状态如何从 4P 降到约 4P/N</h3><p>以下持久内存假设 FP32 参数、梯度和 AdamW m、v；FSDP 额外显示当前阶段的临时完整权重。</p></div></div>
        <div className="three-way-memory">
          <article><span>DDP</span><strong>{formatBytes(ddpBytes)}</strong><code>参数 P + 梯度 P + 状态 2P</code></article>
          <article><span>ZeRO-1 平均</span><strong>{formatBytes(zeroAverageBytes)}</strong><code>参数 P + 梯度 P + 状态 2P/N</code></article>
          <article className="is-emphasized"><span>FSDP 持久 shard</span><strong>{formatBytes(fsdpPersistentAdamBytes)}</strong><code>(参数 + 梯度 + 状态) / N，含 padding</code></article>
          <aside><Memory size={19} /><span>当前临时完整 W</span><strong>+ {formatBytes(memory.transientFullWeightBytes)}</strong></aside>
        </div>
      </section>

      <section className="implementation-evidence" aria-labelledby="fsdp-source-title">
        <div className="evidence-source-heading"><Code size={21} /><div><h3 id="fsdp-source-title">FSDP 实现契约</h3><p>从 shard 生命周期、hook 到 checkpoint 重建的关键机制。</p></div></div>
        <div className="source-contract-grid fsdp-source-grid">
          <article><span>初始化切分</span><code>parameter.data = local_shard</code><p>Linear、Embedding weight 沿第 0 维切分，必要时补零。</p></article>
          <article><span>Forward pre-hook</span><code>self._unshard(state)</code><p>All-Gather 当前层权重，临时切换 parameter.data。</p></article>
          <article><span>Forward hook</span><code>self._reshard(state)</code><p>算完立即释放完整权重，并注册 backward 前的再次聚合。</p></article>
          <article><span>Gradient hook</span><code>reduce_scatter_tensor(..., SUM)</code><p>平均完整梯度并只返回与本地权重同形状的 shard。</p></article>
          <article><span>小参数</span><code>dist.all_reduce(parameter.grad)</code><p>RMSNorm 等复制参数不切分，梯度仍使用普通 All-Reduce。</p></article>
          <article><span>校验与 checkpoint</span><code>gather_full_parameters()</code><p>用 FP32 master shards 临时重建完整参数字典。</p></article>
        </div>
      </section>
    </section>
  );
}
