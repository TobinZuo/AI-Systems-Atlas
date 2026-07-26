import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { Broadcast } from "@phosphor-icons/react/Broadcast";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Code } from "@phosphor-icons/react/Code";
import { Database } from "@phosphor-icons/react/Database";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { UsersFour } from "@phosphor-icons/react/UsersFour";
import { useMemo, useState } from "react";
import {
  adamWFirstStep,
  ddpOptimizerMemoryBytes,
  formatBytes,
  optimizerGroups,
  optimizerMemoryByRank,
  ownerForParameter,
  parameterValueAtPhase,
  shardedOptimizerParameters,
  shardedOptimizerPhases,
} from "../playground/shardedOptimizer";

const formatValue = (value: number) => value.toFixed(4);

export function ShardedOptimizerPlayground() {
  const [worldSize, setWorldSize] = useState(2);
  const [selectedRank, setSelectedRank] = useState(0);
  const [selectedParameterId, setSelectedParameterId] = useState("tied-weight");
  const [phaseIndex, setPhaseIndex] = useState(0);

  const phase = shardedOptimizerPhases[phaseIndex];
  const selectedParameter = shardedOptimizerParameters.find(
    (parameter) => parameter.id === selectedParameterId,
  )!;
  const selectedOwner = ownerForParameter(selectedParameter.index, worldSize);
  const selectedGroup = optimizerGroups.find((group) => group.id === selectedParameter.groupId)!;
  const update = adamWFirstStep(selectedParameter);
  const rankMemories = useMemo(() => optimizerMemoryByRank(worldSize), [worldSize]);
  const selectedMemory = rankMemories[selectedRank];
  const activeBroadcastParameter = phase.kind === "broadcast"
    ? shardedOptimizerParameters[phase.broadcastParameterIndex ?? 0]
    : null;

  const changeWorldSize = (nextWorldSize: number) => {
    setWorldSize(nextWorldSize);
    setSelectedRank((rank) => Math.min(rank, nextWorldSize - 1));
    setPhaseIndex(0);
  };

  const movePhase = (delta: number) => {
    setPhaseIndex((index) => Math.max(0, Math.min(shardedOptimizerPhases.length - 1, index + delta)));
  };

  return (
    <section className="distributed-playground zero-playground" id="sharded-optimizer-playground" aria-label="ZeRO-1 Sharded Optimizer 交互实验台">
      <header className="distributed-playground-header">
        <div>
          <span>ZeRO-1 executable model</span>
          <h2>看清每份状态在哪个 rank，以及参数何时重新一致</h2>
          <p>参数值、AdamW 首步更新、owner 归属和显存数字都由纯函数实时计算。</p>
        </div>
        <div className="distributed-facts" aria-label="模拟配置">
          <span><strong>{worldSize}</strong> ranks</span>
          <span><strong>{shardedOptimizerParameters.length}</strong> unique params</span>
          <span><strong>2</strong> parameter groups</span>
          <span><strong>m + v</strong> sharded</span>
        </div>
      </header>

      <div className="distributed-control-bar">
        <div className="compact-control">
          <span>World size</span>
          <div role="group" aria-label="选择 rank 数量">
            {[2, 4].map((size) => (
              <button type="button" className={worldSize === size ? "is-active" : ""} onClick={() => changeWorldSize(size)} key={size}>
                {size} ranks
              </button>
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
          <button type="button" onClick={() => setPhaseIndex(0)} aria-label="重置到梯度同步完成">
            <ArrowCounterClockwise size={16} />
          </button>
          <button type="button" onClick={() => movePhase(-1)} disabled={phaseIndex === 0} aria-label="上一步">
            <CaretLeft size={16} />
          </button>
          <span><strong>{phase.label}</strong><small>{phaseIndex + 1} / {shardedOptimizerPhases.length}</small></span>
          <button type="button" onClick={() => movePhase(1)} disabled={phaseIndex === shardedOptimizerPhases.length - 1} aria-label="下一步">
            <CaretRight size={16} />
          </button>
        </div>
      </div>

      <div className="distributed-phase-rail" role="tablist" aria-label="ZeRO-1 执行阶段">
        {shardedOptimizerPhases.map((item, index) => (
          <button
            type="button"
            role="tab"
            aria-selected={phaseIndex === index}
            className={phaseIndex === index ? "is-active" : index < phaseIndex ? "is-complete" : ""}
            onClick={() => setPhaseIndex(index)}
            key={item.id}
          >
            <span>{item.compactLabel}</span>
          </button>
        ))}
      </div>

      <div className={`collective-operation-band phase-${phase.kind}`}>
        <div className="operation-icon" aria-hidden="true">
          {phase.kind === "broadcast" ? <Broadcast size={23} weight="duotone" /> : <Database size={23} weight="duotone" />}
        </div>
        <div>
          <strong>{activeBroadcastParameter ? `所有 rank 调用 broadcast(${activeBroadcastParameter.shortName})` : phase.label}</strong>
          <p>{phase.explanation}</p>
        </div>
        <code>
          {activeBroadcastParameter
            ? `src = ${activeBroadcastParameter.index} % ${worldSize} = ${ownerForParameter(activeBroadcastParameter.index, worldSize)}`
            : phase.kind === "owner-update"
              ? "local_optimizer.step()"
              : "same collective order on every rank"}
        </code>
      </div>

      <div className="parameter-selector" aria-label="选择参数">
        <span>追踪参数</span>
        <div>
          {shardedOptimizerParameters.map((parameter) => {
            const owner = ownerForParameter(parameter.index, worldSize);
            return (
              <button
                type="button"
                className={parameter.id === selectedParameter.id ? "is-active" : ""}
                onClick={() => setSelectedParameterId(parameter.id)}
                key={parameter.id}
              >
                <strong>{parameter.shortName}</strong>
                <small>owner R{owner}</small>
              </button>
            );
          })}
        </div>
      </div>

      <div className="zero-main-grid">
        <div className={`zero-rank-grid ranks-${worldSize}`} aria-label="各 rank 的参数和 optimizer state">
          {Array.from({ length: worldSize }, (_, rank) => {
            const memory = rankMemories[rank];
            return (
              <article className={`optimizer-rank-card${selectedRank === rank ? " is-selected" : ""}`} key={rank}>
                <header>
                  <button type="button" onClick={() => setSelectedRank(rank)}>
                    <span>Rank {rank}</span>
                    <strong>Local AdamW</strong>
                  </button>
                  <span>{memory.ownedParameterIds.length} owned</span>
                </header>

                <div className="rank-state-legend">
                  <span>完整 parameter + grad</span>
                  <span>m、v 仅 owner</span>
                </div>

                <div className="rank-parameter-grid">
                  {shardedOptimizerParameters.map((parameter) => {
                    const owner = ownerForParameter(parameter.index, worldSize);
                    const value = parameterValueAtPhase(parameter, rank, worldSize, phase);
                    const isUpdated = Math.abs(value - parameter.value) > 1e-9;
                    const isCurrentBroadcast = activeBroadcastParameter?.id === parameter.id;
                    return (
                      <button
                        type="button"
                        className={`${parameter.id === selectedParameter.id ? "is-selected" : ""}${owner === rank ? " is-owner" : ""}${isUpdated ? " is-updated" : ""}${isCurrentBroadcast ? " is-broadcasting" : ""}`}
                        onClick={() => {
                          setSelectedParameterId(parameter.id);
                          setSelectedRank(rank);
                        }}
                        key={parameter.id}
                      >
                        <span><strong>{parameter.shortName}</strong><i>{formatValue(value)}</i></span>
                        <small>{owner === rank ? "parameter / grad / m / v" : "parameter / grad"}</small>
                      </button>
                    );
                  })}
                </div>

                <footer>
                  <span>本 rank 状态</span>
                  <strong>{formatBytes(memory.optimizerStateBytes)}</strong>
                </footer>
              </article>
            );
          })}
        </div>

        <aside className="distributed-inspector zero-inspector">
          <div className="inspector-section-title">
            <span>当前参数</span>
            <strong>{selectedParameter.name}</strong>
          </div>

          <dl className="owner-equation">
            <div><dt>全局 index</dt><dd>{selectedParameter.index}</dd></div>
            <div><dt>owner 规则</dt><dd>{selectedParameter.index} % {worldSize}</dd></div>
            <div className="owner-result"><dt>唯一 owner</dt><dd>Rank {selectedOwner}</dd></div>
          </dl>

          <section className="inspector-data-block">
            <span>Parameter group 保留下来的配置</span>
            <div className="group-config">
              <strong>{selectedGroup.label}</strong>
              <code>lr={selectedGroup.learningRate}</code>
              <code>weight_decay={selectedGroup.weightDecay}</code>
            </div>
          </section>

          <section className="inspector-data-block adam-calculation">
            <span>Owner 上的 AdamW 首步</span>
            <div><code>g</code><strong>{formatValue(selectedParameter.gradient)}</strong></div>
            <div><code>m</code><strong>{formatValue(update.expAvg)}</strong></div>
            <div><code>v</code><strong>{update.expAvgSq.toFixed(6)}</strong></div>
            <div><code>parameter</code><strong>{formatValue(selectedParameter.value)} → {formatValue(update.nextValue)}</strong></div>
          </section>

          <section className="inspector-data-block selected-rank-inventory">
            <span>Rank {selectedRank} 实际保存什么</span>
            <div><i className="is-present" />完整参数副本</div>
            <div><i className="is-present" />完整梯度副本</div>
            <div><i className={selectedRank === selectedOwner ? "is-present" : ""} />exp_avg (m)</div>
            <div><i className={selectedRank === selectedOwner ? "is-present" : ""} />exp_avg_sq (v)</div>
          </section>

          {selectedParameter.note && <p className="inspector-callout">{selectedParameter.note}</p>}
        </aside>
      </div>

      <section className="memory-ledger" aria-labelledby="zero-memory-title">
        <div className="memory-ledger-heading">
          <HardDrives size={22} weight="duotone" />
          <div><h3 id="zero-memory-title">为什么显存会下降</h3><p>这个 toy model 使用 FP32 AdamW，不额外计算 mixed-precision master weight。</p></div>
        </div>
        <div className="memory-comparison">
          <article>
            <header><span>普通 DDP · 每个 rank</span><strong>{formatBytes(ddpOptimizerMemoryBytes())}</strong></header>
            <div className="memory-blocks four-blocks">
              <span className="memory-param">参数 P</span><span className="memory-grad">梯度 P</span><span className="memory-state">m P</span><span className="memory-state">v P</span>
            </div>
            <code>P + P + 2P = 4P</code>
          </article>
          <article className="is-emphasized">
            <header><span>ZeRO-1 · Rank {selectedRank}</span><strong>{formatBytes(selectedMemory.totalBytes)}</strong></header>
            <div className="memory-blocks zero-blocks">
              <span className="memory-param">完整参数</span><span className="memory-grad">完整梯度</span><span className="memory-state">本地 m、v shard</span>
            </div>
            <code>P + P + 当前 rank 拥有参数的 2P state</code>
          </article>
        </div>
      </section>

      <section className="implementation-evidence" aria-labelledby="zero-source-title">
        <div className="evidence-source-heading"><Code size={21} /><div><h3 id="zero-source-title">对应 assignment2 实现</h3><code className="source-file-path">cs336_systems/sharded_optimizer.py</code></div></div>
        <div className="source-contract-grid">
          <article><span>分配 owner</span><code>owner = parameter_index % world_size</code><p>全局 index 跨 parameter group 连续递增。</p></article>
          <article><span>创建本地 optimizer</span><code>optimizer_cls(local_param_groups, **kwargs)</code><p>只把本 rank 拥有的参数交给 AdamW，同时保留每组超参数。</p></article>
          <article><span>恢复模型一致</span><code>dist.broadcast(parameter.detach(), src=owner)</code><p>所有 rank 以完全相同的参数顺序调用 collective。</p></article>
          <article><span>保存 checkpoint</span><code>_local_optimizer.state_dict()</code><p>每个 rank 只保存自己拥有的 optimizer-state shard。</p></article>
        </div>
      </section>
    </section>
  );
}
