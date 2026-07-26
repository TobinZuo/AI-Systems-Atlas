import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { ArrowsClockwise } from "@phosphor-icons/react/ArrowsClockwise";
import { Broadcast } from "@phosphor-icons/react/Broadcast";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Scales } from "@phosphor-icons/react/Scales";
import { useMemo, useState, type CSSProperties } from "react";
import {
  comparisonModelPresets,
  comparisonPhaseInsights,
  distributedCommunicationSnapshots,
  distributedMemorySnapshots,
  formatComparisonBytes,
  recommendDistributedStrategy,
  type ComparisonPhaseId,
  type DistributedStrategyId,
  type StrategyMemorySnapshot,
  type TrainingConstraintId,
} from "../playground/distributedComparison";
import { comparisonHardwareSnapshot } from "../playground/distributedHardware";
import { DistributedHardwarePath } from "./DistributedHardwarePath";

const phaseOptions: Array<{ id: ComparisonPhaseId; label: string; summary: string }> = [
  { id: "persistent", label: "长期驻留", summary: "训练大部分时间每卡保存什么" },
  { id: "gradient-sync", label: "梯度同步", summary: "local dW 怎样变成 global dW" },
  { id: "optimizer-step", label: "参数更新", summary: "谁运行 AdamW，更新后是否同步" },
  { id: "next-forward", label: "下一次计算", summary: "Kernel 怎样拿到需要的完整权重" },
];

const constraintOptions: Array<{ id: TrainingConstraintId; label: string; detail: string }> = [
  { id: "fits", label: "完整状态能放下", detail: "优先考虑简单和吞吐" },
  { id: "optimizer-pressure", label: "主要卡在 m、v", detail: "参数和梯度仍能放下" },
  { id: "model-pressure", label: "参数或梯度也放不下", detail: "需要进一步切分模型状态" },
];

const detailRoutes: Record<DistributedStrategyId, string> = {
  ddp: "/distributed/ddp",
  "zero-1": "/distributed/zero-1",
  fsdp: "/distributed/fsdp",
};

function memoryFormula(strategy: DistributedStrategyId): string {
  if (strategy === "ddp") return "P + P + 2P = 4P";
  if (strategy === "zero-1") return "P + P + 2P/N";
  return "P/N + P/N + 2P/N = 4P/N";
}

function stateFractionLabel(fraction: number): string {
  return fraction === 1 ? "完整 P" : "P/N";
}

function MemoryTape({
  snapshot,
  ddpTotalBytes,
}: {
  snapshot: StrategyMemorySnapshot;
  ddpTotalBytes: number;
}) {
  const tapeStyle = {
    "--memory-ratio": snapshot.totalBytes / ddpTotalBytes,
  } as CSSProperties;

  return (
    <div className="comparison-memory-viewport" aria-label={`${snapshot.label} 每卡持久模型状态`}>
      <div className="comparison-memory-tape" style={tapeStyle}>
        <span className="memory-param" style={{ flex: snapshot.parameterFraction }}>
          <strong>参数 W</strong><small>{stateFractionLabel(snapshot.parameterFraction)}</small>
        </span>
        <span className="memory-grad" style={{ flex: snapshot.gradientFraction }}>
          <strong>梯度 dW</strong><small>{stateFractionLabel(snapshot.gradientFraction)}</small>
        </span>
        <span className="memory-state" style={{ flex: snapshot.optimizerStateFraction * 2 }}>
          <strong>AdamW m、v</strong><small>{snapshot.optimizerStateFraction === 1 ? "完整 2P" : "2P/N"}</small>
        </span>
      </div>
    </div>
  );
}

export function DistributedComparisonPlayground() {
  const [worldSize, setWorldSize] = useState(4);
  const [modelPresetId, setModelPresetId] = useState("1b");
  const [phaseId, setPhaseId] = useState<ComparisonPhaseId>("persistent");
  const [constraintId, setConstraintId] = useState<TrainingConstraintId>("fits");
  const [hardwareStrategyId, setHardwareStrategyId] = useState<DistributedStrategyId>("ddp");

  const modelPreset = comparisonModelPresets.find((preset) => preset.id === modelPresetId)!;
  const memories = useMemo(
    () => distributedMemorySnapshots(modelPreset.parameterCount, worldSize),
    [modelPreset, worldSize],
  );
  const communications = useMemo(
    () => distributedCommunicationSnapshots(modelPreset.parameterCount, worldSize),
    [modelPreset, worldSize],
  );
  const phaseInsights = comparisonPhaseInsights(phaseId);
  const recommendation = recommendDistributedStrategy(constraintId);
  const ddpTotalBytes = memories[0].totalBytes;
  const hardwareSnapshot = comparisonHardwareSnapshot(hardwareStrategyId, phaseId);

  return (
    <section className="distributed-playground comparison-playground" id="distributed-comparison-playground" aria-label="DDP、ZeRO-1 和 FSDP 横向对比实验台">
      <header className="distributed-playground-header">
        <div>
          <span>Distributed strategy comparator</span>
          <h2>固定同一模型和 GPU 数量，只改变状态切分策略</h2>
          <p>先比较每卡长期驻留状态，再沿一次训练步骤观察通信与参数更新。</p>
        </div>
        <div className="distributed-facts" aria-label="对比条件">
          <span><strong>{modelPreset.label}</strong>模型规模</span>
          <span><strong>{worldSize}</strong>ranks</span>
          <span><strong>FP32</strong>状态口径</span>
          <span><strong>AdamW</strong>m + v</span>
        </div>
      </header>

      <div className="distributed-control-bar comparison-controls">
        <div className="compact-control model-size-control">
          <span>模型参数量</span>
          <div role="group" aria-label="选择模型参数量">
            {comparisonModelPresets.map((preset) => (
              <button
                type="button"
                className={modelPresetId === preset.id ? "is-active" : ""}
                onClick={() => setModelPresetId(preset.id)}
                key={preset.id}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <div className="compact-control comparison-world-control">
          <span>World size</span>
          <div role="group" aria-label="选择 GPU 数量">
            {[2, 4, 8].map((size) => (
              <button
                type="button"
                className={worldSize === size ? "is-active" : ""}
                onClick={() => setWorldSize(size)}
                key={size}
              >
                {size} GPUs
              </button>
            ))}
          </div>
        </div>
        <div className="comparison-assumption">
          <HardDrives size={19} weight="duotone" aria-hidden="true" />
          <span>只计算参数、梯度、AdamW m、v。Activation、padding、allocator 和临时完整 W 另算。</span>
        </div>
      </div>

      <section className="comparison-memory-section" aria-labelledby="comparison-memory-title">
        <header>
          <div><span>每个 rank 的长期模型状态</span><h3 id="comparison-memory-title">哪个状态在切，显存就从哪里省下来</h3></div>
          <code>P = {formatComparisonBytes(modelPreset.parameterCount * 4)}</code>
        </header>
        <div className="strategy-lanes">
          {memories.map((snapshot) => (
            <article className={`strategy-lane strategy-${snapshot.id}`} key={snapshot.id}>
              <header>
                <div><strong>{snapshot.label}</strong><span>{snapshot.fullName}</span></div>
                <div><strong>{formatComparisonBytes(snapshot.totalBytes)}</strong><span>每卡持久状态</span></div>
              </header>
              <MemoryTape snapshot={snapshot} ddpTotalBytes={ddpTotalBytes} />
              <footer>
                <code>{memoryFormula(snapshot.id)}</code>
                <span>{snapshot.updateOwner}</span>
                <a href={`#${detailRoutes[snapshot.id]}`}>进入独立实验台<ArrowRight size={14} /></a>
              </footer>
            </article>
          ))}
        </div>
        <p className="comparison-qualification">
          这是均匀可分片的教学模型。真实 ZeRO-1 会受参数大小和 owner 分配影响，真实 FSDP 还会受 wrap 粒度、padding、预取与临时 buffer 影响。
        </p>
      </section>

      <section className="comparison-phase-section" aria-labelledby="comparison-phase-title">
        <header>
          <div><span>同一时刻对照</span><h3 id="comparison-phase-title">选择一个训练阶段，看三种策略怎样处理同一份数据</h3></div>
        </header>
        <div className="comparison-phase-tabs" role="tablist" aria-label="选择训练阶段">
          {phaseOptions.map((phase) => (
            <button
              type="button"
              role="tab"
              aria-selected={phaseId === phase.id}
              className={phaseId === phase.id ? "is-active" : ""}
              onClick={() => setPhaseId(phase.id)}
              key={phase.id}
            >
              <strong>{phase.label}</strong><small>{phase.summary}</small>
            </button>
          ))}
        </div>
        <div className="phase-comparison-rows">
          {phaseInsights.map((insight) => (
            <article className={`phase-comparison-row strategy-${insight.id}`} key={insight.id}>
              <div className="phase-strategy-name"><strong>{insight.label}</strong><span>{insight.title}</span></div>
              <p>{insight.explanation}</p>
              <dl>
                <div><dt>数据状态</dt><dd>{insight.dataState}</dd></div>
                <div><dt>通信动作</dt><dd>{insight.communication}</dd></div>
              </dl>
            </article>
          ))}
        </div>
        <div className="comparison-hardware-selector">
          <div>
            <span>下钻一个策略</span>
            <div role="group" aria-label="选择要下钻到硬件的策略">
              {(["ddp", "zero-1", "fsdp"] as DistributedStrategyId[]).map((strategy) => (
                <button
                  type="button"
                  className={hardwareStrategyId === strategy ? "is-active" : ""}
                  onClick={() => setHardwareStrategyId(strategy)}
                  key={strategy}
                >
                  {strategy === "zero-1" ? "ZeRO-1" : strategy.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <DistributedHardwarePath
            snapshot={hardwareSnapshot}
            title={`${hardwareStrategyId === "zero-1" ? "ZeRO-1" : hardwareStrategyId.toUpperCase()}：${phaseOptions.find((phase) => phase.id === phaseId)!.label}的硬件路径`}
          />
        </div>
      </section>

      <section className="comparison-communication-section" aria-labelledby="comparison-communication-title">
        <header>
          <div><ArrowsClockwise size={21} weight="duotone" aria-hidden="true" /><span><strong id="comparison-communication-title">一次训练迭代的模型状态通信</strong><small>按每 rank 接收的应用层 payload 估算</small></span></div>
          <code>N={worldSize}，P={formatComparisonBytes(modelPreset.parameterCount * 4)}</code>
        </header>
        <div className="communication-ledger">
          {communications.map((profile) => (
            <article className={`communication-row strategy-${profile.id}`} key={profile.id}>
              <header><strong>{profile.id === "zero-1" ? "ZeRO-1" : profile.id.toUpperCase()}</strong><span>{formatComparisonBytes(profile.totalReceiveBytes)} / rank</span></header>
              <div className="communication-sequence" aria-label={`${profile.id} 通信顺序`}>
                {profile.sequence.map((item, index) => (
                  <span key={item}><i>{index + 1}</i><strong>{item}</strong>{index < profile.sequence.length - 1 && <ArrowRight size={14} aria-hidden="true" />}</span>
                ))}
              </div>
              <dl>
                <div><dt>梯度 payload</dt><dd>{formatComparisonBytes(profile.gradientReceiveBytes)}</dd></div>
                <div><dt>参数 payload</dt><dd>{formatComparisonBytes(profile.parameterReceiveBytes)}</dd></div>
                <div><dt>公式</dt><dd><code>{profile.formula}</code></dd></div>
              </dl>
              <p>{profile.qualification}</p>
            </article>
          ))}
        </div>
        <p className="comparison-qualification">
          这些数字描述 collective 的逻辑 payload，不是端到端耗时。NCCL 算法、拓扑、分桶、重叠执行与链路竞争都会改变真实时间。
        </p>
      </section>

      <section className="comparison-decision-section" aria-labelledby="comparison-decision-title">
        <header>
          <Scales size={22} weight="duotone" aria-hidden="true" />
          <div><h3 id="comparison-decision-title">先看哪个策略</h3><p>这不是自动选型器，只按最先撞到的模型状态瓶颈给出学习起点。</p></div>
        </header>
        <div className="constraint-choices" role="group" aria-label="选择当前显存约束">
          {constraintOptions.map((option) => (
            <button
              type="button"
              className={constraintId === option.id ? "is-active" : ""}
              onClick={() => setConstraintId(option.id)}
              key={option.id}
            >
              <strong>{option.label}</strong><small>{option.detail}</small>
            </button>
          ))}
        </div>
        <div className={`strategy-recommendation strategy-${recommendation.strategy}`} aria-live="polite">
          <div><Broadcast size={21} weight="duotone" aria-hidden="true" /><strong>{recommendation.label}</strong></div>
          <p>{recommendation.reason}</p>
          <span><b>需要接受的代价</b>{recommendation.tradeoff}</span>
          <a href={`#${detailRoutes[recommendation.strategy]}`}>打开 {recommendation.strategy === "zero-1" ? "ZeRO-1" : recommendation.strategy.toUpperCase()} 细节<ArrowRight size={15} /></a>
        </div>
      </section>
    </section>
  );
}
