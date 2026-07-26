import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Cube } from "@phosphor-icons/react/Cube";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { SlidersHorizontal } from "@phosphor-icons/react/SlidersHorizontal";
import { Stack } from "@phosphor-icons/react/Stack";
import { useEffect, useMemo, useState } from "react";
import {
  copyPreset,
  gradientPresets,
  playgroundSteps,
  ranksForStep,
  roundForStep,
  stepIndexForConceptEvent,
  type PlaygroundStep,
} from "../playground/ddp";
import { simulateRingAllReduce, type RingTransfer } from "../sim/ring";

interface StageCopy {
  title: string;
  summary: string;
  cause: string;
  action: string;
  result: string;
  idea: string;
}

const flowNodes = [
  { label: "GPU backward", detail: "产生本地梯度", kind: "compute", jumpTo: 0 },
  { label: "DDP bucket", detail: "连续显存缓冲区", kind: "framework", jumpTo: 0 },
  { label: "Reduce-Scatter", detail: "传输并逐元素求和", kind: "collective", jumpTo: 1 },
  { label: "All-Gather", detail: "传播完整 chunk", kind: "network", jumpTo: 4 },
  { label: "平均梯度", detail: "SUM ÷ 4", kind: "memory", jumpTo: 7 },
  { label: "AdamW", detail: "更新参数与 m、v", kind: "optimizer", jumpTo: 8 },
] as const;

const formatNumber = (value: number) => {
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  return Number(value.toFixed(3)).toLocaleString("en-US");
};

const formatVector = (values: number[]) =>
  `[${values.map(formatNumber).join(", ")}]`;

function copyFor(step: PlaygroundStep, worldSize: number): StageCopy {
  if (step.stage === "local") {
    return {
      title: "每张 GPU 先得到自己的梯度",
      summary: "四个 rank 使用不同 mini-batch，所以此刻四份梯度并不相同。",
      cause: "每个进程完成本地 forward 和 backward。",
      action: "CUDA kernel 把结果写入 parameter.grad，再由 DDP 放入 bucket。",
      result: "通信开始前，每个 rank 都拥有 C0 到 C3 的本地版本。",
      idea: "先复制模型，再切分数据，用冗余状态换取并行计算。",
    };
  }

  if (step.stage === "reduce-scatter") {
    const contributors = (step.round ?? 0) + 2;
    return {
      title: `${step.label}：部分和继续沿 Ring 传递`,
      summary: `每个 rank 同时发送一个 chunk，也从上一张 GPU 接收另一个 chunk。`,
      cause: "四个 rank 以相同顺序进入同一个 collective。",
      action: `NCCL 按 rank 和轮数选 chunk，接收后与本地同编号 chunk 相加。`,
      result: `本轮结束后，在途 chunk 已包含 ${contributors}/${worldSize} 个 rank 的贡献。`,
      idea: "确定性调度让所有参与者无需临时协商，也能各自发送不同的数据。",
    };
  }

  if (step.stage === "all-gather") {
    const known = (step.round ?? 0) + 2;
    return {
      title: `${step.label}：完整 chunk 被继续转发`,
      summary: "归约已经完成，这一阶段只复制结果，不再执行加法。",
      cause: "Reduce-Scatter 结束后，每个 rank 只持有一个完整归约 chunk。",
      action: "每个 rank 把当前拥有的完整 chunk 发给下一跳，并写入固定 offset。",
      result: `本轮结束后，每个 rank 已拥有 ${known}/${worldSize} 个完整 chunk。`,
      idea: "把多播变成分布式接力，避免单一中心节点成为瓶颈。",
    };
  }

  if (step.stage === "average") {
    return {
      title: "四张 GPU 得到同一份平均梯度",
      summary: "All-Reduce 的 SUM 已经完整，DDP 再除以 world_size。",
      cause: "All-Gather 已把 C0 到 C3 复制到所有 rank。",
      action: `每个元素除以 ${worldSize}，bucket 与 parameter.grad 的映射由框架维护。`,
      result: "所有 parameter.grad 数值一致，通信层的工作到这里结束。",
      idea: "通信层只处理指针、dtype、数量和操作，框架层负责恢复模型语义。",
    };
  }

  return {
    title: "每个 rank 独立执行同一次 AdamW 更新",
    summary: "输入梯度、参数和 optimizer state 相同，所以四份模型继续保持一致。",
    cause: "同步后的 parameter.grad 已经可供 optimizer.step() 使用。",
    action: "GPU kernel 更新一阶矩 m、二阶矩 v，并把 weight decay 与梯度更新分开。",
    result: "四份参数得到相同的新值，下一轮数据并行可以开始。",
    idea: "只同步必要状态，让相同的确定性状态转移在各副本本地重复执行。",
  };
}

function TransferExplanation({
  incoming,
  outgoing,
  step,
  rank,
}: {
  incoming?: RingTransfer;
  outgoing?: RingTransfer;
  step: PlaygroundStep;
  rank: number;
}) {
  if (!incoming || !outgoing) {
    if (step.stage === "local") {
      return (
        <div className="rank-action-empty">
          <strong>Rank {rank} 还没有通信</strong>
          <span>它现在只持有自己算出的四个本地 chunk。</span>
        </div>
      );
    }
    if (step.stage === "average") {
      return (
        <div className="rank-action-empty">
          <strong>Rank {rank} 执行逐元素除法</strong>
          <span>完整 SUM 除以 world_size=4，结果写回同一个 bucket。</span>
        </div>
      );
    }
    return (
      <div className="rank-action-empty">
        <strong>Rank {rank} 读取同步梯度</strong>
        <span>AdamW 在本地读取 parameter、grad、m 和 v，并直接修改显存。</span>
      </div>
    );
  }

  const reducing = step.stage === "reduce-scatter";
  return (
    <div className="rank-action-grid">
      <div className="rank-action route-out">
        <span>发送</span>
        <strong>R{outgoing.from} → R{outgoing.to} · C{outgoing.chunk}</strong>
        <code>{formatVector(outgoing.sent)}</code>
      </div>
      <div className="rank-action route-in">
        <span>接收</span>
        <strong>R{incoming.from} → R{incoming.to} · C{incoming.chunk}</strong>
        <code>{formatVector(incoming.sent)}</code>
      </div>
      <div className="rank-equation">
        <span>{reducing ? "本地同编号值" : "固定写入位置"}</span>
        <code>
          {reducing && incoming.before
            ? `${formatVector(incoming.before)} + ${formatVector(incoming.sent)} = ${formatVector(incoming.after)}`
            : `C${incoming.chunk} ← ${formatVector(incoming.after)}`}
        </code>
      </div>
    </div>
  );
}

export function DDPPlayground({ focusEventId }: { focusEventId?: string | null }) {
  const [inputs, setInputs] = useState(() => copyPreset(gradientPresets[0].values));
  const [presetId, setPresetId] = useState<string | null>(gradientPresets[0].id);
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedRank, setSelectedRank] = useState(0);
  const [resultMode, setResultMode] = useState<"sum" | "average">("average");

  useEffect(() => {
    if (focusEventId) setStepIndex(stepIndexForConceptEvent(focusEventId));
  }, [focusEventId]);

  const simulation = useMemo(() => simulateRingAllReduce(inputs), [inputs]);
  const step = playgroundSteps[stepIndex];
  const rankStates = ranksForStep(simulation, step);
  const round = roundForStep(simulation, step);
  const outgoing = round?.transfers.find((transfer) => transfer.from === selectedRank);
  const incoming = round?.transfers.find((transfer) => transfer.to === selectedRank);
  const stageCopy = copyFor(step, simulation.worldSize);
  const finalValues = resultMode === "sum" ? simulation.reduced : simulation.averaged;

  const setGradientValue = (index: number, rawValue: string) => {
    const nextValue = Number(rawValue);
    if (!Number.isFinite(nextValue)) return;
    setInputs((current) =>
      current.map((rank, rankIndex) =>
        rankIndex === selectedRank
          ? rank.map((value, valueIndex) =>
              valueIndex === index ? nextValue : value,
            )
          : [...rank],
      ),
    );
    setPresetId(null);
  };

  const selectPreset = (id: string) => {
    const preset = gradientPresets.find((item) => item.id === id);
    if (!preset) return;
    setInputs(copyPreset(preset.values));
    setPresetId(id);
    setStepIndex(0);
  };

  const activeFlowIndex =
    step.stage === "local"
      ? 1
      : step.stage === "reduce-scatter"
        ? 2
        : step.stage === "all-gather"
          ? 3
          : step.stage === "average"
            ? 4
            : 5;

  const optimizerGradient = simulation.averaged[0];
  const learningRate = 0.001;
  const weightDecay = 0.01;
  const normalizedGradient =
    optimizerGradient === 0 ? 0 : optimizerGradient / (Math.abs(optimizerGradient) + 1e-8);
  const nextParameter = 1 - learningRate * normalizedGradient - learningRate * weightDecay;

  return (
    <section className="ddp-playground" id="ddp-playground" aria-label="DDP interactive playground">
      <header className="playground-header">
        <div>
          <span>DDP Playground</span>
          <h2>改一个梯度，逐轮观察四张 GPU 如何达成一致</h2>
          <p>所有数值都由 Ring All-Reduce 模拟器实时计算。这里没有自动播放。</p>
        </div>
        <div className="playground-facts" aria-label="模拟配置">
          <span><strong>4</strong> ranks</span>
          <span><strong>8</strong> fp32 / rank</span>
          <span><strong>4</strong> chunks</span>
          <span><strong>6</strong> ring rounds</span>
        </div>
      </header>

      <div className="playground-body">
        <aside className="playground-controls" aria-label="梯度输入控制">
          <section className="control-section">
            <div className="control-heading">
              <SlidersHorizontal size={17} aria-hidden="true" />
              <div><strong>选择一组梯度</strong><span>也可以在下面直接改数字</span></div>
            </div>
            <div className="preset-grid">
              {gradientPresets.map((preset) => (
                <button
                  type="button"
                  className={presetId === preset.id ? "is-active" : ""}
                  aria-pressed={presetId === preset.id}
                  title={preset.description}
                  onClick={() => selectPreset(preset.id)}
                  key={preset.id}
                >
                  <strong>{preset.label}</strong>
                  <span>{preset.description}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="control-section">
            <div className="control-heading">
              <Cube size={17} aria-hidden="true" />
              <div><strong>编辑本地梯度</strong><span>当前正在编辑 Rank {selectedRank}</span></div>
            </div>
            <div className="rank-selector" aria-label="选择 rank">
              {inputs.map((_, rank) => (
                <button
                  type="button"
                  className={`rank-tone-${rank}${selectedRank === rank ? " is-active" : ""}`}
                  aria-pressed={selectedRank === rank}
                  onClick={() => setSelectedRank(rank)}
                  key={rank}
                >
                  R{rank}
                </button>
              ))}
            </div>
            <div className="gradient-input-grid">
              {inputs[selectedRank].map((value, index) => (
                <label key={index}>
                  <span>g{index}</span>
                  <input
                    type="number"
                    step="any"
                    value={value}
                    aria-label={`Rank ${selectedRank} gradient ${index}`}
                    onChange={(event) => setGradientValue(index, event.target.value)}
                  />
                </label>
              ))}
            </div>
            <button type="button" className="reset-button" onClick={() => selectPreset(gradientPresets[0].id)}>
              <ArrowCounterClockwise size={15} aria-hidden="true" />恢复默认值
            </button>
          </section>

          <section className="control-section layer-legend">
            <strong>颜色在整张图里含义固定</strong>
            <div><span className="legend-compute">GPU 计算</span><span className="legend-framework">框架</span><span className="legend-collective">集合通信</span><span className="legend-network">链路</span><span className="legend-memory">显存</span><span className="legend-optimizer">优化器</span></div>
          </section>
        </aside>

        <div className="playground-canvas">
          <nav className="phase-toolbar" aria-label="All-Reduce 阶段">
            <button
              type="button"
              className="phase-nav-button"
              disabled={stepIndex === 0}
              onClick={() => setStepIndex((index) => Math.max(0, index - 1))}
              aria-label="上一个状态"
            >
              <CaretLeft size={16} />
            </button>
            <div className="phase-steps">
              {playgroundSteps.map((item, index) => (
                <button
                  type="button"
                  className={`${item.stage}${index === stepIndex ? " is-active" : ""}${index < stepIndex ? " is-past" : ""}`}
                  aria-current={index === stepIndex ? "step" : undefined}
                  onClick={() => setStepIndex(index)}
                  key={item.id}
                >
                  <span>{item.compactLabel}</span>
                  <small>{item.label}</small>
                </button>
              ))}
            </div>
            <button
              type="button"
              className="phase-nav-button"
              disabled={stepIndex === playgroundSteps.length - 1}
              onClick={() => setStepIndex((index) => Math.min(playgroundSteps.length - 1, index + 1))}
              aria-label="下一个状态"
            >
              <CaretRight size={16} />
            </button>
          </nav>

          <section className="system-flow" aria-label="DDP 数据流">
            {flowNodes.map((node, index) => (
              <div className="flow-node-wrap" key={node.label}>
                <button
                  type="button"
                  className={`flow-node flow-${node.kind}${index === activeFlowIndex ? " is-active" : ""}${index < activeFlowIndex ? " is-done" : ""}`}
                  onClick={() => setStepIndex(node.jumpTo)}
                >
                  <strong>{node.label}</strong>
                  <span>{node.detail}</span>
                </button>
                {index < flowNodes.length - 1 && <ArrowRight size={15} className="flow-arrow" aria-hidden="true" />}
              </div>
            ))}
          </section>

          <section className="ring-visualizer" aria-label="每个 rank 的 chunk 状态">
            <header>
              <div>
                <strong>{step.label}</strong>
                <span>{round ? `四条 Ring 边并行传输，每条边发送 ${simulation.chunkSize * 4} bytes` : stageCopy.summary}</span>
              </div>
              <code>send_chunk = rank {step.stage === "all-gather" ? "+ 1 - round" : "- round"} mod 4</code>
            </header>

            <div className="ring-matrix" role="table" aria-label="Rank 与 chunk 状态矩阵">
              <div className="matrix-header" role="row">
                <span role="columnheader">设备</span>
                {Array.from({ length: 4 }, (_, chunk) => (
                  <strong className={`chunk-tone-${chunk}`} role="columnheader" key={chunk}>C{chunk}<small>g{chunk * 2}, g{chunk * 2 + 1}</small></strong>
                ))}
              </div>
              {rankStates.map((rankState) => {
                const rankOutgoing = round?.transfers.find((transfer) => transfer.from === rankState.rank);
                const rankIncoming = round?.transfers.find((transfer) => transfer.to === rankState.rank);
                return (
                  <div className={`matrix-row${selectedRank === rankState.rank ? " is-selected" : ""}`} role="row" key={rankState.rank}>
                    <button type="button" className={`matrix-rank rank-tone-${rankState.rank}`} onClick={() => setSelectedRank(rankState.rank)} role="rowheader">
                      <strong>Rank {rankState.rank}</strong>
                      <small>GPU {rankState.rank}</small>
                    </button>
                    {rankState.chunks.map((chunk) => {
                      const isOutgoing = rankOutgoing?.chunk === chunk.chunk;
                      const isIncoming = rankIncoming?.chunk === chunk.chunk;
                      return (
                        <div
                          className={`matrix-chunk chunk-tone-${chunk.chunk}${chunk.complete ? " is-complete" : ""}${!chunk.values.length ? " is-empty" : ""}${isOutgoing ? " is-sending" : ""}${isIncoming ? " is-receiving" : ""}`}
                          role="cell"
                          key={chunk.chunk}
                        >
                          <code>{chunk.values.length ? formatVector(chunk.values) : "等待传入"}</code>
                          <span>{chunk.contributors.length ? chunk.contributors.map((rank) => `R${rank}`).join(" + ") : "还没有副本"}</span>
                          {(isOutgoing || isIncoming) && <small>{isOutgoing ? "SEND" : "RECV"}</small>}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>

            <div className="selected-rank-action">
              <header>
                <div><strong>只看 Rank {selectedRank}</strong><span>同一轮里，它既是发送者也是接收者</span></div>
                <span className={`rank-badge rank-tone-${selectedRank}`}>R{selectedRank}</span>
              </header>
              <TransferExplanation incoming={incoming} outgoing={outgoing} step={step} rank={selectedRank} />
            </div>
          </section>

          <section className="result-strip" aria-label="最终同步结果">
            <header>
              <div><strong>完整结果</strong><span>任何输入修改都会立即重新计算</span></div>
              <div className="result-toggle" role="group" aria-label="选择查看求和或平均结果">
                <button type="button" className={resultMode === "sum" ? "is-active" : ""} onClick={() => setResultMode("sum")}>SUM</button>
                <button type="button" className={resultMode === "average" ? "is-active" : ""} onClick={() => setResultMode("average")}>AVG</button>
              </div>
            </header>
            <div className="result-values">
              {finalValues.map((value, index) => <div key={index}><span>g{index}</span><strong>{formatNumber(value)}</strong></div>)}
            </div>
          </section>
        </div>

        <aside className="playground-inspector" aria-label="当前状态解释">
          <section className={`inspector-lead stage-${step.stage}`}>
            <span>{stepIndex + 1} / {playgroundSteps.length}</span>
            <h3>{stageCopy.title}</h3>
            <p>{stageCopy.summary}</p>
          </section>

          <section className="causal-chain">
            <div><span>为什么开始</span><p>{stageCopy.cause}</p></div>
            <div><span>系统做什么</span><p>{stageCopy.action}</p></div>
            <div><span>状态变成什么</span><p>{stageCopy.result}</p></div>
          </section>

          <section className="layer-stack">
            <strong>同一个动作穿过哪些层</strong>
            <div className={step.stage === "local" ? "is-active" : ""}><Cpu size={16} /><span><b>CPU 进程</b><small>调用 backward 或 optimizer.step()</small></span></div>
            <div className={step.stage === "local" || step.stage === "optimizer" ? "is-active" : ""}><Lightning size={16} /><span><b>CUDA Stream</b><small>按依赖启动 GPU kernel</small></span></div>
            <div className={round ? "is-active" : ""}><Stack size={16} /><span><b>NCCL kernel</b><small>读取指针指向的 tensor 字节</small></span></div>
            <div className={round ? "is-active" : ""}><Network size={16} /><span><b>NVLink / PCIe</b><small>在 GPU 之间搬运 chunk</small></span></div>
            <div className="is-active"><Database size={16} /><span><b>GPU HBM</b><small>保存 grad、bucket、m 和 v</small></span></div>
          </section>

          <section className="system-idea">
            <span>这一步体现的系统思想</span>
            <strong>{stageCopy.idea}</strong>
          </section>

          {step.stage === "optimizer" && (
            <section className="adam-example">
              <strong>用 g0 演示第一次 AdamW 更新</strong>
              <dl>
                <div><dt>θ</dt><dd>1.000</dd></div>
                <div><dt>平均梯度 g</dt><dd>{formatNumber(optimizerGradient)}</dd></div>
                <div><dt>learning rate</dt><dd>{learningRate}</dd></div>
                <div><dt>weight decay</dt><dd>{weightDecay}</dd></div>
                <div><dt>更新后 θ</dt><dd>{formatNumber(nextParameter)}</dd></div>
              </dl>
              <p>梯度更新与 weight decay 分开计算，这就是 AdamW 的关键。</p>
            </section>
          )}
        </aside>
      </div>
    </section>
  );
}
