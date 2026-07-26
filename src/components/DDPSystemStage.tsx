import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { useEffect, useState } from "react";
import {
  chunkJourney,
  type PlaygroundStage,
  type PlaygroundStep,
} from "../playground/ddp";
import type { RankState, RingRound, RingSimulation } from "../sim/ring";

interface DDPSystemStageProps {
  simulation: RingSimulation;
  step: PlaygroundStep;
  rankStates: RankState[];
  rankStatesBefore: RankState[];
  round: RingRound | null;
  selectedRank: number;
  selectedChunk: number | null;
  replayKey: number;
  onSelectRank: (rank: number) => void;
  onSelectChunk: (chunk: number | null) => void;
  onSelectStep: (step: number) => void;
  onReplay: () => void;
}

const formatNumber = (value: number) =>
  Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));

const formatVector = (values: number[]) =>
  values.length ? `[${values.map(formatNumber).join(", ")}]` : "空";

const kernelForStage: Record<PlaygroundStage, string> = {
  local: "backward kernel",
  "reduce-scatter": "ncclReduceKernel",
  "all-gather": "ncclCopyKernel",
  average: "divide kernel",
  optimizer: "AdamW kernel",
};

const cpuCallForStage: Record<PlaygroundStage, string> = {
  local: "loss.backward()",
  "reduce-scatter": "DDP hook -> NCCL",
  "all-gather": "NCCL collective",
  average: "DDP writeback",
  optimizer: "optimizer.step()",
};

function deviceStatus(
  stage: PlaygroundStage,
  outgoingChunk?: number,
  incomingChunk?: number,
) {
  if (stage === "local") return "SM 正在写 parameter.grad";
  if (stage === "average") return "bucket 中每个元素除以 4";
  if (stage === "optimizer") return "读取 grad、m、v，写回 parameter";
  const verb = stage === "reduce-scatter" ? "接收后求和" : "接收后复制";
  return `发 C${outgoingChunk}，收 C${incomingChunk}，${verb}`;
}

export function DDPSystemStage({
  simulation,
  step,
  rankStates,
  rankStatesBefore,
  round,
  selectedRank,
  selectedChunk,
  replayKey,
  onSelectRank,
  onSelectChunk,
  onSelectStep,
  onReplay,
}: DDPSystemStageProps) {
  const animationId = `${step.id}-${replayKey}`;
  const [settledAnimation, setSettledAnimation] = useState<string | null>(null);
  const transferSettled = !round || settledAnimation === animationId;

  useEffect(() => {
    if (!round || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setSettledAnimation(animationId);
      return;
    }

    setSettledAnimation(null);
    const timer = window.setTimeout(() => setSettledAnimation(animationId), 1020);
    return () => window.clearTimeout(timer);
  }, [animationId, round]);

  const visibleRankStates = transferSettled ? rankStates : rankStatesBefore;
  const incoming = round?.transfers.find((transfer) => transfer.to === selectedRank);
  const outgoing = round?.transfers.find((transfer) => transfer.from === selectedRank);
  const focusedTransfer = selectedChunk === null
    ? incoming
    : round?.transfers.find((transfer) => transfer.chunk === selectedChunk);
  const focusChunk = selectedChunk ?? incoming?.chunk ?? outgoing?.chunk ?? 0;
  const focusedState = visibleRankStates[selectedRank].chunks[focusChunk];
  const journey = selectedChunk === null ? [] : chunkJourney(simulation, selectedChunk);
  const journeyIndex = step.stage === "reduce-scatter"
    ? step.round ?? 0
    : step.stage === "all-gather"
      ? 3 + (step.round ?? 0)
      : step.stage === "local"
        ? -1
        : 6;
  const computeActive = step.stage === "local" || step.stage === "average" || step.stage === "optimizer";
  const communicationActive = Boolean(round);

  const focusMessage = (() => {
    if (!focusedTransfer) {
      if (step.stage === "local") return `R${selectedRank} 的 SM 把本地梯度写入 HBM`;
      if (step.stage === "average") return `R${selectedRank} 把完整 SUM 除以 ${simulation.worldSize}`;
      if (step.stage === "optimizer") return `R${selectedRank} 用相同梯度执行 AdamW`;
      return "这一阶段没有 Ring 传输";
    }
    if (focusedTransfer.from === selectedRank) {
      return transferSettled
        ? `R${selectedRank} 已把 C${focusedTransfer.chunk} 送到下一张 GPU`
        : `R${selectedRank} 正从 HBM 读取 C${focusedTransfer.chunk} 并发送`;
    }
    if (focusedTransfer.to === selectedRank) {
      const operation = step.stage === "reduce-scatter" ? "与本地值相加" : "写入固定位置";
      return transferSettled
        ? `R${selectedRank} 已接收 C${focusedTransfer.chunk}，${operation}`
        : `C${focusedTransfer.chunk} 正沿 NVLink 前往 R${selectedRank}`;
    }
    return `本轮 C${focusedTransfer.chunk} 从 R${focusedTransfer.from} 传到 R${focusedTransfer.to}`;
  })();

  return (
    <section className={`ddp-system-stage stage-${step.stage}`} id="ddp-system-stage" aria-label="GPU 与 Ring 数据流舞台">
      <header className="system-stage-header">
        <div>
          <strong>数据现在在哪，谁正在搬它</strong>
          <span>舞台演示搬运过程，下方矩阵显示本轮完成后的精确数值。点击 GPU 或 chunk 可聚焦。</span>
        </div>
        <div className="stage-tools">
          <div className="chunk-follow-control" role="group" aria-label="选择追踪的 chunk">
            <span>追踪</span>
            <button
              type="button"
              className={selectedChunk === null ? "is-active" : ""}
              aria-pressed={selectedChunk === null}
              onClick={() => onSelectChunk(null)}
            >
              全部
            </button>
            {Array.from({ length: simulation.worldSize }, (_, chunk) => (
              <button
                type="button"
                className={`chunk-tone-${chunk}${selectedChunk === chunk ? " is-active" : ""}`}
                aria-pressed={selectedChunk === chunk}
                onClick={() => onSelectChunk(chunk)}
                key={chunk}
              >
                C{chunk}
              </button>
            ))}
          </div>
          <button type="button" className="replay-transfer" onClick={onReplay} disabled={!round}>
            <ArrowCounterClockwise size={14} aria-hidden="true" />
            重放本轮
          </button>
        </div>
      </header>

      <div className="system-stage-scroll">
        <div className="hardware-stage-map">
          {Array.from({ length: simulation.worldSize }, (_, from) => {
            const transfer = round?.transfers.find((item) => item.from === from);
            return (
              <div className={`ring-edge edge-${from}${round ? " is-active" : ""}`} aria-hidden="true" key={from}>
                <span className="edge-name">NVLink</span>
                <ArrowRight size={14} weight="bold" />
                {transfer && (
                  <div className={`transfer-runner runner-${from}`} key={`${step.id}-${replayKey}-${from}`}>
                    <div
                      className={`transfer-payload chunk-tone-${transfer.chunk}${selectedChunk !== null && selectedChunk !== transfer.chunk ? " is-muted" : ""}`}
                      title={`R${transfer.from} 发送 ${formatVector(transfer.sent)} 到 R${transfer.to}`}
                    >
                      <b>C{transfer.chunk}</b>
                      <code>{formatVector(transfer.sent)}</code>
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {visibleRankStates.map((rankState) => {
            const rankOutgoing = round?.transfers.find((transfer) => transfer.from === rankState.rank);
            const rankIncoming = round?.transfers.find((transfer) => transfer.to === rankState.rank);
            return (
              <article
                className={`hardware-node node-${rankState.rank} rank-tone-${rankState.rank}${selectedRank === rankState.rank ? " is-selected" : ""}`}
                key={rankState.rank}
              >
                <button
                  type="button"
                  className="hardware-node-title"
                  aria-pressed={selectedRank === rankState.rank}
                  onClick={() => onSelectRank(rankState.rank)}
                >
                  <span><Cpu size={14} aria-hidden="true" /> Rank {rankState.rank} 进程</span>
                  <strong>GPU {rankState.rank}</strong>
                </button>
                <div className="node-memory-label">
                  <Database size={13} aria-hidden="true" /> HBM bucket
                </div>
                <div className="node-chunk-grid">
                  {rankState.chunks.map((chunk) => {
                    const sending = rankOutgoing?.chunk === chunk.chunk;
                    const receiving = rankIncoming?.chunk === chunk.chunk;
                    const followed = selectedChunk === chunk.chunk;
                    return (
                      <button
                        type="button"
                        className={`node-chunk chunk-tone-${chunk.chunk}${chunk.complete ? " is-complete" : ""}${!chunk.values.length ? " is-empty" : ""}${sending ? " is-sending" : ""}${receiving ? " is-receiving" : ""}${followed ? " is-followed" : ""}${selectedChunk !== null && !followed ? " is-dimmed" : ""}`}
                        aria-label={`在 Rank ${rankState.rank} 追踪 chunk ${chunk.chunk}，当前值 ${formatVector(chunk.values)}`}
                        aria-pressed={followed}
                        title={`贡献者：${chunk.contributors.map((rank) => `R${rank}`).join(" + ") || "无"}`}
                        onClick={() => {
                          onSelectRank(rankState.rank);
                          onSelectChunk(chunk.chunk);
                        }}
                        key={chunk.chunk}
                      >
                        <b>C{chunk.chunk}</b>
                        <code>{formatVector(chunk.values)}</code>
                        <small>{chunk.contributors.length}/{simulation.worldSize}</small>
                      </button>
                    );
                  })}
                </div>
                <footer>{deviceStatus(step.stage, rankOutgoing?.chunk, rankIncoming?.chunk)}</footer>
              </article>
            );
          })}

          <div className={`hardware-cutaway rank-tone-${selectedRank}`} aria-live="polite">
            <header>
              <span className={`rank-tone-${selectedRank}`}>只看 Rank {selectedRank}</span>
              <strong>一次操作穿过四层</strong>
            </header>
            <div className={`cutaway-layer layer-cpu ${step.stage === "local" || step.stage === "optimizer" ? "is-active" : ""}`}>
              <Cpu size={15} aria-hidden="true" />
              <span><b>CPU 进程</b><code>{cpuCallForStage[step.stage]}</code></span>
            </div>
            <ArrowRight className="cutaway-arrow" size={13} aria-hidden="true" />
            <div className="cuda-streams">
              <div className={computeActive ? "is-active" : ""}><Lightning size={13} /><span>Compute stream</span></div>
              <div className={communicationActive ? "is-active" : ""}><Network size={13} /><span>Comm stream</span></div>
            </div>
            <ArrowRight className="cutaway-arrow" size={13} aria-hidden="true" />
            <div className={`cutaway-layer layer-kernel ${computeActive || communicationActive ? "is-active" : ""}`}>
              <Stack size={15} aria-hidden="true" />
              <span><b>GPU SM</b><code>{kernelForStage[step.stage]}</code></span>
            </div>
            <ArrowRight className="cutaway-arrow" size={13} aria-hidden="true" />
            <div className="cutaway-layer layer-hbm is-active">
              <Database size={15} aria-hidden="true" />
              <span><b>HBM · bucket[C{focusChunk}]</b><code>{formatVector(focusedState.values)}</code></span>
            </div>
            <p>{focusMessage}</p>
            <a className="cutaway-deep-link" href="#/gpu/architecture">
              展开 SM、Warp 与 Lane
              <ArrowRight size={12} aria-hidden="true" />
            </a>
          </div>
        </div>
      </div>

      <div className="chunk-journey" aria-label="所选 chunk 的完整路线">
        {selectedChunk === null ? (
          <p>选择一个 chunk 后，这里会固定显示它在 6 轮通信中的完整路线。</p>
        ) : (
          <>
            <div className={`journey-origin chunk-tone-${selectedChunk}`}>
              <strong>C{selectedChunk}</strong>
              <span>同一份数据身份</span>
            </div>
            <div className="journey-hops">
              {journey.map((hop, index) => (
                <button
                  type="button"
                  className={`${hop.phase}${index === journeyIndex ? " is-active" : ""}${index < journeyIndex ? " is-past" : ""}`}
                  aria-current={index === journeyIndex ? "step" : undefined}
                  onClick={() => onSelectStep(index + 1)}
                  key={`${hop.phase}-${hop.round}`}
                >
                  <small>{hop.phase === "reduce-scatter" ? "归约求和" : "完整复制"}</small>
                  <strong>R{hop.from} <ArrowRight size={12} aria-hidden="true" /> R{hop.to}</strong>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
