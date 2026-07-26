import { ArrowDown, ArrowRight } from "@phosphor-icons/react";
import { useMemo, useState } from "react";
import type { SimulationEvent } from "../../domain/simulation";
import { ddpGradientExample, type RankState, type RingRound } from "../../sim/ring";

function frameFor(event: SimulationEvent): { title: string; round: RingRound | null; ranks: RankState[] } {
  const rs = event.id.match(/^reduce-scatter-(\d)$/);
  if (rs) {
    const round = ddpGradientExample.reduceScatter[Number(rs[1])];
    return { title: `Reduce-Scatter · Round ${round.round + 1}/3`, round, ranks: round.ranks };
  }
  const ag = event.id.match(/^all-gather-(\d)$/);
  if (ag) {
    const round = ddpGradientExample.allGather[Number(ag[1])];
    return { title: `All-Gather · Round ${round.round + 1}/3`, round, ranks: round.ranks };
  }
  if (["bucket-writeback", "optimizer-update", "iteration-complete"].includes(event.id)) {
    const round = ddpGradientExample.allGather[2];
    return { title: "All-Reduce 完成 · 每个 rank 都有完整结果", round, ranks: round.ranks };
  }
  return { title: "输入 · 每个 rank 的本地梯度", round: null, ranks: ddpGradientExample.initial };
}

const format = (values: readonly number[]) => values.length ? `[${values.join(", ")}]` : "待接收";

export function RingCollectiveView({ event }: { event: SimulationEvent }) {
  const [selectedRank, setSelectedRank] = useState(0);
  const frame = useMemo(() => frameFor(event), [event]);
  const incoming = frame.round?.transfers.find((transfer) => transfer.to === selectedRank);
  const outgoing = frame.round?.transfers.find((transfer) => transfer.from === selectedRank);

  return (
    <div className="ring-view detail-view">
      <div className="view-intro-row">
        <div><span className="view-kicker">4 ranks · 8 个 fp32 · 每 chunk 2 个数</span><h3>{frame.title}</h3></div>
        <div className="ring-formula"><code>RS send_chunk = (rank − round) mod 4</code><span>方向固定：R0 → R1 → R2 → R3 → R0</span></div>
      </div>

      <div className="tensor-source">
        <span>具体输入</span>
        {ddpGradientExample.inputs.map((values, rank) => <code key={rank}>R{rank} {format(values)}</code>)}
      </div>

      <section className="ring-state-table" aria-label="Per-rank chunk state">
        <header>
          <span>Rank / 本地 buffer</span>
          {[0, 1, 2, 3].map((chunk) => <strong key={chunk}>Chunk C{chunk}<small>offset {chunk * 2}:{chunk * 2 + 2}</small></strong>)}
        </header>
        {frame.ranks.map((rank) => (
          <button
            type="button"
            className={`rank-buffer${rank.rank === selectedRank ? " is-selected" : ""}`}
            key={rank.rank}
            onClick={() => setSelectedRank(rank.rank)}
          >
            <span className="rank-label"><strong>Rank {rank.rank}</strong><small>GPU {rank.rank} HBM</small></span>
            {rank.chunks.map((chunk) => (
              <span className={`chunk-cell${chunk.complete ? " is-complete" : ""}${chunk.values.length ? "" : " is-empty"}`} key={chunk.chunk}>
                <code>{format(chunk.values)}</code>
                <small>{chunk.complete ? "完整 4/4" : chunk.contributors.length ? `贡献 ${chunk.contributors.length}/4 · R${chunk.contributors.join("+R")}` : "尚未收到"}</small>
              </span>
            ))}
          </button>
        ))}
      </section>

      {frame.round ? (
        <div className="round-detail">
          <section className="selected-transfer">
            <header><span>放大 Rank {selectedRank} 本轮动作</span><strong>{frame.round.phase === "reduce-scatter" ? "SEND + RECV + SUM" : "SEND + RECV + COPY"}</strong></header>
            <div className="duplex-row">
              <div><small>发送到 Rank {outgoing?.to}</small><strong>C{outgoing?.chunk} {format(outgoing?.sent ?? [])}</strong><code>读取本地显存 → 链路字节流</code></div>
              <ArrowRight size={18} />
              <div><small>从 Rank {incoming?.from} 接收</small><strong>C{incoming?.chunk} {format(incoming?.sent ?? [])}</strong><code>链路字节流 → GPU recv buffer</code></div>
            </div>
            <ArrowDown size={17} />
            {frame.round.phase === "reduce-scatter" ? (
              <div className="reduction-equation">
                <code>{format(incoming?.before ?? [])}</code><span>本地值</span><b>+</b><code>{format(incoming?.sent ?? [])}</code><span>收到的 partial</span><b>=</b><code>{format(incoming?.after ?? [])}</code><span>写回 HBM</span>
              </div>
            ) : (
              <div className="reduction-equation copy-equation"><code>C{incoming?.chunk} {format(incoming?.sent ?? [])}</code><b>→</b><span>直接复制到固定 chunk offset，不再归约</span></div>
            )}
          </section>

          <section className="all-transfers">
            <header><span>同一轮四条边并发发生</span><small>每个 rank 的规则相同，代入 rank 后得到不同 chunk</small></header>
            <div>{frame.round.transfers.map((transfer) => <code key={transfer.from}>R{transfer.from} ─ C{transfer.chunk} {format(transfer.sent)} → R{transfer.to}</code>)}</div>
          </section>
        </div>
      ) : (
        <div className="collective-primer">
          <div><small>目标操作</small><strong>逐元素 SUM，再 ÷ 4</strong><code>AllReduce(grad, op=SUM)</code></div>
          <ArrowRight size={18} />
          <div><small>最终 SUM</small><strong>{format(ddpGradientExample.reduced)}</strong><code>每个位置都加过 R0…R3</code></div>
          <ArrowRight size={18} />
          <div><small>DDP 平均梯度</small><strong>{format(ddpGradientExample.averaged)}</strong><code>每个 rank 最终完全相同</code></div>
        </div>
      )}

      <div className="layer-responsibility">
        <div><span>PyTorch / DDP</span><strong>知道语义和格式</strong><small>参数 → bucket offset、shape、dtype、world_size</small></div>
        <div><span>NCCL kernel</span><strong>知道通信操作</strong><small>device pointer、count、dtype、SUM、rank topology</small></div>
        <div><span>NVLink / PCIe</span><strong>只搬运字节</strong><small>packet/flit、路由、流控；不知道“梯度”</small></div>
      </div>
    </div>
  );
}
