import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { ArrowsLeftRight } from "@phosphor-icons/react/ArrowsLeftRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import {
  distributedHardwareRankViews,
  type DistributedHardwareSnapshot,
} from "../playground/distributedHardware";

const patternLabels: Record<DistributedHardwareSnapshot["pattern"], string> = {
  idle: "各 GPU 保持自己的长期状态",
  "local-all": "所有 GPU 并行执行本地计算",
  "owner-compute": "只有参数 owner 更新这份状态",
  broadcast: "Owner 向其余 GPU 广播新参数",
  "all-gather": "每张 GPU 贡献 shard，并重建完整权重",
  "reduce-scatter": "所有 GPU 归约梯度，每张卡只留下目标 shard",
  "all-reduce": "所有 GPU 交换并归约完整梯度",
  "compute-then-broadcast": "先本地更新 owner 状态，再交换新参数",
  release: "每张 GPU 释放临时完整 buffer",
};

export function DistributedHardwarePath({
  snapshot,
  title = "这一步如何落到硬件",
  worldSize = 4,
  selectedRank = 0,
  onSelectRank,
}: {
  snapshot: DistributedHardwareSnapshot;
  title?: string;
  worldSize?: number;
  selectedRank?: number;
  onSelectRank?: (rank: number) => void;
}) {
  const communicationActive = snapshot.activeStream === "comm" || snapshot.activeStream === "compute-then-comm";
  const computeActive = snapshot.activeStream === "compute" || snapshot.activeStream === "compute-then-comm";
  const rankViews = distributedHardwareRankViews(snapshot, worldSize, selectedRank);

  return (
    <section className={`distributed-hardware-path stream-${snapshot.activeStream}`} aria-label={title}>
      <header>
        <div>
          <span>Operation drill-down</span>
          <h3>{title}</h3>
          <p>{snapshot.operation}</p>
        </div>
        <code>{snapshot.payload}</code>
      </header>

      <section className={`distributed-rank-stage pattern-${snapshot.pattern}`} aria-label="多 Rank 与 GPU 的全局执行现场">
        <header>
          <div><span>Global Rank / GPU scene</span><strong>{patternLabels[snapshot.pattern]}</strong></div>
          <small>逻辑 collective 视图。真实 NCCL 会按拓扑选择 Ring、Tree 或其他实现。</small>
        </header>

        <div className="distributed-rank-stage-scroll">
          <div className="distributed-rank-grid" style={{ gridTemplateColumns: `repeat(${worldSize}, minmax(190px, 1fr))` }}>
            {rankViews.map((rankView) => {
              const rankBody = (
                <>
                  <span><Cpu size={14} weight="duotone" aria-hidden="true" />Rank {rankView.rank} 进程</span>
                  <strong>GPU {rankView.rank}</strong>
                </>
              );
              return (
                <article className={`distributed-rank-node role-${rankView.role}${rankView.selected ? " is-selected" : ""}`} key={rankView.rank}>
                  {onSelectRank ? (
                    <button type="button" aria-pressed={rankView.selected} onClick={() => onSelectRank(rankView.rank)}>{rankBody}</button>
                  ) : <div className="distributed-rank-node-title">{rankBody}</div>}
                  <div className="rank-stream-pair" aria-label={`Rank ${rankView.rank} 当前 stream`}>
                    <i className={rankView.stream === "compute" || rankView.stream === "compute-then-comm" ? "is-active" : ""}>Compute</i>
                    <i className={rankView.stream === "comm" || rankView.stream === "compute-then-comm" ? "is-active" : ""}>Comm</i>
                  </div>
                  <div className="distributed-rank-memory"><Database size={14} weight="duotone" aria-hidden="true" /><span><b>HBM</b><code>{rankView.memoryLabel}</code></span></div>
                  <footer>{rankView.roleLabel}</footer>
                </article>
              );
            })}
          </div>

          <div className={`distributed-fabric${communicationActive ? " is-active" : ""}`}>
            <Network size={16} weight="duotone" aria-hidden="true" />
            <strong>{communicationActive ? snapshot.link : "本阶段不经过 GPU 互连"}</strong>
            <span>{snapshot.payload}</span>
            {communicationActive && <ArrowsLeftRight size={17} weight="bold" aria-hidden="true" />}
          </div>
        </div>
      </section>

      <div className="hardware-cutaway-label">
        <span>Selected Rank cutaway</span>
        <strong>Rank {selectedRank} 内部调用链</strong>
      </div>

      <div className="hardware-path-scroll">
        <div className="hardware-path-track">
          <article className="hardware-path-node hardware-path-cpu is-active">
            <span><Cpu size={17} weight="duotone" aria-hidden="true" />CPU 进程</span>
            <strong>{snapshot.cpuCall}</strong>
            <small>{snapshot.cpuDetail}</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-stream${snapshot.activeStream !== "none" ? " is-active" : ""}`}>
            <span><Lightning size={17} weight="duotone" aria-hidden="true" />CUDA Streams</span>
            <div className="hardware-stream-pair">
              <i className={computeActive ? "is-active" : ""}>Compute stream</i>
              <i className={communicationActive ? "is-active" : ""}>Comm stream</i>
            </div>
            <small>同一 GPU 上的两条任务队列</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-kernel${snapshot.activeStream !== "none" ? " is-active" : ""}`}>
            <span><Stack size={17} weight="duotone" aria-hidden="true" />GPU SM</span>
            <strong>{snapshot.kernel}</strong>
            <small>{snapshot.kernelDetail}</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className="hardware-path-node hardware-path-hbm is-active">
            <span><Database size={17} weight="duotone" aria-hidden="true" />GPU HBM</span>
            <strong>{snapshot.hbmObject}</strong>
            <small>{snapshot.hbmDetail}</small>
          </article>
          <ArrowRight className={`hardware-path-arrow${communicationActive ? " is-active" : ""}`} size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-link${communicationActive ? " is-active" : ""}`}>
            <span><Network size={17} weight="duotone" aria-hidden="true" />GPU 互连</span>
            <strong>{snapshot.link}</strong>
            <small>{communicationActive ? "单机 collective 的字节传输路径" : "这一步只访问本卡显存"}</small>
            {communicationActive && <i className="hardware-data-packet" aria-hidden="true" />}
          </article>
        </div>
      </div>

      <footer>
        <strong>此刻发生了什么</strong>
        <p>{snapshot.explanation}</p>
        <div className="hardware-path-footer-note">
          <span>跨服务器时，最后一段通常会换成 NIC + InfiniBand/RoCE + GPUDirect RDMA。</span>
          <a href="#/gpu/architecture">下钻 GPU、SM 与 Warp <ArrowRight size={13} aria-hidden="true" /></a>
          <a href="#/gpu/cuda-stream">下钻 Compute / Comm Stream <ArrowRight size={13} aria-hidden="true" /></a>
        </div>
      </footer>
    </section>
  );
}
