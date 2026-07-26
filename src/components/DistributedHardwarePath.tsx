import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { ArrowsLeftRight } from "@phosphor-icons/react/ArrowsLeftRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import {
  distributedHardwareDataFlow,
  distributedHardwareRankViews,
  type HardwareMemoryState,
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

const memoryStateLabels: Record<HardwareMemoryState, string> = {
  resident: "驻留",
  read: "读取",
  write: "写回",
  send: "发送",
  receive: "接收",
  stale: "等待覆盖",
  absent: "不存在",
  release: "释放",
};

export function DistributedHardwarePath({
  id,
  snapshot,
  title = "这一步如何落到硬件",
  worldSize = 4,
  selectedRank = 0,
  onSelectRank,
}: {
  id?: string;
  snapshot: DistributedHardwareSnapshot;
  title?: string;
  worldSize?: number;
  selectedRank?: number;
  onSelectRank?: (rank: number) => void;
}) {
  const communicationActive = snapshot.activeStream === "comm" || snapshot.activeStream === "compute-then-comm";
  const rankViews = distributedHardwareRankViews(snapshot, worldSize, selectedRank);
  const selectedRankView = rankViews.find((rank) => rank.rank === selectedRank) ?? rankViews[0];
  const selectedCommunicationActive = selectedRankView.stream === "comm" || selectedRankView.stream === "compute-then-comm";
  const selectedComputeActive = selectedRankView.stream === "compute" || selectedRankView.stream === "compute-then-comm";
  const selectedOperationActive = selectedRankView.stream !== "none";
  const selectedKernel = snapshot.pattern === "owner-compute" && selectedRankView.role !== "owner"
    ? "这个 Rank 不更新所选参数"
    : snapshot.kernel;
  const dataFlow = distributedHardwareDataFlow(snapshot, worldSize);

  const renderRanks = (ranks: number[]) => (
    <div className="hardware-endpoint-ranks">
      {ranks.length > 0
        ? ranks.map((rank) => <i className={rank === selectedRank ? "is-selected" : ""} key={rank}>R{rank}</i>)
        : <i>无远端接收者</i>}
    </div>
  );

  return (
    <section className={`distributed-hardware-path stream-${snapshot.activeStream}`} id={id} aria-label={title}>
      <header>
        <div>
          <span>Hardware execution stage</span>
          <h3>{title}</h3>
          <p>{snapshot.operation}</p>
        </div>
        <code>{snapshot.payload}</code>
      </header>

      <section className={`hardware-data-journey${dataFlow.communication ? " is-communication" : " is-local"}${snapshot.pattern === "idle" ? " is-idle" : ""}${snapshot.pattern === "release" ? " is-release" : ""}`} aria-label="真实 Tensor 数据路径">
        <header>
          <div><span>真实 Tensor 数据路径</span><strong>从源 HBM 到目标 HBM</strong></div>
          <small>{snapshot.pattern === "idle" ? "当前没有数据搬运。页面显示各 Rank 此刻真实驻留的状态，以及下一次操作的输入。" : "CPU 提交的是操作与显存地址，GPU kernel 真正读取、搬运并写回地址指向的 Tensor 字节。"}</small>
        </header>
        <div className="hardware-data-journey-scroll">
          <div className="hardware-data-flow-track" key={`${snapshot.operation}-${snapshot.payload}`}>
            <article className="hardware-data-endpoint endpoint-source">
              <span><Database size={16} weight="duotone" aria-hidden="true" />源 HBM</span>
              {renderRanks(dataFlow.sourceRanks)}
              <strong>{dataFlow.sourceObject}</strong>
              <small>{snapshot.pattern === "idle" ? "当前长期驻留的显存对象" : "kernel 从这些显存对象读取输入"}</small>
            </article>

            <div className="hardware-data-arrow" aria-hidden="true"><ArrowRight size={17} /><i /></div>

            <article className="hardware-data-operator">
              <span><Stack size={16} weight="duotone" aria-hidden="true" />{dataFlow.communication ? "集合通信执行" : "本地 GPU 执行"}</span>
              <strong>{dataFlow.operator}</strong>
              <small>{dataFlow.operatorDetail}</small>
              <code>{dataFlow.transport}</code>
            </article>

            <div className="hardware-data-arrow" aria-hidden="true"><ArrowRight size={17} /><i /></div>

            <article className="hardware-data-endpoint endpoint-destination">
              <span><Database size={16} weight="duotone" aria-hidden="true" />目标 HBM</span>
              {renderRanks(dataFlow.destinationRanks)}
              <strong>{dataFlow.destinationObject}</strong>
              <small>{snapshot.pattern === "idle" ? "下一次操作会把状态推进到这里" : "结果直接写入目标显存 buffer"}</small>
            </article>
          </div>
        </div>
      </section>

      <section className={`distributed-rank-stage pattern-${snapshot.pattern}`} aria-label="多 Rank 与 GPU 的全局执行现场">
        <header>
          <div><span>全局 Rank / GPU 现场</span><strong>{patternLabels[snapshot.pattern]}</strong></div>
          <small>逻辑 collective 视图。真实 NCCL 会按拓扑选择 Ring、Tree 或其他实现。</small>
        </header>

        <div className="distributed-rank-stage-scroll">
          <div className="distributed-rank-grid" style={{ gridTemplateColumns: `repeat(${worldSize}, minmax(220px, 1fr))` }}>
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
                  <div className="distributed-rank-memory">
                    <div className="distributed-rank-memory-title"><Database size={14} weight="duotone" aria-hidden="true" /><b>GPU HBM</b></div>
                    <div className="distributed-rank-memory-slots">
                      {rankView.memorySlots.map((slot, index) => (
                        <div className={`memory-state-${slot.state}`} key={`${slot.label}-${index}`}>
                          <span><strong>{slot.label}</strong><small>{memoryStateLabels[slot.state]}</small></span>
                          <code>{slot.detail}</code>
                        </div>
                      ))}
                    </div>
                  </div>
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
        <span>选中 Rank 的内部剖面</span>
        <strong>Rank {selectedRank}：{selectedRankView.roleLabel}</strong>
      </div>

      <div className="hardware-path-scroll">
        <div className="hardware-path-track">
          <article className="hardware-path-node hardware-path-cpu is-active">
            <span><Cpu size={17} weight="duotone" aria-hidden="true" />CPU 进程</span>
            <strong>{snapshot.cpuCall}</strong>
            <small>{snapshot.cpuDetail}</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-stream${selectedOperationActive ? " is-active" : ""}`}>
            <span><Lightning size={17} weight="duotone" aria-hidden="true" />CUDA Streams</span>
            <div className="hardware-stream-pair">
              <i className={selectedComputeActive ? "is-active" : ""}>Compute stream</i>
              <i className={selectedCommunicationActive ? "is-active" : ""}>Comm stream</i>
            </div>
            <small>同一 GPU 上的两条任务队列</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-kernel${selectedOperationActive ? " is-active" : ""}`}>
            <span><Stack size={17} weight="duotone" aria-hidden="true" />GPU SM</span>
            <strong>{selectedKernel}</strong>
            <small>{selectedOperationActive ? snapshot.kernelDetail : selectedRankView.roleLabel}</small>
          </article>
          <ArrowRight className="hardware-path-arrow" size={16} aria-hidden="true" />

          <article className="hardware-path-node hardware-path-hbm is-active">
            <span><Database size={17} weight="duotone" aria-hidden="true" />GPU HBM</span>
            <strong>{selectedRankView.memoryLabel}</strong>
            <small>{snapshot.hbmDetail}</small>
          </article>
          <ArrowRight className={`hardware-path-arrow${selectedCommunicationActive ? " is-active" : ""}`} size={16} aria-hidden="true" />

          <article className={`hardware-path-node hardware-path-link${selectedCommunicationActive ? " is-active" : ""}`}>
            <span><Network size={17} weight="duotone" aria-hidden="true" />GPU 互连</span>
            <strong>{snapshot.link}</strong>
            <small>{selectedCommunicationActive ? "单机 collective 的字节传输路径" : "这个 Rank 在此刻不经过 GPU 互连"}</small>
            {selectedCommunicationActive && <i className="hardware-data-packet" aria-hidden="true" />}
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
