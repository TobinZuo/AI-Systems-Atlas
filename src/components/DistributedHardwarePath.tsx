import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import type { DistributedHardwareSnapshot } from "../playground/distributedHardware";

export function DistributedHardwarePath({
  snapshot,
  title = "这一步如何落到硬件",
}: {
  snapshot: DistributedHardwareSnapshot;
  title?: string;
}) {
  const communicationActive = snapshot.activeStream === "comm" || snapshot.activeStream === "compute-then-comm";
  const computeActive = snapshot.activeStream === "compute" || snapshot.activeStream === "compute-then-comm";

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
