import { ArrowRight, Database, GridFour, Lightning } from "@phosphor-icons/react";
import type { SimulationEvent } from "../../domain/simulation";

export function GpuExecutionView({ event }: { event: SimulationEvent }) {
  const kernelLive = event.id === "gradient-kernel";
  const writeLive = event.id === "gradient-writeback";

  return (
    <div className="gpu-view detail-view">
      <div className="view-intro-row">
        <div>
          <span className="view-kicker">GPU 0 · 数据面</span>
          <h3>一个 dW tile 如何落到 SM、Warp 和显存</h3>
        </div>
        <div className="formula-chip"><code>dW = Xᵀ × dY</code><span>示例输出：8 个 fp32</span></div>
      </div>

      <div className="gpu-cutaway">
        <section className="memory-ladder">
          <header><Database size={17} /><strong>Memory hierarchy</strong></header>
          <div className={writeLive ? "memory-level is-live" : "memory-level"}><span>HBM</span><code>~TB/s · 大容量</code><strong>X, dY, weight.grad</strong></div>
          <ArrowRight size={15} />
          <div className="memory-level"><span>L2 Cache</span><code>全 GPU 共享</code><strong>缓存 global load/store</strong></div>
          <ArrowRight size={15} />
          <div className={kernelLive ? "memory-level is-live" : "memory-level"}><span>L1 / Shared</span><code>每个 SM</code><strong>复用 X 和 dY tile</strong></div>
          <ArrowRight size={15} />
          <div className={kernelLive ? "memory-level is-live" : "memory-level"}><span>Registers</span><code>每线程私有</code><strong>累加 dW fragment</strong></div>
        </section>

        <section className="sm-fabric">
          <header><GridFour size={17} /><strong>GPU contains many Streaming Multiprocessors</strong><code>示意：8 / 实际型号可能更多</code></header>
          <div className="sm-grid">
            {Array.from({ length: 8 }, (_, index) => (
              <div className={index === 2 && kernelLive ? "sm-cell is-live" : "sm-cell"} key={index}>
                <strong>SM {index}</strong><span>{index === 2 ? "Block 17" : `Block ${index + 10}`}</span><small>resident warps</small>
              </div>
            ))}
          </div>
          <p className="scheduler-note"><strong>Block 不会跨 SM：</strong>一个 SM 可以同时驻留多个 Block，只要 registers、shared memory 和线程槽位足够。</p>
        </section>

        <section className={`sm-zoom${kernelLive ? " is-live" : ""}`}>
          <header><Lightning size={17} /><strong>放大 SM 2 · Block 17</strong><code>256 threads = 8 warps</code></header>
          <div className="sm-internals">
            <div className="warp-scheduler">
              <small>Warp Scheduler</small>
              <strong>选择 Warp 3</strong>
              <span>Warp 1 等 HBM，Warp 2 等 dependency；调度器切到 ready warp 隐藏延迟。</span>
            </div>
            <div className="warp-bank">
              <div className="warp-tabs">{Array.from({ length: 8 }, (_, i) => <span className={i === 3 ? "is-active" : ""} key={i}>W{i}</span>)}</div>
              <div className="lane-grid" aria-label="32 CUDA threads in warp 3">
                {Array.from({ length: 32 }, (_, i) => <span className={kernelLive ? "is-live" : ""} key={i}>{i}</span>)}
              </div>
              <p>Warp 内 32 个线程执行同一条指令，各自处理不同元素。分支不一致会让不同路径分批执行。</p>
            </div>
            <div className="execution-units">
              <div><small>Tensor Core / CUDA Core</small><strong>MMA / FMA</strong><code>acc += x × dy</code></div>
              <div><small>Register file</small><strong>thread-local accumulators</strong><code>r0…r7</code></div>
              <div><small>Shared memory</small><strong>cooperatively loaded tile</strong><code>__syncthreads()</code></div>
            </div>
          </div>
        </section>
      </div>

      <div className="granularity-chain" aria-label="Execution granularity">
        {[
          ["Tensor", "整个 dW"], ["Grid", "所有 Blocks"], ["Block", "一个 tile"], ["Warp", "32 threads"], ["Thread", "若干元素"], ["Instruction", "load / MMA / store"],
        ].map(([name, value], index) => <div key={name}><span>{index + 1}</span><strong>{name}</strong><small>{value}</small></div>)}
      </div>
    </div>
  );
}
