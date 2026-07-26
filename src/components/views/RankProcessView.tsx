import { ArrowDown, ArrowRight, Cpu, Database, Lightning } from "@phosphor-icons/react";
import type { SimulationEvent } from "../../domain/simulation";

const isOneOf = (event: SimulationEvent, ids: string[]) => ids.includes(event.id);

export function RankProcessView({ event }: { event: SimulationEvent }) {
  const hostActive = isOneOf(event, ["python-backward", "autograd-schedules"]);
  const computeActive = isOneOf(event, ["autograd-schedules", "gradient-kernel", "gradient-writeback"]);
  const hookActive = event.id === "bucket-ready";
  const commActive = event.kind === "reduce-scatter" || event.kind === "all-gather";
  const gradReady = event.start >= 4.6;

  return (
    <div className="rank-view detail-view">
      <div className="view-intro-row">
        <div>
          <span className="view-kicker">单机 · 进程视角</span>
          <h3>Rank 0 从 Python 到 GPU 的完整控制路径</h3>
        </div>
        <div className="rank-switcher" aria-label="Ranks share the same structure">
          {[0, 1, 2, 3].map((rank) => <span className={rank === 0 ? "is-active" : ""} key={rank}>R{rank}</span>)}
        </div>
      </div>

      <div className="rank-anatomy">
        <section className={`host-stack${hostActive ? " is-live" : ""}`}>
          <header><Cpu size={17} /><strong>Host CPU · Linux Process</strong><code>PID 4100 / rank 0</code></header>
          <div className="host-thread">
            <div className="thread-label"><span>Python main thread</span><code>loss.backward()</code></div>
            <ArrowDown size={15} />
            <div className="runtime-row">
              <div><small>Autograd ready queue</small><strong>MmBackward0</strong><code>pending_deps = 0</code></div>
              <ArrowRight size={16} />
              <div><small>C++ worker thread</small><strong>Engine::evaluate</strong><code>launch, 不逐元素算</code></div>
              <ArrowRight size={16} />
              <div><small>CUDA Runtime</small><strong>kernel launch</strong><code>device pointer + shape</code></div>
            </div>
          </div>
        </section>

        <div className="launch-bridge">
          <span>PCIe command submission</span>
          <ArrowDown size={18} />
          <small>CPU 把命令写入 GPU work queue，调用本身通常异步返回</small>
        </div>

        <section className="cuda-context">
          <header><Lightning size={17} /><strong>CUDA Context · GPU 0</strong><code>device=0</code></header>
          <div className="stream-grid">
            <div className={`stream-lane${computeActive ? " is-live" : ""}`}>
              <div className="stream-name"><span>Compute stream</span><code>cudaStream_t 0xA0</code></div>
              <div className="queue">
                <span className="queue-item is-done">forward GEMM</span>
                <span className="queue-item is-current">gemm_backward</span>
                <span className={`queue-item${hookActive || commActive ? " is-current" : ""}`}>record Event E0</span>
              </div>
            </div>
            <div className={`stream-lane${hookActive || commActive ? " is-live" : ""}`}>
              <div className="stream-name"><span>Communication stream</span><code>cudaStream_t 0xC0</code></div>
              <div className="queue">
                <span className="queue-item">wait Event E0</span>
                <span className={`queue-item${commActive ? " is-current" : ""}`}>ncclKernel</span>
                <span className="queue-item">write bucket output</span>
              </div>
            </div>
          </div>
          <div className="event-contract">
            <span className={hookActive || commActive ? "event-dot is-live" : "event-dot"}>E0</span>
            <p><strong>Event 是依赖契约：</strong>通信流必须看到 E0 完成，才能读取 compute stream 写入的 bucket。两个 Stream 不共享“当前执行到哪”的隐式知识。</p>
          </div>
        </section>

        <section className={`memory-contract${gradReady ? " is-live" : ""}`}>
          <header><Database size={17} /><strong>GPU HBM · 本地虚拟地址空间</strong></header>
          <div className="address-row">
            <div><small>parameter.grad</small><code>0x7f20_0000</code><strong>{gradReady ? "[1, 2, 3, 4, 5, 6, 7, 8]" : "等待 kernel 写入"}</strong></div>
            <ArrowRight size={17} />
            <div><small>DDP bucket view</small><code>offset 0 · fp32 · count 8</code><strong>语义映射由 PyTorch 保存</strong></div>
            <ArrowRight size={17} />
            <div><small>NCCL local API</small><code>ptr, count, dtype, op</code><strong>不理解“第几层梯度”</strong></div>
          </div>
        </section>
      </div>
    </div>
  );
}
