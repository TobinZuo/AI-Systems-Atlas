import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Queue } from "@phosphor-icons/react/Queue";
import { CUDAStreamsPlayground } from "./CUDAStreamsPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Gradient", "CUDA Kernel", "GPU、SM 与 Warp"];

export function CUDAStreamsTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero stream-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">GPU 与性能</a><span>/</span><strong>CUDA Stream</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">CUDA asynchronous execution</p>
            <h1>同一张 GPU，<br />怎样边计算边通信。</h1>
            <p className="topic-lead">跟着 DDP 的一个梯度 bucket，看 CPU 怎样异步提交 backward 与 NCCL kernel，再用 CUDA Event 建立跨 Stream 依赖。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>Stream 是什么</span><strong>GPU 有序工作队列</strong></div>
            <div><span>重叠方法</span><strong>Compute + Comm Stream</strong></div>
            <div><span>正确性工具</span><strong>CUDA Event 依赖</strong></div>
            <div><span>追踪对象</span><strong>DDP gradient bucket</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("cuda-stream-lab")}><Queue size={17} weight="fill" aria-hidden="true" />打开时间线</button>
        </div>
      </section>

      <CUDAStreamsPlayground />

      <section className="project-note stream-project-note">
        <div><p className="eyebrow">Next connection</p><h2>Bucket 同步明白后，就能看懂 DDP 为什么能提速。</h2><p>DDP 在 backward 过程中按 bucket 触发 collective。Stream 决定工作怎样排队，Event 决定通信何时可以安全读取梯度。</p></div>
        <a href="#/distributed/ddp">进入 DDP<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
