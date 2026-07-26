import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Code } from "@phosphor-icons/react/Code";
import { CUDAKernelJourneyPlayground } from "./CUDAKernelJourneyPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Gradient", "GPU、SM 与 Warp", "Tensor device"];

export function CUDAKernelTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero kernel-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">GPU 与性能</a><span>/</span><strong>CUDA Kernel</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Operator to device execution</p>
            <h1>一行 Tensor 运算，<br />如何落成 GPU Kernel。</h1>
            <p className="topic-lead">跟着一次梯度缩放，穿过 PyTorch Dispatcher、CUDA Runtime、Compute Stream，直到 GPU 从 HBM 读取真实数据。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>统一语义</span><strong>ATen operator schema</strong></div>
            <div><span>后端选择</span><strong>PyTorch Dispatcher</strong></div>
            <div><span>设备任务</span><strong>Kernel + Grid + Block</strong></div>
            <div><span>关键边界</span><strong>Host launch 异步返回</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("cuda-kernel-journey")}><Code size={17} weight="fill" aria-hidden="true" />进入调用链</button>
        </div>
      </section>

      <CUDAKernelJourneyPlayground />

      <section className="project-note kernel-project-note">
        <div><p className="eyebrow">Next connection</p><h2>Kernel 入队之后，执行顺序由 Stream 和 Event 管理。</h2><p>下一步把单次 launch 放进多条队列，观察 backward kernel 与 NCCL kernel 怎样安全重叠。</p></div>
        <a href="#/gpu/cuda-stream">进入 CUDA Stream<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
