import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Memory } from "@phosphor-icons/react/Memory";
import { MemoryHierarchyPlayground } from "./MemoryHierarchyPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Gradient", "GPU、SM 与 Warp", "CUDA Kernel"];

export function MemoryHierarchyTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero memory-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">GPU 与性能</a><span>/</span><strong>HBM、Cache 与片上存储</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">GPU memory hierarchy</p>
            <h1>计算只做一次，<br />数据为何反复搬运。</h1>
            <p className="topic-lead">跟着线性层权重梯度 dW，逐层观察 HBM、L2 Cache、Shared Memory、寄存器和 FMA pipeline 如何协作。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>追踪对象</span><strong>dW = Xᵀ × dY</strong></div>
            <div><span>片外存储</span><strong>HBM / Global memory</strong></div>
            <div><span>片上存储</span><strong>L2、Shared、Registers</strong></div>
            <div><span>优化核心</span><strong>分块后复用数据</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("memory-hierarchy-lab")}><Memory size={17} weight="fill" aria-hidden="true" />跟踪一块数据</button>
        </div>
      </section>

      <MemoryHierarchyPlayground />

      <section className="project-note memory-project-note">
        <div><p className="eyebrow">Next connection</p><h2>理解分块与片上复用，才能看懂 FlashAttention。</h2><p>Attention 的核心优化并不是少算注意力，而是重新安排 tile 的生命周期，避免把巨大的中间矩阵反复写入 HBM。</p></div>
        <a href="#/gpu/cuda-stream">进入 CUDA Stream<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
