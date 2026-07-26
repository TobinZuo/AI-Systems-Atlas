import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Target } from "@phosphor-icons/react/Target";
import { GradientPlayground } from "./GradientPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Autograd", "标量 Loss", "向量"];

export function GradientTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero gradient-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">训练机制</a><span>/</span><strong>Gradient</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Per-sample derivatives to parameter update</p>
            <h1>梯度不是答案，<br />而是当前位置的局部方向。</h1>
            <p className="topic-lead">跟着四条样本贡献，观察 batch reduction、`.grad` buffer、参数更新与 DDP 平均怎样连接。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>数学对象</span><strong>标量 Loss 对每个参数的偏导</strong></div>
            <div><span>Batch 语义</span><strong>样本贡献的 Sum 或 Mean</strong></div>
            <div><span>工程形态</span><strong>与参数同 shape/device 的 Tensor</strong></div>
            <div><span>系统连接</span><strong>Optimizer 输入与 DDP payload</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("gradient-lab")}><Target size={17} weight="fill" aria-hidden="true" />进入梯度实验</button>
        </div>
      </section>

      <GradientPlayground />

      <section className="project-note gradient-project-note">
        <div><p className="eyebrow">Next connection</p><h2>`.grad` 已经成为内存中的 Tensor，下一步看硬件怎样处理它。</h2><p>沿显存地址进入 CUDA Grid、Block、Warp、SM、寄存器与 HBM。</p></div>
        <a href="#/gpu/architecture">进入 GPU 执行模型<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
