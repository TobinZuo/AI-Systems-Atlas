import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { GPUArchitecturePlayground } from "./GPUArchitecturePlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Gradient", "向量索引", "一行 PyTorch"];

export function GPUArchitectureTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero gpu-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">GPU 与性能</a><span>/</span><strong>GPU、SM 与 Warp</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">GPU execution model</p>
            <h1>跟着一个梯度元素，<br />进入 GPU 执行现场。</h1>
            <p className="topic-lead">CPU 发起一次 kernel，GPU 把向量拆成 Grid、Block、Warp 和 Lane，并在寄存器与 HBM 之间完成计算。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>解决的问题</span><strong>大量元素怎样并行计算</strong></div>
            <div><span>编程层级</span><strong>Grid → Block → Thread</strong></div>
            <div><span>硬件层级</span><strong>GPU → SM → Warp → Lane</strong></div>
            <div><span>追踪对象</span><strong>grad[i] × scale</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("gpu-architecture-lab")}><Cpu size={17} weight="fill" aria-hidden="true" />进入 GPU</button>
        </div>
      </section>

      <GPUArchitecturePlayground />

      <section className="project-note gpu-project-note">
        <div><p className="eyebrow">Executable explanation</p><h2>不是背术语，是定位一个具体元素。</h2><p>改动 n、blockDim 和 SM 数量，观察同一个索引公式怎样改变 Grid、尾部 Warp、显存事务和调度波次。</p></div>
        <a href="#/distributed/ddp">继续跟踪梯度<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
