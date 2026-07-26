import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Eye } from "@phosphor-icons/react/Eye";
import { AttentionPlayground } from "./AttentionPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Embedding", "Linear", "矩阵乘法"];

export function AttentionTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero attention-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">模型结构</a><span>/</span><strong>Attention</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Self-attention, one query at a time</p>
            <h1>一个 token，<br />怎样读取上下文。</h1>
            <p className="topic-lead">固定一个 Query，逐步观察它如何匹配所有 Key、生成概率，再按权重混合 Value。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>三种角色</span><strong>Query、Key、Value</strong></div>
            <div><span>匹配方式</span><strong>Scaled dot product</strong></div>
            <div><span>概率约束</span><strong>Mask + row-wise Softmax</strong></div>
            <div><span>系统代价</span><strong>T × T 中间状态</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("attention-lab")}><Eye size={17} weight="fill" aria-hidden="true" />选择一个 Query</button>
        </div>
      </section>

      <AttentionPlayground />

      <section className="project-note attention-project-note">
        <div><p className="eyebrow">Next connection</p><h2>公式看懂之后，下一问是如何避免反复读写 T×T 矩阵。</h2><p>先进入 GPU 内存层级理解 HBM、Shared Memory 与分块复用，再继续推导 FlashAttention 的 IO-aware 执行方式。</p></div>
        <a href="#/gpu/memory-hierarchy">进入内存层级<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
