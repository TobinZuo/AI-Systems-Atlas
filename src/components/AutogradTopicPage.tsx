import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Graph } from "@phosphor-icons/react/Graph";
import { AutogradPlayground } from "./AutogradPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Tensor", "标量导数", "Forward"];

export function AutogradTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero autograd-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">训练机制</a><span>/</span><strong>Autograd</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Forward values to backward execution</p>
            <h1>计算图不是公式截图，<br />而是一组可执行的反向依赖。</h1>
            <p className="topic-lead">跟着一个被复用的参数，看 PyTorch 如何边算 Forward 边建图，再用链式法则把梯度累加到叶子 Tensor。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>Forward 阶段</span><strong>执行算子并连接 backward Node</strong></div>
            <div><span>Backward 阶段</span><strong>反向拓扑调度与局部 VJP</strong></div>
            <div><span>参数结果</span><strong>多条贡献累加到 leaf .grad</strong></div>
            <div><span>安全边界</span><strong>no_grad 与 version counter</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("autograd-lab")}><Graph size={17} weight="fill" aria-hidden="true" />进入动态图实验</button>
        </div>
      </section>

      <AutogradPlayground />

      <section className="project-note autograd-project-note">
        <div><p className="eyebrow">Next connection</p><h2>Autograd 已经算出局部导数，下一步看这些数怎样成为训练用的梯度。</h2><p>从样本贡献、batch reduction 和 `.grad` buffer，一直连接到参数更新与 DDP。</p></div>
        <a href="#/training/gradient">进入 Gradient 实验<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
