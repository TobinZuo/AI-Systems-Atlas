import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Network } from "@phosphor-icons/react/Network";
import { CollectivePlayground } from "./CollectivePlayground";
import { DistributedSystemStack } from "./DistributedSystemStack";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Process 与 Rank", "Process Group", "Tensor buffer"];

export function CollectiveTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero collective-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑"><a href="#/">知识地图</a><span>/</span><a href="#/">分布式训练</a><span>/</span><strong>Collective</strong></nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Collective communication contracts</p>
            <h1>Collective 不是一条链路，<br />而是所有 Rank 共同签的契约。</h1>
            <p className="topic-lead">在同一组四张 GPU 上切换九类操作，直接观察输入、接收槽位、最终结果和 Backend 调用参数怎样一起变化。</p>
            <div className="topic-prerequisites" aria-label="前置概念"><span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}</div>
          </div>

          <div className="topic-brief">
            <div><span>参与者</span><strong>Process Group 中的所有 Rank</strong></div>
            <div><span>调用契约</span><strong>相同顺序、count 与 dtype</strong></div>
            <div><span>语义层</span><strong>规定输入怎样变成输出</strong></div>
            <div><span>实现层</span><strong>NCCL 选择算法与传输路径</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions"><a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a><button type="button" className="primary-action" onClick={() => scrollToSection("collective-lab")}><Network size={17} weight="fill" aria-hidden="true" />操作 Collective</button></div>
      </section>

      <DistributedTopicSwitcher current="collective" />
      <DistributedSystemStack topic="collective" />
      <CollectivePlayground />

      <section className="project-note collective-project-note"><div><p className="eyebrow">Next connection</p><h2>Collective 的输入输出契约已经清楚，下一步拆开 Ring AllReduce 的逐轮调度。</h2><p>ReduceScatter 和 AllGather 怎样各走 N-1 轮，又怎样让每个 rank 每轮发送不同 chunk。</p></div><a href="#/distributed/ddp">先在 DDP 中查看 Ring<ArrowRight size={17} aria-hidden="true" /></a></section>
    </main>
  );
}
