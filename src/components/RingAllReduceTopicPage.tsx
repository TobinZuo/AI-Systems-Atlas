import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { ArrowsClockwise } from "@phosphor-icons/react/ArrowsClockwise";
import { DistributedSystemStack } from "./DistributedSystemStack";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { RingAllReducePlayground } from "./RingAllReducePlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Process 与 Rank", "Collective", "Reduce-Scatter", "All-Gather"];

export function RingAllReduceTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero ring-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑"><a href="#/">知识地图</a><span>/</span><a href="#/">分布式训练</a><span>/</span><strong>Ring AllReduce</strong></nav>
        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Bandwidth-oriented collective schedule</p>
            <h1>每轮发哪个 chunk，<br />早已写进公式。</h1>
            <p className="topic-lead">把 M 字节切成 N 份，用模运算安排 Reduce-Scatter 与 All-Gather。每轮所有链路并行，但每个 Rank 发送不同的数据身份。</p>
            <div className="topic-prerequisites" aria-label="前置概念"><span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}</div>
          </div>
          <div className="topic-brief">
            <div><span>要实现的契约</span><strong>AllReduce</strong></div>
            <div><span>两段算法</span><strong>Reduce-Scatter + All-Gather</strong></div>
            <div><span>总顺序步骤</span><strong>2(N-1)</strong></div>
            <div><span>每 Rank 通信量</span><strong>2(N-1)M / N</strong></div>
          </div>
        </div>
        <div className="topic-hero-actions"><a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a><button type="button" className="primary-action" onClick={() => scrollToSection("ring-round-stage")}><ArrowsClockwise size={17} weight="fill" aria-hidden="true" />拆开第一轮</button></div>
      </section>

      <DistributedTopicSwitcher current="ring-allreduce" />
      <DistributedSystemStack topic="ring-allreduce" />
      <RingAllReducePlayground />

      <section className="project-note ring-project-note"><div><p className="eyebrow">Next connection</p><h2>Ring 解释了梯度怎样归约，DDP 决定它何时开始。</h2><p>下一步把 Autograd hook、bucket ready、Comm stream 与 Ring AllReduce 接回同一条训练时间线。</p></div><a href="#/distributed/ddp">进入 DDP<ArrowRight size={17} aria-hidden="true" /></a></section>
    </main>
  );
}
