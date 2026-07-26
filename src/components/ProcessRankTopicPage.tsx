import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Network } from "@phosphor-icons/react/Network";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { DistributedSystemStack } from "./DistributedSystemStack";
import { ProcessRankPlayground } from "./ProcessRankPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["OS Process", "TCP endpoint", "GPU device"];

export function ProcessRankTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero rank-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">分布式训练</a><span>/</span><strong>Process 与 Rank</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Runtime identity and rendezvous</p>
            <h1>进程彼此看不见，<br />通信组让它们成为一个任务。</h1>
            <p className="topic-lead">从 torchrun 启动四个 OS 进程开始，看 rank、环境变量、TCPStore、Process Group 与 GPU 绑定怎样逐层建立共同身份。</p>
            <div className="topic-prerequisites" aria-label="前置概念"><span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}</div>
          </div>

          <div className="topic-brief">
            <div><span>执行实体</span><strong>每个 rank 一个 OS 进程</strong></div>
            <div><span>全局身份</span><strong>RANK ∈ [0, WORLD_SIZE)</strong></div>
            <div><span>本机映射</span><strong>LOCAL_RANK → CUDA device</strong></div>
            <div><span>加入同一任务</span><strong>Rendezvous + Process Group</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("process-rank-lab")}><Network size={17} weight="fill" aria-hidden="true" />启动四个 rank</button>
        </div>
      </section>

      <DistributedTopicSwitcher current="process-rank" />
      <DistributedSystemStack topic="process-rank" />
      <ProcessRankPlayground />

      <section className="project-note rank-project-note">
        <div><p className="eyebrow">Next connection</p><h2>身份与通信组已经建立，下一步研究所有 rank 必须一起调用的 Collective。</h2><p>Broadcast、Reduce、AllReduce 和 AllGather 共享什么抽象，又为什么不能少一个参与者。</p></div>
        <a href="#/distributed/collective">进入 Collective<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
