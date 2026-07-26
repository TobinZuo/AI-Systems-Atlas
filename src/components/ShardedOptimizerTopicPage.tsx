import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Play } from "@phosphor-icons/react/Play";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { ShardedOptimizerPlayground } from "./ShardedOptimizerPlayground";
import { scrollToSection } from "./scrollToSection";

export function ShardedOptimizerTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero compact-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span>
          <a href="#/">分布式训练</a><span>/</span>
          <strong>ZeRO-1</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Sharded Optimizer</p>
            <h1>模型仍然复制，<br />AdamW 状态不再复制。</h1>
            <p className="topic-lead">
              每个参数只交给一个 owner 更新并保存 m、v，更新后再广播参数，让所有模型副本恢复一致。
            </p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>
              <i>DDP</i><i>AdamW</i><i>Parameter Group</i><i>Broadcast</i>
            </div>
          </div>

          <div className="topic-brief">
            <div><span>解决的问题</span><strong>AdamW m、v 在每张卡重复保存</strong></div>
            <div><span>长期分片</span><strong>Optimizer state</strong></div>
            <div><span>仍然复制</span><strong>参数 + 梯度</strong></div>
            <div><span>一致性操作</span><strong>Owner update + Broadcast</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("sharded-optimizer-playground")}><Play size={17} weight="fill" aria-hidden="true" />开始逐步执行</button>
        </div>
      </section>

      <DistributedTopicSwitcher current="zero-1" />
      <ShardedOptimizerPlayground />

      <section className="project-note distributed-project-note">
        <div>
          <h2>关键不是少更新参数，而是改变状态归属。</h2>
          <p>全局语义仍与普通 AdamW 相同。区别只在于谁保存状态、谁执行更新，以及怎样恢复参数副本一致。</p>
        </div>
        <a href="#/distributed/fsdp">继续看 FSDP<ArrowRight size={17} /></a>
      </section>
    </main>
  );
}
