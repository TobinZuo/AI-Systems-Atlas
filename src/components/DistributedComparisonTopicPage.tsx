import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { Play } from "@phosphor-icons/react/Play";
import { DistributedComparisonPlayground } from "./DistributedComparisonPlayground";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { scrollToSection } from "./scrollToSection";

export function DistributedComparisonTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero compact-topic-hero comparison-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span>
          <a href="#/distributed/ddp">分布式训练</a><span>/</span>
          <strong>策略对比</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Distributed state comparison</p>
            <h1>三种策略，<br />切的是不同模型状态。</h1>
            <p className="topic-lead">
              DDP 复制全部状态，ZeRO-1 只切 optimizer state，FSDP 再把参数和梯度一起切开。
            </p>
            <div className="topic-prerequisites" aria-label="对比对象">
              <span>对比对象</span>
              <i>DDP</i><i>Sharded Optimizer</i><i>FSDP</i><i>AdamW</i>
            </div>
          </div>

          <div className="topic-brief">
            <div><span>统一变量</span><strong>参数量 + World size</strong></div>
            <div><span>显存口径</span><strong>W + dW + m + v</strong></div>
            <div><span>执行对照</span><strong>同步 + 更新 + 下一次 Forward</strong></div>
            <div><span>核心问题</span><strong>省了什么，又多通信了什么</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("distributed-comparison-playground")}><Play size={17} weight="fill" aria-hidden="true" />开始横向对照</button>
        </div>
      </section>

      <DistributedTopicSwitcher current="compare" />
      <DistributedComparisonPlayground />

      <section className="project-note distributed-project-note">
        <div>
          <h2>切分不是目的，目标是让训练状态装得下。</h2>
          <p>先定位哪类状态造成显存压力，再判断新增通信、临时 buffer 和实现复杂度是否值得。</p>
        </div>
        <a href="#/distributed/zero-1">拆开看 Sharded Optimizer</a>
      </section>
    </main>
  );
}
