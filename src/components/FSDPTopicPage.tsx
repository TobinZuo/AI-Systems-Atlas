import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Play } from "@phosphor-icons/react/Play";
import { useState } from "react";
import { DistributedTopicSwitcher } from "./DistributedTopicSwitcher";
import { FSDPPlayground } from "./FSDPPlayground";
import { scrollToSection } from "./scrollToSection";

export function FSDPTopicPage() {
  const [phaseIndex, setPhaseIndex] = useState(0);

  return (
    <main id="top">
      <section className="topic-hero compact-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span>
          <a href="#/">分布式训练</a><span>/</span>
          <strong>FSDP</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Fully Sharded Data Parallel</p>
            <h1>平时只留分片，<br />算到这一层才拼完整。</h1>
            <p className="topic-lead">
              参数、梯度和 optimizer state 都长期分片。每层计算前 All-Gather 权重，计算后立即 Reshard。
            </p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>
              <i>ZeRO-1</i><i>All-Gather</i><i>Reduce-Scatter</i><i>Autograd Hook</i>
            </div>
          </div>

          <div className="topic-brief">
            <div><span>解决的问题</span><strong>模型状态超过单卡 HBM</strong></div>
            <div><span>长期分片</span><strong>参数 + 梯度 + Optimizer state</strong></div>
            <div><span>临时完整</span><strong>当前计算层的权重</strong></div>
            <div><span>核心通信</span><strong>All-Gather + Reduce-Scatter</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} />返回知识地图</a>
          <div className="topic-action-cluster">
            <button type="button" className="secondary-action" onClick={() => { setPhaseIndex(0); scrollToSection("fsdp-playground"); }}><Play size={17} weight="fill" aria-hidden="true" />从头执行</button>
            <button type="button" className="primary-action" onClick={() => { setPhaseIndex(1); scrollToSection("fsdp-hardware-stage"); }}><HardDrives size={17} weight="fill" aria-hidden="true" />直接看硬件</button>
          </div>
        </div>
      </section>

      <DistributedTopicSwitcher current="fsdp" />
      <FSDPPlayground phaseIndex={phaseIndex} onPhaseIndexChange={setPhaseIndex} />

      <section className="project-note distributed-project-note">
        <div>
          <h2>FSDP 用通信换显存，也用生命周期控制峰值。</h2>
          <p>只展开当前层让模型可以超过单卡容量；代价是 forward 和 backward 都需要额外的权重 All-Gather。</p>
        </div>
        <a href="#/distributed/ddp">回到 DDP 对照<ArrowRight size={17} /></a>
      </section>
    </main>
  );
}
