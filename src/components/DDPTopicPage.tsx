import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { SlidersHorizontal } from "@phosphor-icons/react/SlidersHorizontal";
import { useState } from "react";
import { AtlasModeBar, type AtlasMode } from "./AtlasModeBar";
import { DDPPlayground } from "./DDPPlayground";
import { TraceWorkspace } from "./TraceWorkspace";

const prerequisiteTopics = ["Gradient", "CUDA Stream", "集合通信", "Ring AllReduce"];

export function DDPTopicPage() {
  const [mode, setMode] = useState<AtlasMode>("concept");
  const [conceptEventId, setConceptEventId] = useState<string | null>(null);

  const openConceptEvent = (eventId: string) => {
    setConceptEventId(eventId);
    setMode("concept");
  };

  return (
    <main id="top">
      <section className="topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a>
          <span>/</span>
          <a href="#/">分布式训练</a>
          <span>/</span>
          <strong>DDP</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Distributed Data Parallel</p>
            <h1>跟着一个梯度，<br />走进 DDP 系统底层。</h1>
            <p className="topic-lead">
              多张 GPU 各算一部分数据，再用集合通信同步梯度，让所有模型副本得到完全相同的参数更新。
            </p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>
              {prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>解决的问题</span><strong>单卡训练太慢</strong></div>
            <div><span>核心方法</span><strong>模型复制 + 梯度同步</strong></div>
            <div><span>关键集合通信</span><strong>AllReduce</strong></div>
            <div><span>当前实现证据</span><strong>CS336 A2 + Simulator + Trace</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/">
            <ArrowLeft size={16} aria-hidden="true" />
            返回知识地图
          </a>
          <a className="primary-action" href={mode === "concept" ? "#ddp-playground" : "#trace-workspace"}>
            <SlidersHorizontal size={17} weight="fill" aria-hidden="true" />
            {mode === "concept" ? "开始操作" : "查看时间线"}
          </a>
        </div>
      </section>

      <AtlasModeBar mode={mode} onChange={setMode} />

      {mode === "concept"
        ? <DDPPlayground focusEventId={conceptEventId} />
        : <TraceWorkspace onOpenConcept={openConceptEvent} />}

      <section className="project-note ddp-project-note">
        <div>
          <p className="eyebrow">Executable explanation</p>
          <h2>不是看动画，是亲手改变系统状态。</h2>
          <p>每个数字和 chunk 状态都来自可测试的模拟器。修改任意 rank 的梯度，所有中间结果立即重新计算。</p>
        </div>
        <a href="https://github.com/TobinZuo/AI-Systems-Atlas" target="_blank" rel="noreferrer">
          查看模拟器源码
          <ArrowRight size={17} aria-hidden="true" />
        </a>
      </section>
    </main>
  );
}
