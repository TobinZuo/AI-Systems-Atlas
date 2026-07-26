import { ArrowLeft } from "@phosphor-icons/react/ArrowLeft";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { SlidersHorizontal } from "@phosphor-icons/react/SlidersHorizontal";
import { AdamWPlayground } from "./AdamWPlayground";
import { scrollToSection } from "./scrollToSection";

const prerequisiteTopics = ["Gradient", "指数移动平均", "参数更新"];

export function AdamWTopicPage() {
  return (
    <main id="top">
      <section className="topic-hero adamw-topic-hero">
        <nav className="topic-breadcrumb" aria-label="面包屑">
          <a href="#/">知识地图</a><span>/</span><a href="#/">训练机制</a><span>/</span><strong>AdamW</strong>
        </nav>

        <div className="topic-hero-grid">
          <div>
            <p className="eyebrow">Gradient to persistent optimizer state</p>
            <h1>梯度只活一轮，<br />动量记住训练历史。</h1>
            <p className="topic-lead">跟着一个参数连续走 5 步，看 AdamW 怎样把当前 `.grad`、历史方向和权重衰减合成一次真实更新。</p>
            <div className="topic-prerequisites" aria-label="前置概念">
              <span>前置概念</span>{prerequisiteTopics.map((topic) => <i key={topic}>{topic}</i>)}
            </div>
          </div>

          <div className="topic-brief">
            <div><span>本轮输入</span><strong>parameter.grad</strong></div>
            <div><span>持久状态</span><strong>一阶动量 m 与二阶动量 v</strong></div>
            <div><span>直接输出</span><strong>原地更新 parameter、m、v</strong></div>
            <div><span>系统连接</span><strong>ZeRO/FSDP 要分片的状态</strong></div>
          </div>
        </div>

        <div className="topic-hero-actions">
          <a className="back-to-atlas" href="#/"><ArrowLeft size={16} aria-hidden="true" />返回知识地图</a>
          <button type="button" className="primary-action" onClick={() => scrollToSection("adamw-lab")}><SlidersHorizontal size={17} weight="fill" aria-hidden="true" />进入 AdamW 实验</button>
        </div>
      </section>

      <AdamWPlayground />

      <section className="project-note adamw-project-note">
        <div><p className="eyebrow">Next connection</p><h2>现在 `m`、`v` 已经占据显存，下一步看 ZeRO-1 怎样只保留自己负责的那份。</h2><p>参数与梯度继续复制，optimizer state 按 rank 分片，更新后再同步参数。</p></div>
        <a href="#/distributed/zero-1">进入 ZeRO-1<ArrowRight size={17} aria-hidden="true" /></a>
      </section>
    </main>
  );
}
