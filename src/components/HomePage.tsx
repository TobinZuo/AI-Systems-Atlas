import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { BookOpen } from "@phosphor-icons/react/BookOpen";
import { Code } from "@phosphor-icons/react/Code";
import { MapTrifold } from "@phosphor-icons/react/MapTrifold";
import { Pulse } from "@phosphor-icons/react/Pulse";
import { availableTopicCount, knowledgeDomains, topicCount } from "../content/topics";
import { KnowledgeMap } from "./KnowledgeMap";
import { LearningPaths } from "./LearningPaths";

const ddpFlow = ["Backward", "Bucket", "Reduce-Scatter", "All-Gather", "AdamW"];

export function HomePage() {
  return (
    <main id="top">
      <section className="atlas-hero">
        <div className="atlas-hero-copy">
          <p className="eyebrow">Interactive knowledge atlas</p>
          <h1>把大模型拆成<br />可以操作的系统专题。</h1>
          <p>
            从 token、Attention 和 Autograd，一路看到 GPU Kernel、集合通信、推理服务与多模态生成。每个概念都放回真实数据流里理解。
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#/distributed/ddp">
              进入 DDP 实验台
              <ArrowRight size={17} aria-hidden="true" />
            </a>
            <a className="secondary-action" href="#/paths">
              <MapTrifold size={17} aria-hidden="true" />
              选择学习路线
            </a>
          </div>
          <dl className="atlas-stats">
            <div><dt>{knowledgeDomains.length}</dt><dd>知识域</dd></div>
            <div><dt>{topicCount}</dt><dd>主题节点</dd></div>
            <div><dt>{availableTopicCount}</dt><dd>可运行专题</dd></div>
          </dl>
        </div>

        <a className="live-topic-preview" href="#/distributed/ddp" aria-label="进入 DDP 一个梯度的旅程专题">
          <div className="preview-heading">
            <span><i />当前可运行专题</span>
            <strong>DDP</strong>
          </div>
          <div className="preview-packet">
            <span>gradient chunk</span>
            <code>[1, 2] · fp32 · 8 B</code>
          </div>
          <div className="preview-flow" aria-hidden="true">
            {ddpFlow.map((step, index) => (
              <div className="preview-step" key={step}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <strong>{step}</strong>
                {index < ddpFlow.length - 1 && <i />}
              </div>
            ))}
          </div>
          <div className="preview-footer">
            <span>跟踪数据穿过框架、GPU、显存与网络</span>
            <ArrowRight size={18} aria-hidden="true" />
          </div>
        </a>
      </section>

      <KnowledgeMap />
      <LearningPaths />

      <section className="evidence-model" aria-labelledby="evidence-title">
        <div className="evidence-heading">
          <p className="eyebrow">How a topic is built</p>
          <h2 id="evidence-title">每个专题都回答四层问题。</h2>
          <p>每个结论都连接到运行状态、系统轨迹和可复现证据，不依赖特定课程的章节顺序。</p>
        </div>
        <ol className="evidence-steps">
          <li>
            <BookOpen size={20} weight="duotone" aria-hidden="true" />
            <span>01</span><strong>心智模型</strong><p>它解决什么问题，为什么需要它。</p>
          </li>
          <li>
            <MapTrifold size={20} weight="duotone" aria-hidden="true" />
            <span>02</span><strong>可操作状态</strong><p>修改输入，亲眼看中间状态怎样变化。</p>
          </li>
          <li>
            <Pulse size={20} weight="duotone" aria-hidden="true" />
            <span>03</span><strong>系统时间线</strong><p>把框架、Kernel、内存和通信放到同一时间轴。</p>
          </li>
          <li>
            <Code size={20} weight="duotone" aria-hidden="true" />
            <span>04</span><strong>源码证据</strong><p>关联实现、测试、Profiler 与论文。</p>
          </li>
        </ol>
      </section>
    </main>
  );
}
