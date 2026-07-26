import { ArrowRight, CheckCircle, WarningCircle } from "@phosphor-icons/react";
import { lessonFor } from "../content/ddpLesson";
import { ddpScenario } from "../scenarios/ddp";
import { eventAtTime } from "../sim/simulator";
import { useSimulationStore } from "../store/simulation";

export function DetailedInspector() {
  const currentTime = useSimulationStore((state) => state.currentTime);
  const active = eventAtTime(ddpScenario.events, currentTime);
  const lesson = lessonFor(active);

  return (
    <aside className="detailed-inspector" aria-label="当前事件解释" aria-live="polite">
      <section className="inspector-lead">
        <div className="inspector-heading"><span>当前发生</span><strong>{lesson.eyebrow}</strong></div>
        <h2>{lesson.title}</h2>
        <p>{lesson.summary}</p>
      </section>

      <section className="causal-chain">
        <span className="section-label">因果链</span>
        <div><small>为什么现在能执行</small><p>{lesson.cause}</p></div>
        <ArrowRight size={16} />
        <div><small>这一层具体做什么</small><p>{lesson.action}</p></div>
        <ArrowRight size={16} />
        <div><small>产生什么可观察结果</small><p>{lesson.result}</p></div>
      </section>

      <section className="call-stack">
        <span className="section-label">调用 / 数据路径</span>
        <div>{lesson.callStack.map((line, index) => <code key={line}><span>{String(index + 1).padStart(2, "0")}</span>{line}</code>)}</div>
      </section>

      <section className="idea-block">
        <CheckCircle size={17} weight="fill" />
        <div><span>这里体现的系统思想</span><p>{lesson.systemIdea}</p></div>
      </section>

      <section className="misconception-block">
        <WarningCircle size={18} />
        <div><span>容易混淆</span><p>{lesson.misconception}</p></div>
      </section>
    </aside>
  );
}
