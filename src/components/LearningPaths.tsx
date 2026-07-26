import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Compass } from "@phosphor-icons/react/Compass";
import { learningJourneys, topicIndex } from "../content/topics";

export function LearningPaths() {
  return (
    <section className="learning-journeys" id="learning-paths" aria-labelledby="learning-paths-title">
      <div className="atlas-section-heading">
        <div>
          <p className="eyebrow">Learning paths</p>
          <h2 id="learning-paths-title">路线不是目录，而是理解问题的顺序。</h2>
        </div>
        <p>同一个主题可以出现在多条路线中。你可以跟着梯度、token 或一次推理请求穿过知识地图。</p>
      </div>

      <div className="journey-list">
        {learningJourneys.map((journey, journeyIndex) => {
          const isActive = journey.status === "active";
          return (
            <article className={`journey-row${isActive ? " is-active" : ""}`} key={journey.id}>
              <div className="journey-summary">
                <div className="journey-number" aria-hidden="true">
                  {isActive ? <CheckCircle size={22} weight="fill" /> : <Compass size={22} weight="duotone" />}
                  <span>0{journeyIndex + 1}</span>
                </div>
                <p className="eyebrow">{journey.eyebrow}</p>
                <h3>{journey.title}</h3>
                <p>{journey.description}</p>
              </div>

              <ol className="journey-steps" aria-label={`${journey.title}的主题顺序`}>
                {journey.topicIds.map((topicId, index) => {
                  const topic = topicIndex.get(topicId)!;
                  const content = (
                    <>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <strong>{topic.title}</strong>
                    </>
                  );

                  return (
                    <li className={topic.status === "available" ? "is-available" : ""} key={topic.id}>
                      {topic.route ? <a href={`#${topic.route}`}>{content}</a> : <div>{content}</div>}
                      {index < journey.topicIds.length - 1 && <ArrowRight size={14} aria-hidden="true" />}
                    </li>
                  );
                })}
              </ol>

              <div className="journey-state">
                <span>{isActive ? "已有可交互节点" : "路线已规划"}</span>
                {isActive && (
                  <a href="#/training/autograd">
                    从 Autograd 开始
                    <ArrowRight size={16} aria-hidden="true" />
                  </a>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
