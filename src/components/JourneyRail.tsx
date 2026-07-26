import { Check, Circle, Play } from "@phosphor-icons/react";
import { lessonFor } from "../content/ddpLesson";
import { ddpScenario } from "../scenarios/ddp";
import { eventAtTime } from "../sim/simulator";
import { useSimulationStore } from "../store/simulation";

const groups = [
  { label: "01 反向计算", ids: ddpScenario.events.slice(0, 5).map((e) => e.id) },
  { label: "02 Ring All-Reduce", ids: ddpScenario.events.slice(5, 11).map((e) => e.id) },
  { label: "03 参数更新", ids: ddpScenario.events.slice(11).map((e) => e.id) },
];

export function JourneyRail() {
  const currentTime = useSimulationStore((state) => state.currentTime);
  const seek = useSimulationStore((state) => state.seek);
  const active = eventAtTime(ddpScenario.events, currentTime);

  return (
    <nav className="journey-rail" aria-label="梯度旅程步骤">
      <div className="rail-title">
        <span>一次训练迭代</span>
        <strong>14 个可观察事件</strong>
      </div>
      {groups.map((group) => (
        <section className="journey-group" key={group.label}>
          <p>{group.label}</p>
          {group.ids.map((id) => {
            const event = ddpScenario.events.find((item) => item.id === id)!;
            const lesson = lessonFor(event);
            const isActive = event.id === active.id;
            const isDone = currentTime >= event.start + event.duration;
            return (
              <button
                type="button"
                className={`journey-step${isActive ? " is-active" : ""}`}
                key={event.id}
                onClick={() => seek(event.start + 0.01)}
                aria-current={isActive ? "step" : undefined}
              >
                <span className="step-state" aria-hidden="true">
                  {isActive ? <Play size={10} weight="fill" /> : isDone ? <Check size={11} /> : <Circle size={9} />}
                </span>
                <span>
                  <small>{event.location}</small>
                  <strong>{lesson.title}</strong>
                </span>
              </button>
            );
          })}
        </section>
      ))}
    </nav>
  );
}
