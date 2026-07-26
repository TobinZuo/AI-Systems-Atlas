import {
  ArrowRight,
  CheckCircle,
  Info,
  MapPin,
} from "@phosphor-icons/react";
import { components, ddpScenario } from "../scenarios/ddp";
import { eventAtTime } from "../sim/simulator";
import { useSimulationStore } from "../store/simulation";

export function Inspector() {
  const currentTime = useSimulationStore((state) => state.currentTime);
  const selected = useSimulationStore((state) => state.selectedComponent);
  const active = eventAtTime(ddpScenario.events, currentTime);
  const component = components[selected];

  return (
    <aside className="inspector" aria-label="Event and component inspector">
      <div className="inspector-section current-event" aria-live="polite">
        <div className="inspector-heading">
          <span>Now happening</span>
          <strong>{active.layer}</strong>
        </div>
        <h2>{active.title}</h2>
        <p>{active.explanation}</p>

        <dl className="event-facts">
          <div>
            <dt>
              <MapPin size={15} aria-hidden="true" /> Location
            </dt>
            <dd>{active.location}</dd>
          </div>
          {active.tensor && (
            <div>
              <dt>
                <ArrowRight size={15} aria-hidden="true" /> Data
              </dt>
              <dd>{active.tensor}</dd>
            </div>
          )}
        </dl>

        <div className="detail-list">
          {active.details.map((detail) => (
            <div key={detail}>
              <CheckCircle size={16} weight="fill" aria-hidden="true" />
              <span>{detail}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="inspector-section component-detail">
        <div className="inspector-heading">
          <span>Selected component</span>
          <strong>{component.category}</strong>
        </div>
        <h3>{component.name}</h3>
        <p>{component.role}</p>
        <div className="knowledge-block">
          <Info size={16} weight="fill" aria-hidden="true" />
          <div>
            <strong>It knows</strong>
            <span>{component.knows}</span>
          </div>
        </div>
        <div className="knowledge-block is-muted">
          <Info size={16} aria-hidden="true" />
          <div>
            <strong>It does not know</strong>
            <span>{component.doesNotKnow}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
