import {
  ArrowCounterClockwise,
  CaretLeft,
  CaretRight,
  Pause,
  Play,
} from "@phosphor-icons/react";
import { ddpScenario } from "../scenarios/ddp";
import { eventAtTime } from "../sim/simulator";
import { useSimulationStore } from "../store/simulation";

const speeds = [0.5, 1, 1.5, 2];

export function EventTimeline() {
  const currentTime = useSimulationStore((state) => state.currentTime);
  const isPlaying = useSimulationStore((state) => state.isPlaying);
  const speed = useSimulationStore((state) => state.speed);
  const toggle = useSimulationStore((state) => state.toggle);
  const reset = useSimulationStore((state) => state.reset);
  const seek = useSimulationStore((state) => state.seek);
  const stepForward = useSimulationStore((state) => state.stepForward);
  const stepBackward = useSimulationStore((state) => state.stepBackward);
  const setSpeed = useSimulationStore((state) => state.setSpeed);
  const active = eventAtTime(ddpScenario.events, currentTime);

  return (
    <section className="timeline" aria-label="Simulation controls">
      <div className="timeline-controls">
        <button
          type="button"
          className="icon-button"
          onClick={reset}
          aria-label="Restart simulation"
          title="Restart"
        >
          <ArrowCounterClockwise size={17} />
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={stepBackward}
          aria-label="Previous event"
          title="Previous event"
        >
          <CaretLeft size={17} weight="bold" />
        </button>
        <button
          type="button"
          className="play-button"
          onClick={toggle}
          aria-label={isPlaying ? "Pause simulation" : "Play simulation"}
        >
          {isPlaying ? (
            <Pause size={18} weight="fill" />
          ) : (
            <Play size={18} weight="fill" />
          )}
          <span>{isPlaying ? "Pause" : "Play"}</span>
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={stepForward}
          aria-label="Next event"
          title="Next event"
        >
          <CaretRight size={17} weight="bold" />
        </button>
      </div>

      <div className="timeline-track-wrap">
        <div className="timeline-meta">
          <strong>{active.compactTitle}</strong>
          <span>
            {currentTime.toFixed(1)}s / {ddpScenario.totalDuration.toFixed(1)}s
          </span>
        </div>
        <div className="event-segments" aria-hidden="true">
          {ddpScenario.events.map((event) => (
            <span
              key={event.id}
              className={event.id === active.id ? "is-active" : ""}
              style={{ flexGrow: event.duration }}
            />
          ))}
        </div>
        <input
          className="timeline-range"
          type="range"
          min="0"
          max={ddpScenario.totalDuration}
          step="0.05"
          value={currentTime}
          onChange={(event) => seek(Number(event.target.value))}
          aria-label="Simulation time"
        />
      </div>

      <label className="speed-control">
        <span>Speed</span>
        <select
          value={speed}
          onChange={(event) => setSpeed(Number(event.target.value))}
        >
          {speeds.map((value) => (
            <option value={value} key={value}>
              {value}x
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
