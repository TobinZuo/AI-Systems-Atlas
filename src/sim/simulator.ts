import type { Scenario, SimulationEvent } from "../domain/simulation";

export function eventAtTime(
  events: SimulationEvent[],
  time: number,
): SimulationEvent {
  if (events.length === 0) {
    throw new Error("A scenario must contain at least one event.");
  }

  const clamped = Math.max(0, time);
  return (
    [...events]
      .reverse()
      .find((event) => clamped >= event.start) ?? events[0]
  );
}

export function eventProgress(event: SimulationEvent, time: number): number {
  if (event.duration <= 0) return 1;
  return Math.min(1, Math.max(0, (time - event.start) / event.duration));
}

export function nextEventTime(
  events: SimulationEvent[],
  currentTime: number,
): number {
  return (
    events.find((event) => event.start > currentTime + 0.001)?.start ??
    events[events.length - 1]?.start ??
    0
  );
}

export function previousEventTime(
  events: SimulationEvent[],
  currentTime: number,
): number {
  const currentIndex = events.findIndex(
    (event) => event.id === eventAtTime(events, currentTime).id,
  );
  return events[Math.max(0, currentIndex - 1)]?.start ?? 0;
}

export function validateScenario(scenario: Scenario): string[] {
  const errors: string[] = [];
  const ids = new Set<string>();

  scenario.events.forEach((event, index) => {
    if (ids.has(event.id)) errors.push(`Duplicate event id: ${event.id}`);
    ids.add(event.id);

    if (event.duration <= 0) {
      errors.push(`Event ${event.id} must have positive duration.`);
    }
    if (index > 0 && event.start < scenario.events[index - 1].start) {
      errors.push(`Event ${event.id} starts before the previous event.`);
    }
  });

  scenario.events.forEach((event) => {
    event.dependencies?.forEach((dependency) => {
      const dependencyEvent = scenario.events.find((item) => item.id === dependency);
      if (!dependencyEvent) {
        errors.push(`Event ${event.id} has unknown dependency ${dependency}.`);
      } else if (dependencyEvent.start >= event.start) {
        errors.push(`Event ${event.id} depends on a non-earlier event.`);
      }
    });
  });

  const lastEvent = scenario.events[scenario.events.length - 1];
  if (lastEvent && lastEvent.start + lastEvent.duration > scenario.totalDuration) {
    errors.push("The final event exceeds the scenario duration.");
  }

  return errors;
}
