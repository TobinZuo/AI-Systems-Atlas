import { create } from "zustand";
import type { ComponentId } from "../domain/simulation";
import { ddpScenario } from "../scenarios/ddp";
import { nextEventTime, previousEventTime } from "../sim/simulator";

interface SimulationStore {
  currentTime: number;
  isPlaying: boolean;
  speed: number;
  selectedComponent: ComponentId;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  reset: () => void;
  seek: (time: number) => void;
  tick: (delta: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  setSpeed: (speed: number) => void;
  selectComponent: (component: ComponentId) => void;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  currentTime: 0,
  isPlaying: false,
  speed: 1,
  selectedComponent: "gpu-0",
  play: () => {
    if (get().currentTime >= ddpScenario.totalDuration) {
      set({ currentTime: 0, isPlaying: true });
    } else {
      set({ isPlaying: true });
    }
  },
  pause: () => set({ isPlaying: false }),
  toggle: () => (get().isPlaying ? get().pause() : get().play()),
  reset: () => set({ currentTime: 0, isPlaying: false }),
  seek: (time) =>
    set({
      currentTime: Math.min(ddpScenario.totalDuration, Math.max(0, time)),
    }),
  tick: (delta) => {
    const next = get().currentTime + delta * get().speed;
    if (next >= ddpScenario.totalDuration) {
      set({ currentTime: ddpScenario.totalDuration, isPlaying: false });
    } else {
      set({ currentTime: next });
    }
  },
  stepForward: () =>
    set({
      currentTime: nextEventTime(ddpScenario.events, get().currentTime),
      isPlaying: false,
    }),
  stepBackward: () =>
    set({
      currentTime: previousEventTime(ddpScenario.events, get().currentTime),
      isPlaying: false,
    }),
  setSpeed: (speed) => set({ speed }),
  selectComponent: (selectedComponent) => set({ selectedComponent }),
}));
