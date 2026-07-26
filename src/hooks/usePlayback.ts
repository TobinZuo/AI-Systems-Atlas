import { useEffect } from "react";
import { useSimulationStore } from "../store/simulation";

export function usePlayback(): void {
  const isPlaying = useSimulationStore((state) => state.isPlaying);
  const tick = useSimulationStore((state) => state.tick);

  useEffect(() => {
    if (!isPlaying) return;

    const interval = window.setInterval(() => tick(0.08), 80);
    return () => window.clearInterval(interval);
  }, [isPlaying, tick]);
}
