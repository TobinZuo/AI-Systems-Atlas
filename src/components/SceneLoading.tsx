export function SceneLoading() {
  return (
    <div className="scene-state" aria-live="polite" aria-label="Loading 3D scene">
      <div className="scene-skeleton" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
      </div>
      <strong>Preparing the system model</strong>
      <span>Loading the GPU topology and simulation state.</span>
    </div>
  );
}
