import { Html, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { ComponentId, SimulationEvent } from "../domain/simulation";
import { useSimulationStore } from "../store/simulation";

type Point = [number, number, number];

const gpuPositions: Point[] = [
  [-3.4, 0, 2.25],
  [3.4, 0, 2.25],
  [3.4, 0, -2.25],
  [-3.4, 0, -2.25],
];

const surface = "#22272a";
const surfaceTop = "#343b3f";
const accent = "#e98245";
const quiet = "#697277";

function DeviceNode({
  position,
  rank,
  active,
  selected,
  reducedMotion,
}: {
  position: Point;
  rank: number;
  active: boolean;
  selected: boolean;
  reducedMotion: boolean;
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const select = useSimulationStore((state) => state.selectComponent);

  useFrame(({ clock }) => {
    if (!mesh.current) return;
    const pulse = active && !reducedMotion ? Math.sin(clock.elapsedTime * 4) * 0.025 : 0;
    const target = selected ? 1.06 : 1 + pulse;
    mesh.current.scale.lerp(new THREE.Vector3(target, target, target), 0.12);
  });

  return (
    <group position={position}>
      <mesh
        ref={mesh}
        onClick={(event) => {
          event.stopPropagation();
          select(`gpu-${rank}` as ComponentId);
        }}
      >
        <boxGeometry args={[2.35, 0.58, 1.48]} />
        <meshStandardMaterial
          color={active ? "#3b3631" : surface}
          emissive={active ? "#4d2411" : "#000000"}
          emissiveIntensity={active ? 0.35 : 0}
          roughness={0.62}
          metalness={0.38}
        />
      </mesh>
      <mesh position={[0, 0.34, 0]}>
        <boxGeometry args={[1.72, 0.12, 0.92]} />
        <meshStandardMaterial color={selected ? accent : surfaceTop} roughness={0.5} />
      </mesh>
      <mesh position={[0.72, 0.43, -0.24]}>
        <boxGeometry args={[0.2, 0.08, 0.2]} />
        <meshStandardMaterial color={active ? accent : quiet} />
      </mesh>
      <Html position={[0, 0.76, 0]} center distanceFactor={8}>
        <button
          type="button"
          className={selected ? "scene-label is-selected" : "scene-label"}
          onClick={() => select(`gpu-${rank}` as ComponentId)}
        >
          <strong>GPU {rank}</strong>
          <span>rank {rank}</span>
        </button>
      </Html>
    </group>
  );
}

function HostCPU({ active, selected }: { active: boolean; selected: boolean }) {
  const select = useSimulationStore((state) => state.selectComponent);

  return (
    <group position={[0, 0.18, 0]}>
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          select("cpu");
        }}
      >
        <cylinderGeometry args={[1.08, 1.08, 0.38, 8]} />
        <meshStandardMaterial
          color={active ? "#3b3631" : surface}
          emissive={active ? "#4d2411" : "#000000"}
          emissiveIntensity={active ? 0.3 : 0}
          roughness={0.7}
          metalness={0.25}
        />
      </mesh>
      <mesh position={[0, 0.24, 0]} rotation={[0, Math.PI / 8, 0]}>
        <cylinderGeometry args={[0.63, 0.63, 0.08, 8]} />
        <meshStandardMaterial color={selected ? accent : surfaceTop} />
      </mesh>
      <Html position={[0, 0.85, 0]} center distanceFactor={8}>
        <button
          type="button"
          className={selected ? "scene-label is-selected" : "scene-label"}
          onClick={() => select("cpu")}
        >
          <strong>Host CPU</strong>
          <span>4 rank processes</span>
        </button>
      </Html>
    </group>
  );
}

function RingLinks({ active }: { active: boolean }) {
  return (
    <group>
      {gpuPositions.map((point, index) => {
        const next = gpuPositions[(index + 1) % gpuPositions.length];
        return (
          <Line
            key={index}
            points={[
              [point[0], 0.02, point[2]],
              [next[0], 0.02, next[2]],
            ]}
            color={active ? accent : quiet}
            lineWidth={active ? 2.2 : 1}
            transparent
            opacity={active ? 0.95 : 0.5}
          />
        );
      })}
    </group>
  );
}

function Packet({
  edge,
  event,
  reducedMotion,
}: {
  edge: number;
  event: SimulationEvent;
  reducedMotion: boolean;
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const from = gpuPositions[edge];
  const to = gpuPositions[(edge + 1) % gpuPositions.length];

  useFrame(() => {
    if (!mesh.current) return;
    const time = useSimulationStore.getState().currentTime;
    const raw = (time - event.start) / event.duration;
    const offset = reducedMotion ? 0.55 : edge * 0.08;
    const progress = reducedMotion
      ? 0.55
      : Math.max(0, Math.min(1, raw * 1.25 - offset));
    mesh.current.position.set(
      THREE.MathUtils.lerp(from[0], to[0], progress),
      0.72 + Math.sin(progress * Math.PI) * 0.48,
      THREE.MathUtils.lerp(from[2], to[2], progress),
    );
    mesh.current.rotation.y += reducedMotion ? 0 : 0.015;
  });

  return (
    <mesh ref={mesh}>
      <boxGeometry args={[0.34, 0.34, 0.34]} />
      <meshStandardMaterial color={accent} roughness={0.34} metalness={0.3} />
    </mesh>
  );
}

function RingPackets({
  event,
  reducedMotion,
}: {
  event: SimulationEvent;
  reducedMotion: boolean;
}) {
  return (
    <group>
      {gpuPositions.map((_, edge) => (
        <Packet
          edge={edge}
          event={event}
          reducedMotion={reducedMotion}
          key={`${event.id}-${edge}`}
        />
      ))}
    </group>
  );
}

function ComputeTiles({ reducedMotion }: { reducedMotion: boolean }) {
  const group = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    if (!group.current || reducedMotion) return;
    group.current.children.forEach((child, index) => {
      child.position.y = 0.82 + Math.sin(clock.elapsedTime * 4 + index) * 0.12;
    });
  });

  return (
    <group ref={group}>
      {gpuPositions.map((position, index) => (
        <mesh position={[position[0], 0.82, position[2]]} key={index}>
          <boxGeometry args={[0.5, 0.12, 0.5]} />
          <meshStandardMaterial color={accent} roughness={0.45} />
        </mesh>
      ))}
    </group>
  );
}

function Topology({
  event,
  reducedMotion,
}: {
  event: SimulationEvent;
  reducedMotion: boolean;
}) {
  const selected = useSimulationStore((state) => state.selectedComponent);
  const select = useSimulationStore((state) => state.selectComponent);
  const communication =
    event.kind === "reduce-scatter" || event.kind === "all-gather";
  const gpuWork = ["compute", "write", "ready", "synchronize", "update"].includes(
    event.kind,
  );
  const cpuWork = event.kind === "launch";

  return (
    <>
      <ambientLight intensity={1.25} />
      <directionalLight position={[5, 9, 5]} intensity={2.2} />
      <directionalLight position={[-4, 4, -4]} intensity={0.7} />

      <mesh position={[0, -0.25, 0]}>
        <boxGeometry args={[9.7, 0.22, 6.9]} />
        <meshStandardMaterial color="#181c1e" roughness={0.85} metalness={0.12} />
      </mesh>

      <RingLinks active={communication} />
      {communication && (
        <RingPackets event={event} reducedMotion={reducedMotion} />
      )}
      {event.kind === "compute" && <ComputeTiles reducedMotion={reducedMotion} />}

      {gpuPositions.map((position, rank) => (
        <DeviceNode
          position={position}
          rank={rank}
          active={gpuWork || communication}
          selected={selected === `gpu-${rank}`}
          reducedMotion={reducedMotion}
          key={rank}
        />
      ))}
      <HostCPU active={cpuWork} selected={selected === "cpu"} />

      <mesh
        position={[0, 0.02, 3.02]}
        onClick={(pointerEvent) => {
          pointerEvent.stopPropagation();
          select("nvlink");
        }}
      >
        <boxGeometry args={[2.1, 0.08, 0.08]} />
        <meshStandardMaterial color={communication ? accent : quiet} />
      </mesh>

      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={7.2}
        maxDistance={13}
        minPolarAngle={0.58}
        maxPolarAngle={1.23}
        target={[0, 0.25, 0]}
      />
    </>
  );
}

export function SceneCanvas({
  event,
  reducedMotion,
}: {
  event: SimulationEvent;
  reducedMotion: boolean;
}) {
  const camera = useMemo(
    () => ({ position: [7.8, 7.1, 8.7] as Point, fov: 40 }),
    [],
  );

  return (
    <Canvas
      camera={camera}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onPointerMissed={() => useSimulationStore.getState().selectComponent("gpu-0")}
    >
      <Topology event={event} reducedMotion={reducedMotion} />
    </Canvas>
  );
}
