'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { theme } from '@/styles/theme';

interface MoodPoint {
  id: number;
  time: number;   // 0-1
  mood: string;
  intensity: number; // 0-1
}

interface MoodCurveChartProps {
  points: MoodPoint[];
  onUpdatePoint: (id: number, field: 'time' | 'intensity', value: number) => void;
}

const MOOD_COLORS: Record<string, string> = {
  '放松': '#10b981',
  '专注': '#3b82f6',
  '活力': '#f59e0b',
  '平静': '#a78bfa',
  '浪漫': '#ec4899',
  '疗愈': '#14b8a6',
  '开心': '#f97316',
  '悲伤': '#6366f1',
};

const CHART_PADDING = { top: 28, right: 20, bottom: 32, left: 42 };

export default function MoodCurveChart({ points, onUpdatePoint }: MoodCurveChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [svgRect, setSvgRect] = useState<DOMRect | null>(null);
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  const WIDTH = 500;
  const HEIGHT = 220;
  const plotW = WIDTH - CHART_PADDING.left - CHART_PADDING.right;
  const plotH = HEIGHT - CHART_PADDING.top - CHART_PADDING.bottom;

  const toSVG = useCallback(
    (time: number, intensity: number) => ({
      x: CHART_PADDING.left + time * plotW,
      y: CHART_PADDING.top + (1 - intensity) * plotH,
    }),
    [plotW, plotH]
  );

  const fromSVG = useCallback(
    (clientX: number, clientY: number) => {
      if (!svgRect) return { time: 0, intensity: 0 };
      const x = clientX - svgRect.left;
      const y = clientY - svgRect.top;
      const time = Math.max(0, Math.min(1, (x - CHART_PADDING.left) / plotW));
      const intensity = Math.max(0, Math.min(1, 1 - (y - CHART_PADDING.top) / plotH));
      return { time: Math.round(time * 100) / 100, intensity: Math.round(intensity * 20) / 20 };
    },
    [svgRect, plotW, plotH]
  );

  const sorted = [...points].sort((a, b) => a.time - b.time);

  // Build smooth bezier path
  const buildPath = () => {
    if (sorted.length < 2) return '';
    const pts = sorted.map((p) => toSVG(p.time, p.intensity));
    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const curr = pts[i];
      const cpx = (prev.x + curr.x) / 2;
      d += ` C ${cpx} ${prev.y}, ${cpx} ${curr.y}, ${curr.x} ${curr.y}`;
    }
    return d;
  };

  // Build area fill path
  const buildAreaPath = () => {
    const linePath = buildPath();
    if (!linePath) return '';
    const lastPt = toSVG(sorted[sorted.length - 1].time, sorted[sorted.length - 1].intensity);
    const firstPt = toSVG(sorted[0].time, sorted[0].intensity);
    const bottom = CHART_PADDING.top + plotH;
    return `${linePath} L ${lastPt.x} ${bottom} L ${firstPt.x} ${bottom} Z`;
  };

  const handlePointerDown = (id: number) => {
    if (svgRef.current) {
      setSvgRect(svgRef.current.getBoundingClientRect());
    }
    setDragging(id);
  };

  useEffect(() => {
    if (dragging === null) return;

    const handleMove = (e: PointerEvent) => {
      const { time, intensity } = fromSVG(e.clientX, e.clientY);
      onUpdatePoint(dragging, 'time', time);
      onUpdatePoint(dragging, 'intensity', intensity);
    };

    const handleUp = () => setDragging(null);

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };
  }, [dragging, fromSVG, onUpdatePoint]);

  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div
      style={{
        borderRadius: theme.borderRadius.lg,
        border: `1px solid ${theme.colors.border.default}`,
        backgroundColor: 'rgba(0,0,0,0.3)',
        padding: '0.5rem',
        userSelect: 'none',
      }}
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ width: '100%', height: 'auto', touchAction: 'none' }}
      >
        <defs>
          <linearGradient id="curveGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1db954" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#1db954" stopOpacity={0.02} />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {gridLines.map((v) => {
          const y = CHART_PADDING.top + (1 - v) * plotH;
          return (
            <g key={`grid-${v}`}>
              <line
                x1={CHART_PADDING.left}
                x2={WIDTH - CHART_PADDING.right}
                y1={y}
                y2={y}
                stroke="rgba(255,255,255,0.06)"
                strokeWidth={1}
              />
              <text
                x={CHART_PADDING.left - 6}
                y={y + 3}
                textAnchor="end"
                fill="rgba(255,255,255,0.3)"
                fontSize={9}
              >
                {v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Time axis labels */}
        {[0, 25, 50, 75, 100].map((pct) => {
          const x = CHART_PADDING.left + (pct / 100) * plotW;
          return (
            <text
              key={`t-${pct}`}
              x={x}
              y={HEIGHT - 8}
              textAnchor="middle"
              fill="rgba(255,255,255,0.3)"
              fontSize={9}
            >
              {pct}%
            </text>
          );
        })}

        {/* Area fill */}
        {sorted.length >= 2 && (
          <path d={buildAreaPath()} fill="url(#curveGradient)" />
        )}

        {/* Curve line */}
        {sorted.length >= 2 && (
          <path
            d={buildPath()}
            fill="none"
            stroke="#1db954"
            strokeWidth={2.5}
            strokeLinecap="round"
            opacity={0.85}
          />
        )}

        {/* Points */}
        {sorted.map((point) => {
          const { x, y } = toSVG(point.time, point.intensity);
          const color = MOOD_COLORS[point.mood] || '#1db954';
          const isHovered = hoveredId === point.id;
          const isDragging = dragging === point.id;

          return (
            <g key={point.id}>
              {/* Glow */}
              <circle
                cx={x}
                cy={y}
                r={isHovered || isDragging ? 16 : 10}
                fill={color}
                opacity={0.15}
                style={{ transition: 'r 0.15s' }}
              />
              {/* Node */}
              <circle
                cx={x}
                cy={y}
                r={isHovered || isDragging ? 7 : 5.5}
                fill={color}
                stroke="#fff"
                strokeWidth={2}
                style={{ cursor: 'grab', transition: 'r 0.15s' }}
                onPointerDown={() => handlePointerDown(point.id)}
                onPointerEnter={() => setHoveredId(point.id)}
                onPointerLeave={() => setHoveredId(null)}
              />
              {/* Label */}
              <text
                x={x}
                y={y - 12}
                textAnchor="middle"
                fill={color}
                fontSize={10}
                fontWeight={600}
                style={{ pointerEvents: 'none' }}
              >
                {point.mood}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
