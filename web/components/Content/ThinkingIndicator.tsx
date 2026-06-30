'use client';

import { theme } from '@/styles/theme';

interface ThinkingIndicatorProps {
  message?: string;
}

export default function ThinkingIndicator({ message }: ThinkingIndicatorProps) {
  const steps = ['理解意图', '检索曲库', '对齐听感'];

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '0.85rem 1rem',
        marginBottom: '1rem',
        borderRadius: '0.85rem',
        border: '1px solid rgba(255,255,255,0.08)',
        backgroundColor: 'rgba(18,18,18,0.74)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)',
      }}
    >
      <div
        style={{
          width: '42px',
          height: '28px',
          display: 'grid',
          gridTemplateColumns: 'repeat(7, 1fr)',
          alignItems: 'center',
          gap: '3px',
          flexShrink: 0,
        }}
        aria-hidden="true"
      >
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <span
            key={i}
            style={{
              display: 'block',
              width: '100%',
              height: `${8 + (i % 4) * 4}px`,
              borderRadius: '999px',
              backgroundColor: i === 3 ? theme.colors.primary.accent : 'rgba(29,185,84,0.45)',
              animation: `waveform 1.35s ease-in-out ${i * 0.08}s infinite`,
            }}
          />
        ))}
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
          minWidth: 0,
          flex: 1,
        }}
      >
        <span
          style={{
            color: theme.colors.text.primary,
            fontSize: '0.86rem',
            fontWeight: 600,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {message || '正在理解你的音乐偏好'}
        </span>
        <div
          style={{
            display: 'flex',
            gap: '0.45rem',
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          {steps.map((step, i) => (
            <span
              key={step}
              style={{
                color: 'rgba(255,255,255,0.48)',
                fontSize: '0.72rem',
                animation: `stepFade 2.4s ease-in-out ${i * 0.35}s infinite`,
              }}
            >
              {step}
            </span>
          ))}
        </div>
      </div>

      <div
        style={{
          position: 'relative',
          width: '44px',
          height: '2px',
          borderRadius: '999px',
          overflow: 'hidden',
          backgroundColor: 'rgba(255,255,255,0.08)',
          flexShrink: 0,
        }}
        aria-hidden="true"
      >
        <span
          style={{
            position: 'absolute',
            inset: 0,
            width: '45%',
            borderRadius: '999px',
            backgroundColor: theme.colors.primary.accent,
            animation: 'scan 1.2s ease-in-out infinite',
          }}
        />
      </div>

      <style jsx>{`
        @keyframes waveform {
          0%, 100% {
            opacity: 0.38;
            transform: scaleY(0.58);
          }
          50% {
            opacity: 1;
            transform: scaleY(1.25);
          }
        }
        @keyframes stepFade {
          0%, 100% {
            opacity: 0.35;
          }
          45%, 60% {
            opacity: 0.9;
          }
        }
        @keyframes scan {
          0% {
            transform: translateX(-110%);
          }
          55% {
            transform: translateX(80%);
          }
          100% {
            transform: translateX(230%);
          }
        }
      `}</style>
    </div>
  );
}

