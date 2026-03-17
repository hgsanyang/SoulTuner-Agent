'use client';

import { theme } from '@/styles/theme';

interface MusicShareCardProps {
  title: string;
  artist: string;
  mood?: string;
  segmentLabel?: string;
  headline?: string;
  subline?: string;
  hashtags?: string[];
}

export default function MusicShareCard({
  title,
  artist,
  mood,
  segmentLabel,
  headline,
  subline,
  hashtags,
}: MusicShareCardProps) {
  return (
    <div
      style={{
        width: 360,
        height: 540,
        borderRadius: 32,
        padding: '1.6rem 1.6rem 1.4rem',
        background:
          'radial-gradient(circle at 10% 0%, rgba(255,255,255,0.18), transparent 55%), linear-gradient(145deg, #111827 0%, #020617 45%, #0f172a 100%)',
        color: '#f9fafb',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        boxShadow: '0 24px 60px rgba(15,23,42,0.65)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div
          style={{
            padding: '0.25rem 0.8rem',
            borderRadius: 999,
            border: '1px solid rgba(148,163,184,0.4)',
            fontSize: '0.78rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background:
                'radial-gradient(circle, #22c55e 0%, #4ade80 40%, rgba(74,222,128,0.1) 100%)',
              boxShadow: '0 0 10px rgba(34,197,94,0.8)',
            }}
          />
          音乐旅程卡片
        </div>
        {segmentLabel && (
          <span
            style={{
              fontSize: '0.78rem',
              color: 'rgba(226,232,240,0.8)',
            }}
          >
            {segmentLabel}
          </span>
        )}
      </div>

      {/* Middle artwork area */}
      <div
        style={{
          marginTop: '1.6rem',
          borderRadius: 24,
          padding: '1.3rem 1.3rem 1.1rem',
          background:
            'radial-gradient(circle at 0% 0%, rgba(250,250,250,0.18), transparent 60%), linear-gradient(135deg, #334155 0%, #020617 100%)',
          border: '1px solid rgba(148,163,184,0.4)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            inset: '-40%',
            opacity: 0.3,
            background:
              'radial-gradient(circle at 10% 10%, rgba(248,250,252,0.9), transparent 55%), radial-gradient(circle at 90% 80%, rgba(59,130,246,0.85), transparent 50%)',
            mixBlendMode: 'screen',
          }}
        />

        <div
          style={{
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
          }}
        >
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.45rem',
              padding: '0.3rem 0.8rem',
              borderRadius: 999,
              backgroundColor: 'rgba(15,23,42,0.7)',
              border: '1px solid rgba(148,163,184,0.5)',
              fontSize: '0.78rem',
            }}
          >
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                border: '2px solid rgba(148,163,184,0.7)',
                borderTopColor: '#e5e7eb',
              }}
            />
            Deep Search · Journey
          </div>

          <div style={{ marginTop: '0.4rem' }}>
            <h2
              style={{
                margin: 0,
                fontSize: '1.45rem',
                lineHeight: 1.35,
                letterSpacing: '-0.02em',
                color: '#f9fafb',
              }}
            >
              {headline || title}
            </h2>
            <p
              style={{
                margin: '0.4rem 0 0',
                fontSize: '0.9rem',
                color: 'rgba(226,232,240,0.9)',
              }}
            >
              {subline || artist}
            </p>
          </div>

          {mood && (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                marginTop: '0.6rem',
                fontSize: '0.8rem',
                color: 'rgba(226,232,240,0.9)',
              }}
            >
              <span
                style={{
                  padding: '0.2rem 0.7rem',
                  borderRadius: 999,
                  backgroundColor: 'rgba(15,23,42,0.75)',
                  border: '1px solid rgba(148,163,184,0.7)',
                }}
              >
                情绪 · {mood}
              </span>
              <span style={{ opacity: 0.85 }}>为这一章的画面配乐</span>
            </div>
          )}
        </div>
      </div>

      {/* Footer / controls hint */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.75rem',
          marginTop: '1.5rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            color: 'rgba(148,163,184,0.9)',
            fontSize: '0.8rem',
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 999,
              background:
                'radial-gradient(circle, rgba(52,211,153,0.5) 0%, rgba(15,23,42,1) 70%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 18px rgba(52,211,153,0.7)',
            }}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#bbf7d0"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: '0.8rem' }}>截图保存这张卡片</div>
            <div style={{ fontSize: '0.78rem', opacity: 0.85 }}>分享到你的朋友圈 / 群聊</div>
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-end',
            gap: '0.25rem',
          }}
        >
          {hashtags && hashtags.length > 0 && (
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.25rem',
                maxWidth: 180,
                justifyContent: 'flex-end',
              }}
            >
              {hashtags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  style={{
                    fontSize: '0.7rem',
                    color: 'rgba(209,213,219,0.9)',
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
          <span
            style={{
              fontSize: '0.78rem',
              color: 'rgba(148,163,184,0.9)',
            }}
          >
            deep search · music journey
          </span>
        </div>
      </div>
    </div>
  );
}


