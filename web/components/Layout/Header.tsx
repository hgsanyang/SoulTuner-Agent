'use client';

import { theme } from '@/styles/theme';

interface HeaderProps {
  onMenuToggle?: () => void;
  isMobile?: boolean;
}

export default function Header({ onMenuToggle, isMobile = false }: HeaderProps) {
  return (
    <header
      style={{
        position: isMobile ? 'relative' : 'sticky',
        top: isMobile ? 0 : '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1.25rem',
        padding: isMobile ? '0.5rem 0.8rem' : '0.6rem 1.25rem',
        borderRadius: theme.borderRadius.full, // Changed to full for pill shape
        backgroundColor: 'rgba(36, 36, 36, 0.4)',
        backdropFilter: 'blur(16px)',
        border: `1px solid ${theme.colors.border.default}`,
        boxShadow: theme.shadows.sm,
        zIndex: 5,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {onMenuToggle && (
          <button
            type="button"
            aria-label="打开导航"
            onClick={onMenuToggle}
            style={{
              width: '40px',
              height: '40px',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              backgroundColor: theme.colors.background.card,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '5px',
              }}
            >
              {[0, 1, 2].map((line) => (
                <span
                  key={line}
                  style={{
                    width: '22px',
                    height: '2px',
                    borderRadius: '9999px',
                    backgroundColor: theme.colors.text.primary,
                  }}
                />
              ))}
            </span>
          </button>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <h1
            style={{
              fontSize: isMobile ? '1.1rem' : '1.25rem',
              color: theme.colors.text.primary,
              margin: 0,
              fontWeight: 600,
              letterSpacing: '-0.01em',
            }}
          >
            SoulTuner Studio
          </h1>
          
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.15rem 0.5rem',
              borderRadius: theme.borderRadius.full,
              backgroundColor: 'rgba(29, 185, 84, 0.15)',
              color: theme.colors.primary.accent,
              fontSize: '0.7rem',
              fontWeight: 600,
            }}
          >
            Beta
          </span>

          {!isMobile && (
            <>
              <span style={{ color: theme.colors.border.default, marginLeft: '0.5rem', marginRight: '0.5rem' }}>|</span>
              <p
                style={{
                  margin: 0,
                  color: theme.colors.text.muted,
                  fontSize: '0.85rem',
                }}
              >
                懂你此刻的情绪 · 沉浸式专属歌单生成器
              </p>
            </>
          )}
        </div>
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          flexWrap: 'wrap',
          justifyContent: isMobile ? 'flex-end' : 'flex-start',
        }}
      >
        <a
          href="/recommendations"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 1rem',
            borderRadius: theme.borderRadius.full,
            backgroundColor: 'rgba(255, 255, 255, 0.08)',
            border: `1px solid rgba(255, 255, 255, 0.1)`,
            color: theme.colors.text.primary,
            fontWeight: 500,
            fontSize: '0.85rem',
            textDecoration: 'none',
            transition: 'background-color 0.2s',
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.15)')}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.08)')}
        >
          立即使用
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M7 17L17 7" />
            <path d="M7 7h10v10" />
          </svg>
        </a>
      </div>
    </header>
  );
}

