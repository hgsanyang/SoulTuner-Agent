'use client';

import { ReactNode, useState } from 'react';
import NavItem from './NavItem';
import SettingsPanel from '@/components/Settings/SettingsPanel';
import UserProfilePanel from '@/components/Profile/UserProfilePanel';
import { theme } from '@/styles/theme';

interface NavItemConfig {
  href: string;
  label: string;
  description: string;
  icon: ReactNode;
}

interface NavGroup {
  title: string;
  subtitle: string;
  items: NavItemConfig[];
}

interface SidebarProps {
  isMobile?: boolean;
  isOpen?: boolean;
  onClose?: () => void;
}

const navGroups: NavGroup[] = [
  {
    title: '核心流程',
    subtitle: '从入门到推荐',
    items: [
      {
        href: '/',
        label: '发现音乐',
        description: '你的音乐主界面',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M3 9.5L12 3l9 6.5" />
            <path d="M5 11v9h14v-9" />
          </svg>
        ),
      },
      {
        href: '/recommendations',
        label: '智能推荐',
        description: 'AI 交互探索',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
        ),
      },
    ],
  },
  {
    title: '音乐库',
    subtitle: '你的专属收藏',
    items: [
      {
        href: '/library/likes',
        label: '我的喜欢',
        description: '已赞歌曲',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
          </svg>
        ),
      },
      {
        href: '/library/collections',
        label: '我的收藏夹',
        description: '管理自建歌单',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            <line x1="12" y1="11" x2="12" y2="17" />
            <line x1="9" y1="14" x2="15" y2="14" />
          </svg>
        ),
      },
      {
        href: '/library/dislikes',
        label: '不喜欢管理',
        description: '查看已屏蔽歌曲',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="12" cy="12" r="10" />
            <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
          </svg>
        ),
      },
    ],
  },
  {
    title: '创作工具',
    subtitle: '深度定制与编排',
    items: [
      {
        href: '/playlist',
        label: '风格编排器',
        description: '定制私人口味',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="4" width="18" height="4" rx="1" />
            <rect x="3" y="12" width="13" height="4" rx="1" />
            <circle cx="18" cy="14" r="2" />
          </svg>
        ),
      },
      {
        href: '/journey',
        label: '音乐旅程',
        description: '听歌轨迹生成',
        icon: (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M3 17l6-6 4 4 8-8" />
            <path d="M14 7h7v7" />
          </svg>
        ),
      },
    ],
  },
];

export default function Sidebar({ isMobile = false, isOpen = true, onClose }: SidebarProps) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  if (isMobile && !isOpen) {
    return null;
  }

  const commonStyles = {
    height: '100vh',
    background: 'rgba(10, 10, 10, 0.65)', // Semi-transparent to let stars show
    backdropFilter: 'blur(16px)',
    borderRight: `1px solid ${theme.colors.border.default}`,
    padding: '2rem 1.25rem',
    overflowY: 'auto' as const,
    zIndex: 10,
    color: '#b3b3b3',
  };

  const asideStyles = isMobile
    ? {
      ...commonStyles,
      position: 'fixed' as const,
      left: 0,
      top: 0,
      width: '80vw',
      maxWidth: '320px',
      transform: isOpen ? 'translateX(0)' : 'translateX(-100%)',
      transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      borderRadius: '0 16px 16px 0',
    }
    : {
      ...commonStyles,
      position: 'fixed' as const,
      left: 0,
      top: 0,
      width: `${theme.layout.sidebarWidth}px`,
    };

  return (
    <aside style={asideStyles}>
      {isMobile && (
        <button
          type="button"
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            border: 'none',
            background: 'transparent',
            fontSize: '1rem',
            color: theme.colors.text.muted,
            cursor: 'pointer',
          }}
        >
          关闭
        </button>
      )}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          marginBottom: '2rem',
        }}
      >
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.85rem',
            paddingLeft: '0.5rem',
          }}
        >
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: theme.borderRadius.full,
              background: theme.colors.primary.accent,
              color: '#000',
              fontWeight: 800,
              fontSize: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: `0 4px 12px rgba(29, 185, 84, 0.4)`,
            }}
          >
            MA
          </div>
          <div>
            <p style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: theme.colors.text.primary, letterSpacing: '-0.02em' }}>Music Agent</p>
            <span style={{ fontSize: '0.75rem', color: theme.colors.text.muted }}>AI Powered Studio</span>
          </div>
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {navGroups.map((group) => (
          <div key={group.title} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ paddingLeft: '0.5rem', marginBottom: '0.2rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>
                {group.title}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
              {group.items.map((item) => (
                <NavItem key={item.href} {...item} />
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* ──── 底部按钮区 ──── */}
      <div style={{ marginTop: 'auto', paddingTop: '1.5rem', borderTop: `1px solid ${theme.colors.border.default}`, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {/* 🎭 用户画像按钮 */}
        <button
          onClick={() => setProfileOpen(true)}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            padding: '0.7rem 0.8rem',
            background: 'transparent',
            border: `1px solid ${theme.colors.border.default}`,
            borderRadius: theme.borderRadius.sm,
            color: theme.colors.text.secondary,
            cursor: 'pointer',
            fontSize: '0.82rem',
            transition: 'all 0.2s',
          }}
          onMouseEnter={e => {
            (e.target as HTMLElement).style.borderColor = theme.colors.primary.accent;
            (e.target as HTMLElement).style.color = theme.colors.text.primary;
          }}
          onMouseLeave={e => {
            (e.target as HTMLElement).style.borderColor = theme.colors.border.default;
            (e.target as HTMLElement).style.color = theme.colors.text.secondary;
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
          我的画像
        </button>

        {/* ⚙️ 设置按钮 */}
        <button
          onClick={() => setSettingsOpen(true)}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            padding: '0.7rem 0.8rem',
            background: 'transparent',
            border: `1px solid ${theme.colors.border.default}`,
            borderRadius: theme.borderRadius.sm,
            color: theme.colors.text.secondary,
            cursor: 'pointer',
            fontSize: '0.82rem',
            transition: 'all 0.2s',
          }}
          onMouseEnter={e => {
            (e.target as HTMLElement).style.borderColor = theme.colors.primary.accent;
            (e.target as HTMLElement).style.color = theme.colors.text.primary;
          }}
          onMouseLeave={e => {
            (e.target as HTMLElement).style.borderColor = theme.colors.border.default;
            (e.target as HTMLElement).style.color = theme.colors.text.secondary;
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
          系统设置
        </button>
      </div>

      {/* 模态面板 */}
      <UserProfilePanel isOpen={profileOpen} onClose={() => setProfileOpen(false)} />
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </aside>
  );
}
