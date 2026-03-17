'use client';

/**
 * ✈️ SendButton.tsx (用户对话发送按钮小组件)
 * 作用：音乐大模型对话框右侧孤立的发送激活按钮。
 * 功能细节：
 * 1. 接收从 ChatInput 传递下来的 `disabled` (禁用) 属性来动态改变自身的外观样式和光标悬浮颜色。
 * 2. 我们对其做过强视觉升级：输入框激活后内部 SVG Path (小纸飞机) 会将边框线变为黑色，搭配绿底增强深色模式里的视觉可见度。
 */

import { theme } from '@/styles/theme';

interface SendButtonProps {
  onClick: (e: React.MouseEvent) => void;
  disabled?: boolean;
}

export default function SendButton({ onClick, disabled }: SendButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        width: '44px',
        height: '44px',
        borderRadius: theme.borderRadius.full,
        backgroundColor: disabled ? 'rgba(255,255,255,0.1)' : theme.colors.primary.accent,
        border: 'none',
        cursor: disabled ? 'not-allowed' : 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'transform 0.2s ease, background-color 0.2s ease',
        flexShrink: 0,
        color: disabled ? theme.colors.text.muted : '#000000',
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = 'translateY(-1px)';
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = 'translateY(0)';
        }
      }}
    >
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke={disabled ? "currentColor" : "#000000"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="22" y1="2" x2="11" y2="13" />
        <polygon points="22 2 15 22 11 13 2 9 22 2" />
      </svg>
    </button>
  );
}

