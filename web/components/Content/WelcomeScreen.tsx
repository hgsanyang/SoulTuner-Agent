'use client';

import { theme } from '@/styles/theme';
import { useState, useEffect } from 'react';

function useCurrentTime() {
  const [time, setTime] = useState('');
  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }));
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

interface WelcomeScreenProps {
  title?: string;
  description?: string;
  badgeLabel?: string;
  subtitle?: string;
  onPromptClick?: (text: string) => void;
}

// ── 场景卡片数据 ──
const sceneCards = [
  { emoji: '🌙', title: '深夜独处', desc: '万籁俱寂，只剩月光与心事', prompt: '深夜一个人，有些微醺，想听点慵懒温柔的歌陪着发呆', color: 'rgba(99, 102, 241, 0.15)', border: 'rgba(99, 102, 241, 0.25)' },
  { emoji: '🏃', title: '运动燃脂', desc: '心跳加速，让汗水更有节奏', prompt: '跑步健身中，需要热血沸腾、节奏感强的音乐让我停不下来', color: 'rgba(245, 158, 11, 0.15)', border: 'rgba(245, 158, 11, 0.25)' },
  { emoji: '☕', title: '午后咖啡', desc: '阳光洒落，慵懒的周末时光', prompt: '慵懒的周末下午，阳光洒在阳台上，想听点放松惬意的音乐', color: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.25)' },
  { emoji: '💻', title: '专注编程', desc: '屏幕微光，沉入深度心流', prompt: '屏幕微光，沉入深度心流，专注编程的夜晚', color: 'rgba(34, 211, 238, 0.15)', border: 'rgba(34, 211, 238, 0.25)' },
  { emoji: '🌧️', title: '雨天发呆', desc: '窗外细雨，配一杯热茶', prompt: '窗外淅淅沥沥下着雨，窝在沙发上发呆，想听点安静治愈的', color: 'rgba(107, 114, 128, 0.2)', border: 'rgba(107, 114, 128, 0.3)' },
  { emoji: '🚗', title: '公路旅行', desc: '风驰电掣，自由在路上', prompt: '一个人开车在空旷的公路上，窗外风景飞驰而过', color: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.25)' },
  { emoji: '🎉', title: '派对狂欢', desc: '灯光闪烁，释放所有快乐', prompt: '朋友聚会嗨起来，要那种让人忍不住跟着跳的动感音乐', color: 'rgba(236, 72, 153, 0.15)', border: 'rgba(236, 72, 153, 0.25)' },
  { emoji: '📖', title: '阅读时光', desc: '翻开书页，沉浸在文字的世界', prompt: '安静地看书，想要轻柔舒缓的纯音乐作为背景', color: 'rgba(139, 92, 246, 0.15)', border: 'rgba(139, 92, 246, 0.25)' },
  { emoji: '🌅', title: '清晨觉醒', desc: '第一缕阳光，唤醒美好的一天', prompt: '刚睡醒迎接新的早晨，想听清新温暖、充满希望的音乐', color: 'rgba(251, 191, 36, 0.15)', border: 'rgba(251, 191, 36, 0.25)' },
  { emoji: '🍷', title: '烛光晚餐', desc: '烛光摇曳，共度浪漫夜晚', prompt: '烛光晚餐的浪漫夜晚，想要优雅温馨、有情调的音乐', color: 'rgba(220, 38, 38, 0.15)', border: 'rgba(220, 38, 38, 0.25)' },
  { emoji: '🌌', title: '星空漫游', desc: '银河流转，思绪飘向远方', prompt: '仰望星空，迷幻而浪漫，仿佛漂浮在宇宙之间', color: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.25)' },
  { emoji: '🧘', title: '冥想放松', desc: '呼吸之间，找到内心的宁静', prompt: '闭上眼睛冥想，想听空灵平静、让心慢慢沉下来的音乐', color: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.25)' },
];

// ── 能力标签 ──
const capabilities = [
  { icon: '🎯', label: '情绪感知' },
  { icon: '🌐', label: '联网搜索' },
  { icon: '🎵', label: '智能匹配' },
  { icon: '📊', label: '音频分析' },
];

// ── 时间问候 ──
function getGreeting(): { text: string; emoji: string } {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 9) return { text: '早安，新的一天从音乐开始', emoji: '🌅' };
  if (hour >= 9 && hour < 12) return { text: '上午好，来点提神的旋律', emoji: '☀️' };
  if (hour >= 12 && hour < 14) return { text: '午后好，让音乐陪你放松', emoji: '🌤️' };
  if (hour >= 14 && hour < 18) return { text: '下午好，要不要一杯咖啡摇滚', emoji: '☕' };
  if (hour >= 18 && hour < 21) return { text: '傍晚好，来点陪伴晚餐的音乐', emoji: '🌇' };
  return { text: '夜深了，来点助眠或微醺的旋律', emoji: '🌙' };
}

export default function WelcomeScreen({
  title = '听懂你的每一刻情绪',
  description = '用一段自然语言描述此刻的心境或场景，即刻生成专属的沉浸式歌单。',
  badgeLabel = 'SoulTuner 引擎已就绪',
  subtitle = '探索未知的音乐旅程',
  onPromptClick,
}: WelcomeScreenProps) {
  const greeting = getGreeting();
  const currentTime = useCurrentTime();
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  return (
    <section style={{ width: '100%', padding: '1rem 1rem 2rem', background: 'transparent' }}>
      <div style={{
        width: '100%', maxWidth: `${theme.layout.contentMaxWidth}px`,
        margin: '0 auto', display: 'flex', flexDirection: 'column',
        gap: '1.75rem', alignItems: 'center', textAlign: 'center',
      }}>

        {/* ── 时间问候 ── */}
        <div style={{
          fontSize: '0.9rem', color: theme.colors.text.muted,
          display: 'flex', alignItems: 'center', gap: '0.5rem',
          animation: 'fadeUp 0.6s ease-out',
        }}>
          <span style={{ fontSize: '1.2rem' }}>{greeting.emoji}</span>
          {greeting.text}
          {currentTime && (
            <span style={{
              marginLeft: '0.4rem',
              padding: '0.15rem 0.55rem',
              borderRadius: theme.borderRadius.full,
              backgroundColor: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.1)',
              fontSize: '0.8rem',
              fontFamily: 'monospace',
              color: theme.colors.text.secondary,
              letterSpacing: '0.05em',
            }}>
              {currentTime}
            </span>
          )}
        </div>

        {/* ── Badge ── */}
        <div style={{
          padding: '0.35rem 0.9rem', borderRadius: theme.borderRadius.full,
          backgroundColor: 'rgba(29, 185, 84, 0.1)', border: '1px solid rgba(29, 185, 84, 0.2)',
          color: theme.colors.primary.accent, fontSize: '0.8rem',
          display: 'inline-flex', alignItems: 'center', gap: '0.45rem',
          animation: 'fadeUp 0.6s ease-out 0.1s both',
        }}>
          <span style={{
            width: '7px', height: '7px', borderRadius: '50%',
            backgroundColor: theme.colors.primary.accent,
            boxShadow: `0 0 10px ${theme.colors.primary.accent}`,
          }} />
          {badgeLabel}
        </div>

        {/* ── Hero Text ── */}
        <div style={{ maxWidth: '640px', animation: 'fadeUp 0.6s ease-out 0.2s both' }}>
          <h1 style={{
            margin: 0, fontSize: '2.6rem', lineHeight: 1.2, fontWeight: 800,
            color: theme.colors.text.primary, letterSpacing: '-0.02em',
          }}>
            {title}
          </h1>
          <h2 style={{
            margin: '0.4rem 0 1rem', fontSize: '1.5rem', fontWeight: 700,
            color: theme.colors.primary.accent, letterSpacing: '-0.01em',
          }}>
            {subtitle}
          </h2>
          <p style={{
            margin: '0 auto', fontSize: '1rem', lineHeight: 1.6,
            color: theme.colors.text.secondary, maxWidth: '480px',
          }}>
            {description}
          </p>
        </div>

        {/* ── 能力标签条 ── */}
        <div style={{
          display: 'flex', gap: '0.6rem', flexWrap: 'wrap', justifyContent: 'center',
          animation: 'fadeUp 0.6s ease-out 0.3s both',
        }}>
          {capabilities.map((cap, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '0.35rem',
              padding: '0.35rem 0.75rem', borderRadius: theme.borderRadius.full,
              backgroundColor: 'rgba(255,255,255,0.04)',
              border: `1px solid ${theme.colors.border.default}`,
              fontSize: '0.78rem', color: theme.colors.text.secondary,
            }}>
              <span style={{ fontSize: '0.85rem' }}>{cap.icon}</span>
              {cap.label}
            </div>
          ))}
        </div>



        {/* ── 场景卡片网格 ── */}
        <div style={{ width: '100%', maxWidth: '780px', animation: 'fadeUp 0.6s ease-out 0.35s both' }}>
          <div style={{
            fontSize: '0.85rem', color: theme.colors.text.muted,
            marginBottom: '1rem', letterSpacing: '0.05em',
            display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'center',
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(29,185,84,0.5)" strokeWidth="2" strokeLinecap="round"><path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707" /></svg>
            选择一个场景，快速开始
          </div>

          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '0.75rem',
          }}>
            {sceneCards.map((card, i) => (
              <div
                key={i}
                onClick={() => onPromptClick?.(card.prompt)}
                onMouseEnter={() => setHoveredCard(i)}
                onMouseLeave={() => setHoveredCard(null)}
                style={{
                  padding: '1.1rem 1.2rem',
                  borderRadius: '0.85rem',
                  backgroundColor: hoveredCard === i ? card.color : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${hoveredCard === i ? card.border : 'rgba(255,255,255,0.08)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.25s ease',
                  textAlign: 'left',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.85rem',
                  transform: hoveredCard === i ? 'translateY(-2px)' : 'none',
                  boxShadow: hoveredCard === i ? `0 8px 24px ${card.color}` : 'none',
                }}
              >
                <span style={{
                  fontSize: '1.8rem', lineHeight: 1,
                  flexShrink: 0, marginTop: '0.1rem',
                }}>
                  {card.emoji}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: '0.95rem', fontWeight: 600,
                    color: theme.colors.text.primary,
                    marginBottom: '0.25rem',
                  }}>
                    {card.title}
                  </div>
                  <div style={{
                    fontSize: '0.8rem', color: theme.colors.text.muted,
                    lineHeight: 1.4,
                  }}>
                    {card.desc}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
}
