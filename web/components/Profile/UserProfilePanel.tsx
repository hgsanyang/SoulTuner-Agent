'use client';

import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { theme } from '@/styles/theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

// ---- 预设标签 ----
const PRESET_GENRES = [
  '摇滚', '电子', '爵士', '嘻哈', '民谣', '古典', 'R&B', '后摇',
  '金属', '流行', '独立', '氛围', 'Lo-fi', '朋克', 'Soul',
];
const PRESET_MOODS = ['开心', '悲伤', '放松', '热血', '浪漫', '治愈', '怀旧', '深沉'];
const PRESET_SCENARIOS = ['学习', '跑步', '开车', '睡前', '派对', '冥想'];
const PRESET_LANGUAGES = ['中文', '英文', '日语', '韩语', '纯音乐'];

const LEARNED_PREF_FIELDS = [
  { key: 'avoid_genres', label: '避开流派' },
  { key: 'avoid_moods', label: '避开情绪' },
  { key: 'avoid_scenarios', label: '避开场景' },
  { key: 'add_moods', label: '偏好情绪' },
  { key: 'add_scenarios', label: '偏好场景' },
  { key: 'activity_contexts', label: '探索倾向' },
];

interface UserProfilePanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface MemoryProfile {
  user_id: string;
  episodic_backends?: string[];
  profile?: Record<string, any>;
  diagnostics?: {
    positive_preference_count?: number;
    negative_preference_count?: number;
    context_preference_count?: number;
  };
  editable_sections?: Array<{
    field: string;
    label: string;
    tone?: string;
    values: string[];
    count?: number;
    deletable?: boolean;
  }>;
}

export default function UserProfilePanel({ isOpen, onClose }: UserProfilePanelProps) {
  const [genres, setGenres] = useState<string[]>([]);
  const [moods, setMoods] = useState<string[]>([]);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);
  const [freeText, setFreeText] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [dirty, setDirty] = useState(false);
  const [memory, setMemory] = useState<MemoryProfile | null>(null);

  // ---- 加载偏好 ----
  const loadProfile = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/user-profile`);
      if (res.ok) {
        const data = await res.json();
        if (data.success && data.profile) {
          setGenres(data.profile.preferred_genres || []);
          setMoods(data.profile.preferred_moods || []);
          setScenarios(data.profile.preferred_scenarios || []);
          setLanguages(data.profile.preferred_languages || []);
          setFreeText(data.profile.free_text || '');
        }
      }
      const memoryRes = await fetch(`${API_URL}/api/memory/profile`);
      if (memoryRes.ok) {
        const memoryData = await memoryRes.json();
        if (memoryData.success && memoryData.memory) {
          setMemory(memoryData.memory);
        }
      }
    } catch (e) {
      console.error('Failed to load profile:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      setDirty(false);
      setMessage('');
      loadProfile();
    }
  }, [isOpen, loadProfile]);

  // ---- 标签切换 ----
  const toggleTag = (
    list: string[],
    setList: React.Dispatch<React.SetStateAction<string[]>>,
    tag: string,
  ) => {
    setList(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]);
    setDirty(true);
  };

  // ---- 保存 ----
  const saveProfile = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${API_URL}/api/user-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preferred_genres: genres,
          preferred_moods: moods,
          preferred_scenarios: scenarios,
          preferred_languages: languages,
          free_text: freeText,
        }),
      });
      const result = await res.json();
      if (result.success) {
        setDirty(false);
        setMessage('✅ 偏好已保存');
        setTimeout(() => setMessage(''), 3000);
      } else {
        setMessage('❌ 保存失败');
        setTimeout(() => setMessage(''), 3000);
      }
    } catch {
      setMessage('❌ 连接失败，请确认后端已启动');
      setTimeout(() => setMessage(''), 3000);
    } finally {
      setSaving(false);
    }
  };

  // ---- 清空 ----
  const resetProfile = () => {
    setGenres([]);
    setMoods([]);
    setScenarios([]);
    setLanguages([]);
    setFreeText('');
    setDirty(true);
  };

  const removeLearnedPreference = async (field: string, value: string) => {
    try {
      const resp = await fetch(
        `${API_URL}/api/memory/preference?field=${encodeURIComponent(field)}&value=${encodeURIComponent(value)}`,
        { method: 'DELETE' },
      );
      if (!resp.ok) throw new Error(`删除失败: ${resp.status}`);
      setMessage('✅ 已删除该条记忆');
      await loadProfile();
      setTimeout(() => setMessage(''), 2500);
    } catch (e) {
      console.error('Failed to remove memory preference:', e);
      setMessage('❌ 删除记忆失败');
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const clearLearnedPreferences = async () => {
    const ok = window.confirm('只清空系统从反馈中学到的偏好，不会删除你手动设置的画像、喜欢或收藏。确定继续吗？');
    if (!ok) return;
    try {
      const resp = await fetch(`${API_URL}/api/memory/profile`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(`清空失败: ${resp.status}`);
      setMessage('✅ 已清空系统学习偏好');
      await loadProfile();
      setTimeout(() => setMessage(''), 2500);
    } catch (e) {
      console.error('Failed to clear learned preferences:', e);
      setMessage('❌ 清空学习记忆失败');
      setTimeout(() => setMessage(''), 3000);
    }
  };

  if (!isOpen) return null;

  // ---- Tag 芯片渲染 ----
  const renderTagGroup = (
    label: string,
    icon: string,
    presets: string[],
    selected: string[],
    setSelected: React.Dispatch<React.SetStateAction<string[]>>,
  ) => (
    <div style={{ marginBottom: '1.4rem' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.4rem',
        marginBottom: '0.6rem',
      }}>
        <span style={{ fontSize: '1rem' }}>{icon}</span>
        <span style={{
          fontSize: '0.82rem', fontWeight: 600,
          color: theme.colors.text.primary,
        }}>{label}</span>
        {selected.length > 0 && (
          <span style={{
            fontSize: '0.7rem', color: theme.colors.primary.accent,
            background: 'rgba(29, 185, 84, 0.15)',
            padding: '0.1rem 0.45rem', borderRadius: '8px',
          }}>
            {selected.length} 项
          </span>
        )}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {presets.map(tag => {
          const isActive = selected.includes(tag);
          return (
            <button
              key={tag}
              onClick={() => toggleTag(selected, setSelected, tag)}
              style={{
                padding: '0.35rem 0.75rem',
                borderRadius: '16px',
                border: isActive
                  ? `1.5px solid ${theme.colors.primary.accent}`
                  : `1px solid ${theme.colors.border.default}`,
                background: isActive
                  ? 'rgba(29, 185, 84, 0.15)'
                  : 'rgba(255, 255, 255, 0.04)',
                color: isActive ? theme.colors.primary.accent : theme.colors.text.secondary,
                fontSize: '0.78rem',
                cursor: 'pointer',
                transition: 'all 0.2s',
                fontWeight: isActive ? 600 : 400,
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  (e.target as HTMLElement).style.borderColor = theme.colors.primary.accent + '60';
                  (e.target as HTMLElement).style.background = 'rgba(29, 185, 84, 0.08)';
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  (e.target as HTMLElement).style.borderColor = theme.colors.border.default;
                  (e.target as HTMLElement).style.background = 'rgba(255, 255, 255, 0.04)';
                }
              }}
            >
              {tag}
            </button>
          );
        })}
      </div>
    </div>
  );

  const totalSelected = genres.length + moods.length + scenarios.length + languages.length;
  const learnedPrefs = memory?.profile || {};
  const editableSections = memory?.editable_sections?.length
    ? memory.editable_sections
    : LEARNED_PREF_FIELDS.map(field => ({
      field: field.key,
      label: field.label,
      values: Array.isArray(learnedPrefs[field.key]) ? learnedPrefs[field.key] : [],
      deletable: true,
    }));
  const hasLearnedPrefs = editableSections.some(section => section.values.length > 0);

  return createPortal(
    <>
      {/* 遮罩 */}
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        zIndex: 9999, backdropFilter: 'blur(4px)',
      }} />

      {/* 面板 */}
      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        width: '560px', maxWidth: '90vw', maxHeight: '85vh',
        background: theme.colors.background.card,
        border: `1px solid ${theme.colors.border.default}`,
        borderRadius: theme.borderRadius.lg,
        boxShadow: theme.shadows.lg,
        zIndex: 10000, display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* 头部 */}
        <div style={{
          padding: '1.2rem 1.5rem', borderBottom: `1px solid ${theme.colors.border.default}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div>
            <h3 style={{ margin: 0, color: theme.colors.text.primary, fontSize: '1.1rem' }}>
              🎭 我的音乐画像
            </h3>
            <span style={{ fontSize: '0.75rem', color: theme.colors.text.muted }}>
              设置偏好后，推荐系统会更懂你的口味
            </span>
          </div>
          <button onClick={onClose} style={{
            background: 'transparent', border: 'none', color: theme.colors.text.muted,
            fontSize: '1.2rem', cursor: 'pointer', padding: '0.3rem',
          }}>✕</button>
        </div>

        {/* 主体 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.2rem 1.5rem' }}>
          {loading ? (
            <div style={{ textAlign: 'center', color: theme.colors.text.muted, padding: '3rem' }}>
              加载中...
            </div>
          ) : (
            <>
              {renderTagGroup('流派偏好', '🎸', PRESET_GENRES, genres, setGenres)}
              {renderTagGroup('情绪倾向', '💭', PRESET_MOODS, moods, setMoods)}
              {renderTagGroup('常听场景', '🎧', PRESET_SCENARIOS, scenarios, setScenarios)}
              {renderTagGroup('语言偏好', '🌍', PRESET_LANGUAGES, languages, setLanguages)}

              {/* 自由描述 */}
              <div style={{ marginBottom: '1rem' }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                  marginBottom: '0.6rem',
                }}>
                  <span style={{ fontSize: '1rem' }}>✍️</span>
                  <span style={{
                    fontSize: '0.82rem', fontWeight: 600,
                    color: theme.colors.text.primary,
                  }}>其他偏好描述</span>
                </div>
                <textarea
                  value={freeText}
                  onChange={e => { setFreeText(e.target.value); setDirty(true); }}
                  placeholder="例如：喜欢 C418 的 Minecraft 原声带风格、偏爱暗黑氛围电子、不太喜欢韩国流行..."
                  rows={3}
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.8rem',
                    background: theme.colors.background.card,
                    border: `1px solid ${theme.colors.border.default}`,
                    borderRadius: theme.borderRadius.sm,
                    color: theme.colors.text.primary,
                    fontSize: '0.82rem',
                    lineHeight: '1.5',
                    outline: 'none',
                    resize: 'vertical',
                    fontFamily: 'inherit',
                    transition: 'border-color 0.2s',
                  }}
                  onFocus={e => { e.target.style.borderColor = theme.colors.primary.accent; }}
                  onBlur={e => { e.target.style.borderColor = theme.colors.border.default; }}
                />
              </div>

              <div style={{ marginBottom: '1.2rem' }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                  marginBottom: '0.6rem',
                }}>
                  <span style={{ fontSize: '1rem' }}>🧠</span>
                  <span style={{
                    fontSize: '0.82rem', fontWeight: 600,
                    color: theme.colors.text.primary,
                  }}>系统学到的偏好</span>
                  {(memory?.episodic_backends || []).map(name => (
                    <span key={name} style={{
                      fontSize: '0.68rem',
                      color: theme.colors.text.muted,
                      background: 'rgba(255,255,255,0.06)',
                      border: `1px solid ${theme.colors.border.default}`,
                      borderRadius: '999px',
                      padding: '0.1rem 0.45rem',
                    }}>{name}</span>
                  ))}
                  {hasLearnedPrefs && (
                    <button
                      onClick={clearLearnedPreferences}
                      style={{
                        marginLeft: 'auto',
                        border: '1px solid rgba(240,96,96,0.35)',
                        background: 'rgba(240,96,96,0.08)',
                        color: '#fca5a5',
                        borderRadius: '999px',
                        padding: '0.16rem 0.55rem',
                        fontSize: '0.7rem',
                        cursor: 'pointer',
                      }}
                    >
                      清空学习记忆
                    </button>
                  )}
                </div>
                <div style={{
                  display: 'grid',
                  gap: '0.55rem',
                  background: 'rgba(255,255,255,0.035)',
                  border: `1px solid ${theme.colors.border.default}`,
                  borderRadius: theme.borderRadius.sm,
                  padding: '0.75rem',
                }}>
                  {!!memory?.diagnostics && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', marginBottom: '0.1rem' }}>
                      <span style={{ fontSize: '0.68rem', color: '#86efac', background: 'rgba(34,197,94,0.10)', borderRadius: '999px', padding: '0.12rem 0.45rem' }}>
                        正向 {memory.diagnostics.positive_preference_count || 0}
                      </span>
                      <span style={{ fontSize: '0.68rem', color: '#fca5a5', background: 'rgba(248,113,113,0.10)', borderRadius: '999px', padding: '0.12rem 0.45rem' }}>
                        避开 {memory.diagnostics.negative_preference_count || 0}
                      </span>
                      <span style={{ fontSize: '0.68rem', color: '#93c5fd', background: 'rgba(59,130,246,0.10)', borderRadius: '999px', padding: '0.12rem 0.45rem' }}>
                        场景 {memory.diagnostics.context_preference_count || 0}
                      </span>
                    </div>
                  )}
                  {editableSections.map(section => {
                    const values = section.values || [];
                    if (!values.length) return null;
                    return (
                      <div key={section.field}>
                        <div style={{
                          fontSize: '0.72rem',
                          color: theme.colors.text.muted,
                          marginBottom: '0.35rem',
                        }}>{section.label}</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                          {values.map((value: string) => (
                            <button
                              key={`${section.field}-${value}`}
                              onClick={() => section.deletable !== false && removeLearnedPreference(section.field, value)}
                              title="点击删除这条学习记忆"
                              style={{
                                border: '1px solid rgba(29,185,84,0.35)',
                                background: 'rgba(29,185,84,0.10)',
                                color: theme.colors.text.primary,
                                borderRadius: '999px',
                                padding: '0.26rem 0.58rem',
                                fontSize: '0.72rem',
                                cursor: section.deletable === false ? 'default' : 'pointer',
                              }}
                            >
                              {value}{section.deletable === false ? '' : ' ×'}
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                  {!hasLearnedPrefs && (
                    <span style={{ color: theme.colors.text.muted, fontSize: '0.76rem' }}>
                      暂无可编辑的学习偏好。点赞、拉黑、歌单反馈后会逐步生成。
                    </span>
                  )}
                </div>
              </div>
            </>
          )}
        </div>

        {/* 底部 */}
        <div style={{
          padding: '0.8rem 1.5rem', borderTop: `1px solid ${theme.colors.border.default}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span style={{
            fontSize: '0.78rem',
            color: message
              ? (message.includes('✅') ? theme.colors.primary.accent : '#f06060')
              : dirty
                ? '#f0a040'
                : theme.colors.text.muted,
          }}>
            {message || (dirty
              ? `已选 ${totalSelected} 项偏好，点击保存`
              : totalSelected > 0
                ? `已设置 ${totalSelected} 项偏好`
                : '暂未设置偏好'
            )}
          </span>
          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <button onClick={resetProfile} style={{
              padding: '0.5rem 1rem', background: 'transparent',
              border: `1px solid ${theme.colors.border.default}`,
              borderRadius: theme.borderRadius.sm, color: '#f06060',
              cursor: 'pointer', fontSize: '0.78rem',
            }}>
              ↩ 清空
            </button>
            <button onClick={onClose} style={{
              padding: '0.5rem 1.2rem', background: 'transparent',
              border: `1px solid ${theme.colors.border.default}`,
              borderRadius: theme.borderRadius.sm, color: theme.colors.text.secondary,
              cursor: 'pointer', fontSize: '0.82rem',
            }}>
              关闭
            </button>
            <button
              onClick={saveProfile}
              disabled={!dirty || saving}
              style={{
                padding: '0.5rem 1.5rem',
                background: dirty ? theme.colors.primary.accent : theme.colors.primary[400],
                border: 'none', borderRadius: theme.borderRadius.sm,
                color: dirty ? '#000' : theme.colors.text.muted,
                cursor: dirty ? 'pointer' : 'default',
                fontWeight: 600, fontSize: '0.82rem',
                transition: 'all 0.2s',
              }}
            >
              {saving ? '保存中...' : '💾 保存偏好'}
            </button>
          </div>
        </div>
      </div>
    </>,
    document.body,
  );
}
