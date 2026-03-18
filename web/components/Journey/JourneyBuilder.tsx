'use client';

import { useState } from 'react';
import { JourneyRequest, MoodTransitionInput } from '@/lib/api';
import { theme } from '@/styles/theme';
import MoodCurveChart from './MoodCurveChart';

const DEFAULT_STORY = '早晨起床 → 通勤路上 → 工作中 → 下班放松 → 夜晚休息';

const STORY_PRESETS = [
  {
    icon: '✈️',
    title: '旅途篇',
    story: '机场清晨 → 登机等待 → 起飞穿云 → 落地黄昏 → 入住酒店的小惊喜',
  },
  {
    icon: '☕',
    title: '写作日',
    story: '雨天咖啡馆 → 专注写作 → 灵感爆发 → 走出门口的微风',
  },
  {
    icon: '🏃',
    title: '夜跑',
    story: '城市夜跑 → 河边灯光 → 冲刺冲线 → 回家拉伸与放松',
  },
];

const DEFAULT_MOOD_POINTS = [
  { id: 1, time: 0, mood: '放松', intensity: 0.5 },
  { id: 2, time: 0.35, mood: '专注', intensity: 0.7 },
  { id: 3, time: 0.65, mood: '活力', intensity: 0.9 },
  { id: 4, time: 1, mood: '平静', intensity: 0.4 },
];

const moodOptions = ['放松', '专注', '活力', '平静', '浪漫', '疗愈', '开心', '悲伤'];

interface JourneyBuilderProps {
  loading: boolean;
  onGenerate: (payload: JourneyRequest) => void;
}

interface MoodPointForm {
  id: number;
  time: number;
  mood: string;
  intensity: number;
}

export default function JourneyBuilder({ loading, onGenerate }: JourneyBuilderProps) {
  const [mode, setMode] = useState<'story' | 'mood'>('story');
  const [story, setStory] = useState(DEFAULT_STORY);
  const [duration, setDuration] = useState(60);
  const [moodPoints, setMoodPoints] = useState<MoodPointForm[]>(DEFAULT_MOOD_POINTS);
  const [context, setContext] = useState({ location: '上海', weather: '晴朗', activity: '通勤' });
  const [inlineError, setInlineError] = useState<string | null>(null);

  const handleAddMoodPoint = () => {
    const nextId = moodPoints.length ? Math.max(...moodPoints.map((p) => p.id)) + 1 : 1;
    setMoodPoints([
      ...moodPoints,
      { id: nextId, time: 1, mood: '放松', intensity: 0.5 },
    ]);
  };

  const handleUpdateMoodPoint = (id: number, field: keyof MoodPointForm, value: number | string) => {
    setMoodPoints((prev) =>
      prev.map((point) => (point.id === id ? { ...point, [field]: value } : point))
    );
  };

  const handleDeleteMoodPoint = (id: number) => {
    setMoodPoints((prev) => prev.filter((point) => point.id !== id));
  };

  const handleCurveUpdate = (id: number, field: 'time' | 'intensity', value: number) => {
    handleUpdateMoodPoint(id, field, value);
  };

  const buildRequestPayload = (): JourneyRequest | null => {
    setInlineError(null);
    if (mode === 'story') {
      if (!story.trim()) {
        setInlineError('请先写一点故事，再让我们为它配一段音乐。');
        return null;
      }
      return { story: story.trim(), duration, context };
    }

    const validPoints: MoodTransitionInput[] = moodPoints
      .filter((point) => point.time >= 0 && point.time <= 1)
      .sort((a, b) => a.time - b.time)
      .map((point) => ({
        time: point.time,
        mood: point.mood,
        intensity: Math.max(0, Math.min(1, point.intensity)),
      }));

    if (!validPoints.length) return null;
    return { mood_transitions: validPoints, duration, context };
  };

  const handleGenerate = () => {
    const payload = buildRequestPayload();
    if (!payload) return;
    onGenerate(payload);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '1rem' }}>
      {/* Header + Mode toggle */}
      <div>
        <h2
          style={{
            margin: 0,
            fontSize: '1.35rem',
            color: theme.colors.text.primary,
            letterSpacing: '-0.01em',
          }}
        >
          🗺️ 音乐旅程
        </h2>
        <p style={{ margin: '0.25rem 0 0', color: theme.colors.text.muted, fontSize: '0.85rem' }}>
          用故事或情绪曲线，编排一段沉浸式音乐旅程
        </p>
      </div>

      {/* Mode toggle pills */}
      <div
        style={{
          display: 'flex',
          gap: '0.35rem',
          backgroundColor: 'rgba(255,255,255,0.04)',
          padding: '0.25rem',
          borderRadius: theme.borderRadius.full,
          border: `1px solid ${theme.colors.border.default}`,
        }}
      >
        {[
          { label: '📖 故事驱动', value: 'story' as const },
          { label: '📈 情绪曲线', value: 'mood' as const },
        ].map((item) => (
          <button
            key={item.value}
            onClick={() => setMode(item.value)}
            type="button"
            style={{
              flex: 1,
              border: 'none',
              backgroundColor: mode === item.value ? theme.colors.primary.accent : 'transparent',
              color: mode === item.value ? '#fff' : theme.colors.text.secondary,
              padding: '0.5rem 0.75rem',
              borderRadius: theme.borderRadius.full,
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '0.85rem',
              transition: 'all 0.2s ease',
            }}
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* Story mode */}
      {mode === 'story' ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <label style={{ color: theme.colors.text.secondary, fontWeight: 600, fontSize: '0.85rem' }}>
            故事情节
          </label>
          <textarea
            value={story}
            onChange={(e) => setStory(e.target.value)}
            rows={4}
            placeholder="例如：早晨起床 → 通勤路上 → 工作中 → 下班放松 → 夜晚休息"
            style={{
              width: '100%',
              resize: 'vertical',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              padding: '0.85rem',
              fontSize: '0.92rem',
              lineHeight: 1.6,
              color: theme.colors.text.primary,
              backgroundColor: theme.colors.background.elevated,
              transition: 'border-color 0.2s',
              outline: 'none',
            }}
            onFocus={(e) => (e.target.style.borderColor = theme.colors.primary.accent)}
            onBlur={(e) => (e.target.style.borderColor = theme.colors.border.default)}
          />

          {/* Preset cards */}
          <div>
            <span style={{ fontSize: '0.78rem', color: theme.colors.text.muted }}>
              💡 灵感模板
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', marginTop: '0.4rem' }}>
              {STORY_PRESETS.map((preset) => (
                <button
                  key={preset.title}
                  type="button"
                  onClick={() => setStory(preset.story)}
                  style={{
                    border: `1px solid ${theme.colors.border.default}`,
                    borderRadius: theme.borderRadius.md,
                    padding: '0.6rem 0.5rem',
                    cursor: 'pointer',
                    backgroundColor: theme.colors.background.elevated,
                    color: theme.colors.text.secondary,
                    textAlign: 'left',
                    transition: 'all 0.2s',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.2rem',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = theme.colors.primary.accent;
                    e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.06)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = theme.colors.border.default;
                    e.currentTarget.style.backgroundColor = theme.colors.background.elevated;
                  }}
                >
                  <span style={{ fontSize: '1.1rem' }}>{preset.icon}</span>
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: theme.colors.text.primary }}>
                    {preset.title}
                  </span>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      color: theme.colors.text.muted,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {preset.story.split('→')[0]}→...
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Mood curve mode */
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* SVG curve chart */}
          <MoodCurveChart points={moodPoints} onUpdatePoint={handleCurveUpdate} />

          {/* Mood points list */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ color: theme.colors.text.secondary, fontWeight: 600, fontSize: '0.82rem' }}>
              情绪节点
            </label>
            <button
              type="button"
              onClick={handleAddMoodPoint}
              style={{
                border: `1px solid ${theme.colors.border.default}`,
                backgroundColor: 'transparent',
                color: theme.colors.primary.accent,
                fontWeight: 600,
                cursor: 'pointer',
                fontSize: '0.78rem',
                padding: '0.25rem 0.7rem',
                borderRadius: theme.borderRadius.full,
              }}
            >
              + 添加
            </button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', maxHeight: '200px' }}>
            {moodPoints.map((point) => (
              <div
                key={point.id}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1.2fr 0.8fr auto',
                  gap: '0.5rem',
                  alignItems: 'center',
                  padding: '0.55rem 0.65rem',
                  borderRadius: theme.borderRadius.md,
                  border: `1px solid ${theme.colors.border.default}`,
                  backgroundColor: theme.colors.background.elevated,
                  fontSize: '0.82rem',
                }}
              >
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={Math.round(point.time * 100)}
                  onChange={(e) => handleUpdateMoodPoint(point.id, 'time', Number(e.target.value) / 100)}
                  placeholder="%"
                  style={{
                    width: '100%',
                    padding: '0.35rem',
                    borderRadius: theme.borderRadius.sm,
                    border: `1px solid ${theme.colors.border.default}`,
                    color: theme.colors.text.primary,
                    backgroundColor: theme.colors.background.main,
                    fontSize: '0.82rem',
                  }}
                />
                <select
                  value={point.mood}
                  onChange={(e) => handleUpdateMoodPoint(point.id, 'mood', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.35rem',
                    borderRadius: theme.borderRadius.sm,
                    border: `1px solid ${theme.colors.border.default}`,
                    color: theme.colors.text.primary,
                    backgroundColor: theme.colors.background.main,
                    fontSize: '0.82rem',
                  }}
                >
                  {moodOptions.map((mood) => (
                    <option key={mood} value={mood}>{mood}</option>
                  ))}
                </select>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={point.intensity}
                  onChange={(e) => handleUpdateMoodPoint(point.id, 'intensity', Number(e.target.value))}
                  style={{ width: '100%', accentColor: theme.colors.primary.accent }}
                />
                <button
                  type="button"
                  onClick={() => handleDeleteMoodPoint(point.id)}
                  style={{
                    border: 'none',
                    backgroundColor: 'transparent',
                    color: 'rgba(239,68,68,0.7)',
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    padding: '0.15rem 0.3rem',
                  }}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Context row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '0.5rem',
        }}
      >
        {[
          { label: '⏱ 时长', value: String(duration), onChange: (v: string) => setDuration(Number(v)), type: 'number', placeholder: '分钟' },
          { label: '📍 场景', value: context.location, onChange: (v: string) => setContext((prev) => ({ ...prev, location: v })), type: 'text', placeholder: '地点' },
          { label: '🌤 天气', value: context.weather, onChange: (v: string) => setContext((prev) => ({ ...prev, weather: v })), type: 'text', placeholder: '天气/活动' },
        ].map((field) => (
          <div key={field.label}>
            <label style={{ display: 'block', fontSize: '0.72rem', color: theme.colors.text.muted, marginBottom: '0.25rem' }}>
              {field.label}
            </label>
            <input
              type={field.type}
              value={field.value}
              onChange={(e) => field.onChange(e.target.value)}
              placeholder={field.placeholder}
              style={{
                width: '100%',
                padding: '0.5rem 0.6rem',
                borderRadius: theme.borderRadius.md,
                border: `1px solid ${theme.colors.border.default}`,
                color: theme.colors.text.primary,
                backgroundColor: theme.colors.background.elevated,
                fontSize: '0.85rem',
                outline: 'none',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = theme.colors.border.focus)}
              onBlur={(e) => (e.target.style.borderColor = theme.colors.border.default)}
            />
          </div>
        ))}
      </div>

      {/* Error */}
      {inlineError && (
        <div style={{ fontSize: '0.82rem', color: '#f87171', padding: '0.4rem 0' }}>
          ⚠️ {inlineError}
        </div>
      )}

      {/* Generate button */}
      <button
        type="button"
        onClick={handleGenerate}
        disabled={loading}
        style={{
          width: '100%',
          padding: '0.85rem 1rem',
          background: loading
            ? 'rgba(29,185,84,0.3)'
            : 'linear-gradient(135deg, #1db954 0%, #179342 100%)',
          borderRadius: theme.borderRadius.lg,
          border: 'none',
          color: '#fff',
          fontSize: '1rem',
          fontWeight: 700,
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.25s ease',
          letterSpacing: '0.02em',
          boxShadow: loading ? 'none' : '0 4px 16px rgba(29,185,84,0.3)',
        }}
      >
        {loading ? '⏳ 生成中...' : '🎵 生成音乐旅程'}
      </button>
    </div>
  );
}
