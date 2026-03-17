'use client';

import { useState } from 'react';
import { JourneyRequest, MoodTransitionInput } from '@/lib/api';
import { theme } from '@/styles/theme';

const DEFAULT_STORY = '早晨起床 → 通勤路上 → 工作中 → 下班放松 → 夜晚休息';

const STORY_PRESETS: string[] = [
  '机场清晨 → 登机等待 → 起飞穿云 → 落地黄昏 → 入住酒店的小惊喜',
  '雨天咖啡馆 → 专注写作 → 灵感爆发 → 走出门口的微风',
  '城市夜跑 → 河边灯光 → 冲刺冲线 → 回家拉伸与放松',
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
  time: number; // 0-1
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
      {
        id: nextId,
        time: 1,
        mood: '放松',
        intensity: 0.5,
      },
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

  const buildRequestPayload = (): JourneyRequest | null => {
    setInlineError(null);
    if (mode === 'story') {
      if (!story.trim()) {
        setInlineError('请先写一点故事，再让我们为它配一段音乐。');
        return null;
      }
      return {
        story: story.trim(),
        duration,
        context,
      };
    }

    const validPoints: MoodTransitionInput[] = moodPoints
      .filter((point) => point.time >= 0 && point.time <= 1)
      .sort((a, b) => a.time - b.time)
      .map((point) => ({
        time: point.time,
        mood: point.mood,
        intensity: Math.max(0, Math.min(1, point.intensity)),
      }));

    if (!validPoints.length) {
      return null;
    }

    return {
      mood_transitions: validPoints,
      duration,
      context,
    };
  };

  const handleGenerate = () => {
    const payload = buildRequestPayload();
    if (!payload) {
      return;
    }
    onGenerate(payload);
  };

  return (
    <section
      style={{
        backgroundColor: theme.colors.background.card,
        borderRadius: theme.borderRadius.lg,
        border: `1px solid ${theme.colors.border.default}`,
        padding: '1.75rem',
        marginBottom: '2rem',
        boxShadow: '0 10px 30px rgba(15, 23, 42, 0.08)',
      }}
    >
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1.5rem',
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: '1.5rem',
              color: theme.colors.text.primary,
            }}
          >
            音乐旅程生成器
          </h2>
          <p style={{ marginTop: '0.35rem', color: theme.colors.text.muted }}>
            通过故事或情绪曲线创建沉浸式音乐旅程
          </p>
        </div>
        <div
          style={{
            display: 'flex',
            gap: '0.5rem',
            backgroundColor: theme.colors.background.main,
            padding: '0.25rem',
            borderRadius: theme.borderRadius.md,
          }}
        >
          {[
            { label: '故事驱动', value: 'story' },
            { label: '情绪曲线', value: 'mood' },
          ].map((item) => (
            <button
              key={item.value}
              onClick={() => setMode(item.value as 'story' | 'mood')}
              type="button"
              style={{
                border: 'none',
                backgroundColor:
                  mode === item.value ? theme.colors.primary[700] : 'transparent',
                color: mode === item.value ? '#fff' : theme.colors.text.secondary,
                padding: '0.45rem 0.9rem',
                borderRadius: theme.borderRadius.md,
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      </header>

      {mode === 'story' ? (
        <div style={{ marginBottom: '1.5rem' }}>
          <label
            style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: theme.colors.text.secondary,
              fontWeight: 600,
            }}
          >
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
              padding: '1rem',
              fontSize: '1rem',
              lineHeight: 1.6,
              color: theme.colors.text.primary,
              backgroundColor: theme.colors.background.elevated,
            }}
          />
          <div
            style={{
              marginTop: '0.6rem',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.5rem',
              alignItems: 'center',
            }}
          >
            <span
              style={{
                fontSize: '0.85rem',
                color: theme.colors.text.muted,
              }}
            >
              不知道写什么？试试这些旅程草稿：
            </span>
            {STORY_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                onClick={() => setStory(preset)}
                style={{
                  border: 'none',
                  borderRadius: theme.borderRadius.full,
                  padding: '0.3rem 0.75rem',
                  fontSize: '0.82rem',
                  cursor: 'pointer',
                  backgroundColor: theme.colors.background.main,
                  color: theme.colors.text.secondary,
                }}
              >
                一键注入
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div style={{ marginBottom: '1.5rem' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: '0.75rem',
              alignItems: 'center',
            }}
          >
            <label
              style={{
                color: theme.colors.text.secondary,
                fontWeight: 600,
              }}
            >
              情绪曲线
            </label>
            <button
              type="button"
              onClick={handleAddMoodPoint}
              style={{
                border: 'none',
                backgroundColor: 'transparent',
                color: theme.colors.text.secondary,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              + 添加节点
            </button>
          </div>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            {moodPoints.map((point) => (
              <div
                key={point.id}
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, minmax(0, 1fr)) auto',
                  gap: '0.75rem',
                  alignItems: 'center',
                  padding: '0.85rem',
                  borderRadius: theme.borderRadius.md,
                  border: `1px solid ${theme.colors.border.default}`,
                  backgroundColor: theme.colors.background.main,
                }}
              >
                <div>
                  <label
                    style={{
                      display: 'block',
                      fontSize: '0.85rem',
                      color: theme.colors.text.secondary,
                      marginBottom: '0.25rem',
                    }}
                  >
                    时间点（%）
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={Math.round(point.time * 100)}
                    onChange={(e) =>
                      handleUpdateMoodPoint(point.id, 'time', Number(e.target.value) / 100)
                    }
                    style={{
                      width: '100%',
                      padding: '0.5rem',
                      borderRadius: theme.borderRadius.sm,
                      border: `1px solid ${theme.colors.border.default}`,
                      color: theme.colors.text.primary,
                      backgroundColor: theme.colors.background.elevated,
                    }}
                  />
                </div>
                <div>
                  <label
                    style={{
                      display: 'block',
                      fontSize: '0.85rem',
                      color: theme.colors.text.secondary,
                      marginBottom: '0.25rem',
                    }}
                  >
                    情绪
                  </label>
                  <select
                    value={point.mood}
                    onChange={(e) => handleUpdateMoodPoint(point.id, 'mood', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.5rem',
                      borderRadius: theme.borderRadius.sm,
                      border: `1px solid ${theme.colors.border.default}`,
                      color: theme.colors.text.primary,
                      backgroundColor: theme.colors.background.elevated,
                    }}
                  >
                    {moodOptions.map((mood) => (
                      <option key={mood} value={mood}>
                        {mood}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label
                    style={{
                      display: 'block',
                      fontSize: '0.85rem',
                      color: theme.colors.text.secondary,
                      marginBottom: '0.25rem',
                    }}
                  >
                    强度
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={point.intensity}
                    onChange={(e) =>
                      handleUpdateMoodPoint(point.id, 'intensity', Number(e.target.value))
                    }
                    style={{
                      width: '100%',
                      padding: '0.5rem',
                      borderRadius: theme.borderRadius.sm,
                      border: `1px solid ${theme.colors.border.default}`,
                      color: theme.colors.text.primary,
                      backgroundColor: theme.colors.background.elevated,
                    }}
                  />
                </div>
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.35rem',
                  }}
                >
                  <label
                    style={{
                      fontSize: '0.85rem',
                      color: theme.colors.text.secondary,
                    }}
                  >
                    强度滑块
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={point.intensity}
                    onChange={(e) =>
                      handleUpdateMoodPoint(point.id, 'intensity', Number(e.target.value))
                    }
                  />
                </div>
                <button
                  type="button"
                  onClick={() => handleDeleteMoodPoint(point.id)}
                  style={{
                    border: 'none',
                    backgroundColor: 'transparent',
                    color: theme.colors.text.secondary,
                    cursor: 'pointer',
                  }}
                >
                  删除
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          marginBottom: '1.5rem',
        }}
      >
        <div>
          <label
            style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: theme.colors.text.secondary,
              fontWeight: 600,
            }}
          >
            旅程时长（分钟）
          </label>
          <input
            type="number"
            min={20}
            max={180}
            step={5}
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              color: theme.colors.text.primary,
              backgroundColor: theme.colors.background.elevated,
            }}
          />
        </div>
        <div>
          <label
            style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: theme.colors.text.secondary,
              fontWeight: 600,
            }}
          >
            场景 / 地点
          </label>
          <input
            type="text"
            value={context.location}
            onChange={(e) => setContext((prev) => ({ ...prev, location: e.target.value }))}
            placeholder="如：上海 / 办公室 / 旅途"
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              color: theme.colors.text.primary,
              backgroundColor: theme.colors.background.elevated,
            }}
          />
        </div>
        <div>
          <label
            style={{
              display: 'block',
              marginBottom: '0.5rem',
              color: theme.colors.text.secondary,
              fontWeight: 600,
            }}
          >
            天气 / 活动
          </label>
          <input
            type="text"
            value={context.weather}
            onChange={(e) => setContext((prev) => ({ ...prev, weather: e.target.value }))}
            placeholder="如：晴朗 / 雨天 / 夜跑"
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              color: theme.colors.text.primary,
              backgroundColor: theme.colors.background.elevated,
            }}
          />
        </div>
      </div>

      <button
        type="button"
        onClick={handleGenerate}
        disabled={loading}
        style={{
          width: '100%',
          padding: '0.9rem 1.25rem',
          backgroundColor: loading ? 'rgba(29,185,84,0.5)' : theme.colors.primary.accent,
          borderRadius: theme.borderRadius.lg,
          border: 'none',
          color: '#fff',
          fontSize: '1.0625rem',
          fontWeight: 600,
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.25s ease',
          letterSpacing: '0.02em',
        }}
      >
        {loading ? '⏳ 生成中...' : '🎵 生成音乐旅程'}
      </button>
    </section>
  );
}

