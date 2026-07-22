'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { theme } from '@/styles/theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

type MemorySection = {
  field: string;
  label: string;
  tone?: string;
  values: string[];
  count?: number;
  deletable?: boolean;
};

type ProfileViewItem = {
  record_id: string;
  layer: 'L1' | 'L2';
  field: string;
  value: string;
  confidence: number;
  source: string;
  expires_at?: number | null;
  evidence_count: number;
  decision_summary?: string;
  why_used?: string;
};

type ProfileView = {
  scope: string;
  title: string;
  items: ProfileViewItem[];
};

type ProfileViews = {
  views?: ProfileView[];
  recent_tendency?: { title?: string; items?: ProfileViewItem[] };
};

type MemoryProfile = {
  user_id?: string;
  episodic_backends?: string[];
  diagnostics?: {
    positive_preference_count?: number;
    negative_preference_count?: number;
    context_preference_count?: number;
    hot_path_has_signal?: boolean;
    episodic_enabled?: boolean;
    needs_more_feedback?: boolean;
  };
  editable_sections?: MemorySection[];
  records?: MemoryRecord[];
  profile_views?: ProfileViews;
};

type MemoryRecord = {
  record_id: string;
  layer: 'L0' | 'L1' | 'L2' | 'L3';
  kind: string;
  source: string;
  confidence: number;
  created_at: number;
  expires_at?: number | null;
  payload?: Record<string, unknown>;
  why_used?: string;
};

const FIELD_OPTIONS = [
  { value: 'add_moods', label: '偏好情绪' },
  { value: 'add_scenarios', label: '偏好场景' },
  { value: 'avoid_genres', label: '避开流派' },
  { value: 'avoid_moods', label: '避开情绪' },
  { value: 'avoid_scenarios', label: '避开场景' },
  { value: 'activity_contexts', label: '探索倾向' },
];

const FIELD_LABELS = Object.fromEntries(FIELD_OPTIONS.map(item => [item.value, item.label]));

function toneStyle(tone?: string): { color: string; background: string; border: string } {
  if (tone === 'avoid') return { color: '#fca5a5', background: 'rgba(248,113,113,0.10)', border: 'rgba(248,113,113,0.25)' };
  if (tone === 'context') return { color: '#93c5fd', background: 'rgba(59,130,246,0.10)', border: 'rgba(59,130,246,0.25)' };
  return { color: '#86efac', background: 'rgba(34,197,94,0.10)', border: 'rgba(34,197,94,0.25)' };
}

export default function MemoryPage() {
  const [memory, setMemory] = useState<MemoryProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [field, setField] = useState(FIELD_OPTIONS[0].value);
  const [value, setValue] = useState('');
  const [editing, setEditing] = useState<{ field: string; oldValue: string; nextValue: string } | null>(null);

  const loadMemory = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch(`${API_URL}/api/memory/profile`);
      const data = await resp.json();
      if (!resp.ok || !data.success) throw new Error(data.error || `读取失败: ${resp.status}`);
      setMemory(data.memory);
    } catch (err: any) {
      setMessage(`读取记忆失败：${err.message || '请确认后端已启动'}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadMemory(); }, [loadMemory]);

  const sections = useMemo<MemorySection[]>(() => {
    const existing = memory?.editable_sections || [];
    const known = new Set(existing.map(section => section.field));
    return [
      ...existing,
      ...FIELD_OPTIONS.filter(item => !known.has(item.value)).map(item => ({
        field: item.value,
        label: item.label,
        tone: 'positive',
        values: [],
        deletable: true,
      })),
    ];
  }, [memory]);

  const addPreference = async (targetField = field, targetValue = value) => {
    const cleaned = targetValue.trim();
    if (!cleaned) return;
    const resp = await fetch(`${API_URL}/api/memory/preference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: 'local_admin', preferences: { [targetField]: [cleaned] } }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.success) throw new Error(data.error || `保存失败: ${resp.status}`);
    setMessage(`已保存：${FIELD_LABELS[targetField] || targetField} / ${cleaned}`);
    setValue('');
    await loadMemory();
  };

  const deletePreference = async (targetField: string, targetValue: string) => {
    const resp = await fetch(
      `${API_URL}/api/memory/preference?field=${encodeURIComponent(targetField)}&value=${encodeURIComponent(targetValue)}`,
      { method: 'DELETE' },
    );
    if (!resp.ok) throw new Error(`删除失败: ${resp.status}`);
    setMessage(`已删除：${targetValue}`);
    await loadMemory();
  };

  const saveEdit = async () => {
    if (!editing) return;
    const nextValue = editing.nextValue.trim();
    if (!nextValue || nextValue === editing.oldValue) {
      setEditing(null);
      return;
    }
    await deletePreference(editing.field, editing.oldValue);
    await addPreference(editing.field, nextValue);
    setEditing(null);
  };

  const clearLearnedMemory = async () => {
    if (!window.confirm('清空所有系统学习到的偏好？手动画像、喜欢和收藏不会被删除。')) return;
    const resp = await fetch(`${API_URL}/api/memory/profile`, { method: 'DELETE' });
    if (!resp.ok) {
      setMessage(`清空失败：${resp.status}`);
      return;
    }
    setMessage('已清空系统学习偏好');
    await loadMemory();
  };

  const deleteRecord = async (recordId: string) => {
    const resp = await fetch(`${API_URL}/api/memory/record/${encodeURIComponent(recordId)}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error(`删除失败: ${resp.status}`);
    setMessage('已删除该条记忆；审计历史保留为删除标记');
    await loadMemory();
  };

  const diagnostics = memory?.diagnostics || {};

  return (
    <div style={{ padding: '1.25rem', color: theme.colors.text.primary, display: 'grid', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '18rem' }}>
          <p style={{ margin: 0, color: theme.colors.text.muted, fontSize: '0.78rem', fontWeight: 700, letterSpacing: '0.08em' }}>MEMORY</p>
          <h1 style={{ margin: '0.15rem 0', fontSize: '2rem', letterSpacing: '-0.03em' }}>我的记忆 / 我的偏好</h1>
          <p style={{ margin: 0, color: theme.colors.text.secondary, fontSize: '0.9rem' }}>
            管理系统从点赞、收藏、拉黑和歌单反馈中学到的偏好。记忆保存在本地结构化账本中，可随时查看、修改、删除。
          </p>
        </div>
        <button onClick={loadMemory} style={smallButtonStyle()}>刷新</button>
        <button onClick={clearLearnedMemory} style={smallButtonStyle('#fca5a5', 'rgba(248,113,113,0.10)', 'rgba(248,113,113,0.30)')}>清空学习记忆</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
        <MetricCard label="正向偏好" value={diagnostics.positive_preference_count || 0} color="#86efac" />
        <MetricCard label="避开偏好" value={diagnostics.negative_preference_count || 0} color="#fca5a5" />
        <MetricCard label="场景/上下文" value={diagnostics.context_preference_count || 0} color="#93c5fd" />
        <MetricCard label="旁路记忆" value={memory?.episodic_backends?.length || 0} color="#c4b5fd" />
      </div>

      {message && (
        <div style={{ padding: '0.7rem 0.85rem', borderRadius: theme.borderRadius.sm, border: `1px solid ${theme.colors.border.default}`, color: message.includes('失败') ? '#fca5a5' : '#86efac', background: 'rgba(255,255,255,0.035)' }}>
          {message}
        </div>
      )}

      <div style={{ display: 'grid', gap: '0.75rem', gridTemplateColumns: 'minmax(0, 1fr)', padding: '1rem', borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.025)' }}>
        <div style={{ fontWeight: 700 }}>新增或修正偏好</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(140px, 200px) minmax(180px, 1fr) auto', gap: '0.7rem', alignItems: 'center' }}>
          <select value={field} onChange={e => setField(e.target.value)} style={inputStyle()}>
            {FIELD_OPTIONS.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
          </select>
          <input value={value} onChange={e => setValue(e.target.value)} placeholder="例如：雨天、lo-fi、不要太吵、少人声" style={inputStyle()} />
          <button onClick={() => addPreference().catch(err => setMessage(err.message))} style={smallButtonStyle(theme.colors.primary.accent, 'rgba(29,185,84,0.12)', 'rgba(29,185,84,0.35)')}>保存</button>
        </div>
      </div>

      {loading ? (
        <div style={{ color: theme.colors.text.muted, padding: '2rem' }}>加载中...</div>
      ) : (
        <div style={{ display: 'grid', gap: '0.85rem' }}>
          {sections.map(section => {
            const style = toneStyle(section.tone);
            return (
              <section key={section.field} style={{ padding: '1rem', borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.025)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.7rem' }}>
                  <div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 800 }}>{section.label}</div>
                    <div style={{ fontSize: '0.74rem', color: theme.colors.text.muted }}>{section.field} · {section.values.length} 条</div>
                  </div>
                </div>
                {section.values.length ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {section.values.map(item => {
                      const isEditing = editing?.field === section.field && editing.oldValue === item;
                      return isEditing ? (
                        <span key={item} style={{ display: 'inline-flex', gap: '0.35rem', alignItems: 'center' }}>
                          <input value={editing.nextValue} onChange={e => setEditing({ ...editing, nextValue: e.target.value })} style={{ ...inputStyle(), width: '11rem', padding: '0.32rem 0.55rem', fontSize: '0.76rem' }} />
                          <button onClick={() => saveEdit().catch(err => setMessage(err.message))} style={chipActionStyle('#86efac')}>保存</button>
                          <button onClick={() => setEditing(null)} style={chipActionStyle()}>取消</button>
                        </span>
                      ) : (
                        <span key={item} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.28rem 0.45rem 0.28rem 0.65rem', borderRadius: '999px', border: `1px solid ${style.border}`, background: style.background, color: style.color, fontSize: '0.76rem' }}>
                          {item}
                          <button onClick={() => setEditing({ field: section.field, oldValue: item, nextValue: item })} title="修改" style={miniIconButtonStyle()}>改</button>
                          <button onClick={() => deletePreference(section.field, item).catch(err => setMessage(err.message))} title="删除" style={miniIconButtonStyle('#fca5a5')}>×</button>
                        </span>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>暂无记录。使用推荐、点赞/拉黑或歌单反馈后会逐步形成。</div>
                )}
              </section>
            );
          })}
        </div>
      )}

      {!loading && ((memory?.profile_views?.views?.length || 0) > 0 || (memory?.profile_views?.recent_tendency?.items?.length || 0) > 0) && (
        <section style={{ padding: '1rem', borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.025)' }}>
          <div style={{ fontWeight: 800, marginBottom: '0.25rem' }}>场景画像视图</div>
          <div style={{ fontSize: '0.76rem', color: theme.colors.text.muted, marginBottom: '0.8rem' }}>
            偏好按适用场景分组：场景绑定的偏好只在对应场景生效，不会带进其他场景。每条都可删除。
          </div>
          <div style={{ display: 'grid', gap: '0.8rem' }}>
            {(memory?.profile_views?.views || []).map(view => (
              <div key={view.scope}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.45rem', color: '#c4b5fd' }}>
                  {view.title} <span style={{ color: theme.colors.text.muted, fontWeight: 400, fontSize: '0.72rem' }}>({view.scope})</span>
                </div>
                <div style={{ display: 'grid', gap: '0.45rem' }}>
                  {view.items.map(item => (
                    <div key={item.record_id} style={{ display: 'grid', gridTemplateColumns: 'auto minmax(0,1fr) auto', gap: '0.6rem', alignItems: 'center', padding: '0.55rem 0.7rem', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm }}>
                      <span style={{ color: item.layer === 'L1' ? '#86efac' : '#93c5fd', fontWeight: 800, fontSize: '0.73rem' }}>{item.layer}</span>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: '0.82rem' }}>
                          {item.value} <span style={{ color: theme.colors.text.muted, fontSize: '0.72rem' }}>({FIELD_LABELS[item.field] || item.field})</span>
                        </div>
                        <div style={{ color: theme.colors.text.muted, fontSize: '0.7rem' }}>
                          置信度 {Math.round(item.confidence * 100)}% · 证据 {item.evidence_count} 条 · {item.expires_at ? `${new Date(item.expires_at).toLocaleDateString()} 前有效` : '长期有效'}
                          {item.decision_summary ? ` · ${item.decision_summary}` : ''}
                        </div>
                      </div>
                      <button onClick={() => deleteRecord(item.record_id).catch(err => setMessage(err.message))} style={chipActionStyle('#fca5a5')}>删除</button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            {(memory?.profile_views?.recent_tendency?.items?.length || 0) > 0 && (
              <div>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.45rem', color: '#fcd34d' }}>
                  {memory?.profile_views?.recent_tendency?.title || '最近推断倾向'}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                  {(memory?.profile_views?.recent_tendency?.items || []).map(item => (
                    <span key={item.record_id} style={{ padding: '0.28rem 0.6rem', borderRadius: '999px', border: '1px solid rgba(252,211,77,0.3)', background: 'rgba(252,211,77,0.08)', color: 'rgba(253,230,175,0.9)', fontSize: '0.74rem' }}>
                      {item.value} · {Math.round(item.confidence * 100)}%
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {!loading && (memory?.records?.length || 0) > 0 && (
        <section style={{ padding: '1rem', borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.025)' }}>
          <div style={{ fontWeight: 800, marginBottom: '0.25rem' }}>记忆来源与有效期</div>
          <div style={{ fontSize: '0.76rem', color: theme.colors.text.muted, marginBottom: '0.8rem' }}>
            L0 是原始行为，L1 是你明确设置的偏好，L2 是有期限的系统推断，L3 是对话情节摘要。
          </div>
          <div style={{ display: 'grid', gap: '0.55rem' }}>
            {(memory?.records || []).slice(0, 40).map(record => {
              const payload = record.payload || {};
              const summary = String(payload.value || payload.description || payload.title || record.kind);
              const expires = record.expires_at ? new Date(record.expires_at).toLocaleDateString() : '长期有效';
              const scope = String(payload.scope || '');
              const evidenceIds = Array.isArray(payload.evidence_ids) ? payload.evidence_ids : [];
              return (
                <div key={record.record_id} style={{ display: 'grid', gridTemplateColumns: 'auto minmax(0, 1fr) auto', gap: '0.65rem', alignItems: 'center', padding: '0.65rem 0.75rem', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm }}>
                  <span style={{ color: '#93c5fd', fontWeight: 800, fontSize: '0.75rem' }}>{record.layer}</span>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: '0.82rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{summary}</div>
                    <div style={{ color: theme.colors.text.muted, fontSize: '0.7rem' }}>
                      {record.source} · 置信度 {Math.round(record.confidence * 100)}% · {expires}
                      {scope && scope !== 'global' ? ` · 场景:${scope}` : ''}
                      {evidenceIds.length ? ` · 证据 ${evidenceIds.length} 条` : ''}
                      {record.why_used ? ` · ${record.why_used}` : ''}
                    </div>
                  </div>
                  <button onClick={() => deleteRecord(record.record_id).catch(err => setMessage(err.message))} style={chipActionStyle('#fca5a5')}>删除</button>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ padding: '0.85rem 1rem', borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.025)' }}>
      <div style={{ color, fontSize: '1.35rem', fontWeight: 800 }}>{value}</div>
      <div style={{ color: theme.colors.text.secondary, fontSize: '0.78rem' }}>{label}</div>
    </div>
  );
}

function inputStyle(): React.CSSProperties {
  return {
    width: '100%',
    padding: '0.58rem 0.75rem',
    borderRadius: theme.borderRadius.sm,
    border: `1px solid ${theme.colors.border.default}`,
    background: 'rgba(255,255,255,0.05)',
    color: theme.colors.text.primary,
    outline: 'none',
  };
}

function smallButtonStyle(color = theme.colors.text.primary, background = 'rgba(255,255,255,0.05)', border = theme.colors.border.default): React.CSSProperties {
  return {
    border: `1px solid ${border}`,
    background,
    color,
    borderRadius: theme.borderRadius.sm,
    cursor: 'pointer',
    padding: '0.58rem 0.85rem',
    fontWeight: 700,
    fontSize: '0.8rem',
  };
}

function chipActionStyle(color = theme.colors.text.secondary): React.CSSProperties {
  return {
    border: `1px solid ${theme.colors.border.default}`,
    background: 'rgba(255,255,255,0.04)',
    color,
    borderRadius: '999px',
    cursor: 'pointer',
    padding: '0.18rem 0.45rem',
    fontSize: '0.72rem',
  };
}

function miniIconButtonStyle(color = theme.colors.text.muted): React.CSSProperties {
  return {
    border: 'none',
    background: 'rgba(255,255,255,0.08)',
    color,
    borderRadius: '999px',
    cursor: 'pointer',
    minWidth: '1.35rem',
    height: '1.35rem',
    fontSize: '0.68rem',
  };
}
