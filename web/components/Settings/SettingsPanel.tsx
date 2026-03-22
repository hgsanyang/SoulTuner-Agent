'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { theme } from '@/styles/theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

// ---- LLM 提供商预设列表 ----
const LLM_PROVIDERS = [
  { value: 'siliconflow', label: 'SiliconFlow', defaultModel: 'deepseek-ai/DeepSeek-V3' },
  { value: 'dashscope', label: '通义千问 (DashScope)', defaultModel: 'qwen-plus' },
  { value: 'google', label: 'Google Gemini', defaultModel: 'gemini-2.5-flash' },
  { value: 'ollama', label: 'Ollama (本地)', defaultModel: 'qwen2.5:7b' },
  { value: 'vllm', label: 'vLLM (微调模型)', defaultModel: '' },
];

// ---- 标签页定义 ----
type TabKey = 'models' | 'retrieval' | 'paths' | 'memory';
const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: 'models', label: '模型配置', icon: '🤖' },
  { key: 'retrieval', label: '检索参数', icon: '🔍' },
  { key: 'paths', label: '音乐数据', icon: '🎵' },
  { key: 'memory', label: '记忆系统', icon: '🧠' },
];

interface Settings {
  [key: string]: string | number | boolean;
}

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsPanel({ isOpen, onClose }: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<TabKey>('models');
  const [settings, setSettings] = useState<Settings>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState<Set<string>>(new Set());
  const [saveMessage, setSaveMessage] = useState('');

  // ★ 快照：记录上次从后端拿到的干净数据
  const snapshotRef = useRef<Settings>({});

  // ---- 加载设置 ----
  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/settings`);
      if (res.ok) {
        const data = await res.json();
        snapshotRef.current = { ...data };   // 保存快照
        setSettings(data);
      }
    } catch (e) {
      console.error('Failed to load settings:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  // ★ 关闭时恢复到快照（丢弃本地未保存修改）
  const handleClose = useCallback(() => {
    setSettings({ ...snapshotRef.current });  // 还原快照
    setDirty(new Set());
    setSaveMessage('');
    onClose();
  }, [onClose]);

  useEffect(() => {
    if (isOpen) {
      setDirty(new Set());
      setSaveMessage('');
      loadSettings();
    }
  }, [isOpen, loadSettings]);

  // ---- 更新单个字段 ----
  const updateField = (key: string, value: string | number | boolean) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setDirty(prev => new Set(prev).add(key));
  };

  // ---- 保存修改 ----
  const saveSettings = async () => {
    if (dirty.size === 0) return;
    setSaving(true);
    try {
      const payload: Record<string, unknown> = {};
      dirty.forEach(key => { payload[key] = settings[key]; });

      const res = await fetch(`${API_URL}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await res.json();
      if (result.success) {
        snapshotRef.current = { ...settings };  // 保存成功 → 更新快照
        setDirty(new Set());
        setSaveMessage(`✅ 已更新: ${result.updated.join(', ')}`);
        setTimeout(() => setSaveMessage(''), 3000);
      }
    } catch (e) {
      setSaveMessage('❌ 保存失败，请确认后端已启动');
      setTimeout(() => setSaveMessage(''), 3000);
    } finally {
      setSaving(false);
    }
  };

  // ---- 还原默认配置 ----
  const resetToDefaults = async () => {
    try {
      setSaving(true);
      const res = await fetch(`${API_URL}/api/settings/reset`, { method: 'POST' });
      if (res.ok) {
        const result = await res.json();
        snapshotRef.current = { ...result.settings };
        setSettings(result.settings);
        setDirty(new Set());
        setSaveMessage('✅ 已还原为默认配置');
        setTimeout(() => setSaveMessage(''), 3000);
      }
    } catch (e) {
      setSaveMessage('❌ 还原失败，请确认后端已启动');
      setTimeout(() => setSaveMessage(''), 3000);
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  // ---- 通用控件样式 ----
  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '0.6rem 0.8rem',
    background: theme.colors.background.card,
    border: `1px solid ${theme.colors.border.default}`,
    borderRadius: theme.borderRadius.sm,
    color: theme.colors.text.primary,
    fontSize: '0.85rem',
    outline: 'none',
    transition: 'border-color 0.2s',
  };

  const selectStyle: React.CSSProperties = { ...inputStyle, cursor: 'pointer' };

  const labelStyle: React.CSSProperties = {
    fontSize: '0.8rem',
    color: theme.colors.text.secondary,
    marginBottom: '0.3rem',
    display: 'block',
  };

  const fieldGroup: React.CSSProperties = { marginBottom: '1rem' };

  const sliderStyle: React.CSSProperties = {
    width: '100%',
    accentColor: theme.colors.primary.accent,
    cursor: 'pointer',
  };

  // ---- 渲染控件 ----
  const renderSelect = (key: string, label: string, options: { value: string; label: string }[]) => (
    <div style={fieldGroup}>
      <label style={labelStyle}>{label}</label>
      <select
        style={selectStyle}
        value={String(settings[key] || '')}
        onChange={e => updateField(key, e.target.value)}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );

  const renderInput = (key: string, label: string, placeholder?: string, type?: string) => (
    <div style={fieldGroup}>
      <label style={labelStyle}>{label}</label>
      <input
        style={inputStyle}
        type={type || 'text'}
        value={String(settings[key] || '')}
        placeholder={placeholder}
        onChange={e => updateField(key, type === 'number' ? Number(e.target.value) : e.target.value)}
      />
    </div>
  );

  const renderSlider = (key: string, label: string, min: number, max: number, step: number, unit?: string) => (
    <div style={fieldGroup}>
      <label style={labelStyle}>
        {label}: <strong style={{ color: theme.colors.primary.accent }}>{settings[key]}{unit || ''}</strong>
      </label>
      <input
        style={sliderStyle}
        type="range"
        min={min} max={max} step={step}
        value={Number(settings[key] || min)}
        onChange={e => updateField(key, Number(e.target.value))}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: theme.colors.text.muted }}>
        <span>{min}{unit || ''}</span><span>{max}{unit || ''}</span>
      </div>
    </div>
  );

  const renderToggle = (key: string, label: string) => (
    <div style={{ ...fieldGroup, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <label style={{ ...labelStyle, marginBottom: 0 }}>{label}</label>
      <button
        onClick={() => updateField(key, !settings[key])}
        style={{
          width: '44px', height: '24px', borderRadius: '12px', border: 'none', cursor: 'pointer',
          background: settings[key] ? theme.colors.primary.accent : theme.colors.primary[400],
          position: 'relative', transition: 'background 0.2s',
        }}
      >
        <div style={{
          width: '18px', height: '18px', borderRadius: '50%', background: '#fff',
          position: 'absolute', top: '3px',
          left: settings[key] ? '23px' : '3px',
          transition: 'left 0.2s',
        }} />
      </button>
    </div>
  );

  // ---- 标签页内容 ----
  const renderModelsTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🤖 LLM 模型配置</h4>
      {renderSelect('llm_default_provider', '主 LLM 提供商', LLM_PROVIDERS.map(p => ({ value: p.value, label: p.label })))}
      {renderInput('llm_default_model', '主 LLM 模型名', LLM_PROVIDERS.find(p => p.value === settings.llm_default_provider)?.defaultModel)}

      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>专用模型（可选，空则复用主模型）</span>
      </div>
      {renderSelect('intent_llm_provider', '意图分析 LLM', [{ value: '', label: '-- 复用主模型 --' }, ...LLM_PROVIDERS.map(p => ({ value: p.value, label: p.label }))])}
      {renderInput('intent_llm_model', '意图分析模型名', '如: qwen3.5-4b-soultuner')}
      {renderSelect('hyde_llm_provider', 'HyDE 描述 LLM', [{ value: '', label: '-- 复用主模型 --' }, ...LLM_PROVIDERS.map(p => ({ value: p.value, label: p.label }))])}
      {renderInput('hyde_llm_model', 'HyDE 描述模型名', '')}
      {renderInput('intent_model_path', '意图分析微调模型路径', '/path/to/intent-sft-model')}
      {renderInput('hyde_model_path', 'HyDE 描述微调模型路径', '/path/to/hyde-grpo-model')}
      {renderSlider('llm_timeout', 'LLM 超时', 10, 120, 5, '秒')}
    </>
  );

  const renderRetrievalTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🔍 检索 & 排序参数</h4>
      {renderSlider('graph_search_limit', '图谱检索数量', 3, 30, 1)}
      {renderSlider('semantic_search_limit', '向量检索数量', 3, 30, 1)}
      {renderSlider('hybrid_retrieval_limit', '歌单输出数量', 3, 30, 1)}
      {renderSlider('web_search_max_results', '联网搜索数量', 1, 10, 1)}

      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>RRF 加权融合</span>
      </div>
      {renderSlider('rrf_weight_vector', '向量检索权重', 0, 1, 0.05)}
      <div style={{ fontSize: '0.75rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
        图谱权重自动计算为 {(1 - Number(settings.rrf_weight_vector || 0.7)).toFixed(2)}
      </div>

      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>Neo4j 图距离加权</span>
      </div>
      {renderToggle('graph_affinity_enabled', '启用图距离加权')}
      {settings.graph_affinity_enabled && (
        <>
          {renderSlider('graph_affinity_weight', '亲和力权重', 0, 0.5, 0.05)}
          {renderSlider('graph_affinity_max_hops', '最大跳数', 2, 8, 1)}
        </>
      )}
    </>
  );

  const renderPathsTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🎵 音乐数据目录</h4>
      {renderInput('audio_data_dir', '本地音乐目录', 'data/processed_audio/audio')}
      {renderInput('mtg_audio_dir', 'MTG 数据集目录', 'data/mtg_sample/audio')}
      {renderInput('online_acquired_dir', '联网获取目录', 'data/online_acquired')}
      {renderInput('model_output_dir', '模型训练导出目录', 'output/sft-checkpoint')}
    </>
  );

  const renderMemoryTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🧠 记忆 & 上下文</h4>
      {renderSlider('memory_retain_rounds', '上下文保留轮数', 1, 20, 1, '轮')}
      {renderInput('default_user_id', '用户 ID', 'local_admin')}
    </>
  );

  const tabContent: Record<TabKey, () => JSX.Element> = {
    models: renderModelsTab,
    retrieval: renderRetrievalTab,
    paths: renderPathsTab,
    memory: renderMemoryTab,
  };

  return createPortal(
    <>
      {/* 遮罩 */}
      <div onClick={handleClose} style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
        zIndex: 9999, backdropFilter: 'blur(4px)',
      }} />

      {/* 面板 */}
      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
        width: '680px', maxWidth: '90vw', maxHeight: '85vh',
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
            <h3 style={{ margin: 0, color: theme.colors.text.primary, fontSize: '1.1rem' }}>⚙️ 系统设置</h3>
            <span style={{ fontSize: '0.75rem', color: theme.colors.text.muted }}>修改后点击保存即时生效，关闭则丢弃未保存修改</span>
          </div>
          <button onClick={handleClose} style={{
            background: 'transparent', border: 'none', color: theme.colors.text.muted,
            fontSize: '1.2rem', cursor: 'pointer', padding: '0.3rem',
          }}>✕</button>
        </div>

        {/* 主体 */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* 标签栏 */}
          <div style={{
            width: '140px', borderRight: `1px solid ${theme.colors.border.default}`,
            padding: '0.8rem 0', display: 'flex', flexDirection: 'column', gap: '0.2rem',
          }}>
            {TABS.map(tab => (
              <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
                display: 'flex', alignItems: 'center', gap: '0.5rem',
                padding: '0.7rem 1rem', border: 'none', cursor: 'pointer',
                background: activeTab === tab.key ? theme.colors.background.hover : 'transparent',
                color: activeTab === tab.key ? theme.colors.text.primary : theme.colors.text.muted,
                fontSize: '0.82rem', textAlign: 'left',
                borderRight: activeTab === tab.key ? `2px solid ${theme.colors.primary.accent}` : '2px solid transparent',
                transition: 'all 0.15s',
              }}>
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          {/* 内容区 */}
          <div style={{
            flex: 1, padding: '1.2rem 1.5rem', overflowY: 'auto',
          }}>
            {loading ? (
              <div style={{ textAlign: 'center', color: theme.colors.text.muted, padding: '2rem' }}>
                加载中...
              </div>
            ) : tabContent[activeTab]()}
          </div>
        </div>

        {/* 底部操作栏 */}
        <div style={{
          padding: '0.8rem 1.5rem', borderTop: `1px solid ${theme.colors.border.default}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span style={{ fontSize: '0.78rem', color: dirty.size > 0 ? '#f0a040' : theme.colors.text.muted }}>
            {saveMessage || (dirty.size > 0 ? `${dirty.size} 项修改未保存` : '所有配置已同步')}
          </span>
          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <button onClick={resetToDefaults} style={{
              padding: '0.5rem 1rem', background: 'transparent',
              border: `1px solid ${theme.colors.border.default}`,
              borderRadius: theme.borderRadius.sm, color: '#f06060',
              cursor: 'pointer', fontSize: '0.78rem',
            }}>
              ↩ 还原默认
            </button>
            <button onClick={handleClose} style={{
              padding: '0.5rem 1.2rem', background: 'transparent',
              border: `1px solid ${theme.colors.border.default}`,
              borderRadius: theme.borderRadius.sm, color: theme.colors.text.secondary,
              cursor: 'pointer', fontSize: '0.82rem',
            }}>
              关闭
            </button>
            <button onClick={saveSettings} disabled={dirty.size === 0 || saving} style={{
              padding: '0.5rem 1.5rem',
              background: dirty.size > 0 ? theme.colors.primary.accent : theme.colors.primary[400],
              border: 'none', borderRadius: theme.borderRadius.sm,
              color: dirty.size > 0 ? '#000' : theme.colors.text.muted,
              cursor: dirty.size > 0 ? 'pointer' : 'default',
              fontWeight: 600, fontSize: '0.82rem',
              transition: 'all 0.2s',
            }}>
              {saving ? '保存中...' : '💾 保存设置'}
            </button>
          </div>
        </div>
      </div>
    </>,
    document.body
  );
}
