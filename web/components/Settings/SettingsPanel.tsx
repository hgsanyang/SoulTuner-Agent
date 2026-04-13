'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { theme } from '@/styles/theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

// ---- LLM 提供商预设列表 ----
const LLM_PROVIDERS = [
  { value: 'siliconflow', label: 'SiliconFlow (API)', defaultModel: 'deepseek-ai/DeepSeek-V3.2' },
  { value: 'volcengine', label: '火山引擎 / 豆包 (字节跳动)', defaultModel: 'ep-20260405142751-x4jm6' },
  { value: 'dashscope', label: '通义千问 DashScope (API)', defaultModel: 'qwen-plus' },
  { value: 'google', label: 'Google Gemini (API)', defaultModel: 'gemini-3-flash-preview' },
  { value: 'deepseek', label: 'DeepSeek (API)', defaultModel: 'deepseek-chat' },
  { value: 'sglang', label: 'SGLang (本地推荐)', defaultModel: 'local-planner-qwen3-4b-fp8' },
  { value: 'ollama', label: 'Ollama (本地)', defaultModel: 'qwen2.5:7b' },
  { value: 'vllm', label: 'vLLM (本地微调)', defaultModel: '' },
];

// ---- 每个 Provider 的常用模型预设列表 ----
const MODEL_PRESETS: Record<string, string[]> = {
  siliconflow: [
    'deepseek-ai/DeepSeek-V3.2',
    'Qwen/Qwen3.5-35B-A3B',
    'THUDM/GLM-4-32B-0414',
    'Pro/Qwen/Qwen2.5-7B-Instruct',
  ],
  deepseek: ['deepseek-chat', 'deepseek-reasoner'],
  dashscope: ['deepseek-v3.2', 'deepseek-v3', 'qwen-plus', 'qwen-turbo', 'qwen-max', 'qwen3.5-flash'],
  google: ['gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-pro'],
  volcengine: ['ep-20260405142751-x4jm6'],
  sglang: ['local-planner-qwen3-4b-fp8'],
  ollama: ['qwen2.5:7b', 'qwen2.5:3b', 'llama3.1:8b'],
  vllm: ['Qwen/Qwen2.5-7B-Instruct'],
};

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
  // 部署模式: 'api' = 意图+HyDE共用一个API模型, 'local' = 分别配置意图/HyDE模型
  const [deployMode, setDeployMode] = useState<'api' | 'local'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('soultuner_deploy_mode') as 'api' | 'local') || 'api';
    }
    return 'api';
  });

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

  // ---- 复合控件：Provider 选择 + 模型预设下拉 ----
  const renderModelPicker = (
    providerKey: string, modelKey: string,
    providerLabel: string, modelLabel: string,
    allowReuse = false,
  ) => {
    const currentProvider = String(settings[providerKey] || '');
    const presets = MODEL_PRESETS[currentProvider] || [];
    const currentModel = String(settings[modelKey] || '');
    const isCustom = currentModel !== '' && !presets.includes(currentModel);

    return (
      <div style={{ marginBottom: '1rem' }}>
        {/* Provider 选择 */}
        <div style={fieldGroup}>
          <label style={labelStyle}>{providerLabel}</label>
          <select
            style={selectStyle}
            value={currentProvider}
            onChange={e => {
              updateField(providerKey, e.target.value);
              // 自动填入新 provider 的默认模型
              const newPresets = MODEL_PRESETS[e.target.value];
              if (newPresets && newPresets.length > 0) {
                updateField(modelKey, newPresets[0]);
              } else {
                const p = LLM_PROVIDERS.find(p => p.value === e.target.value);
                if (p) updateField(modelKey, p.defaultModel);
              }
            }}
          >
            {allowReuse && <option value="">-- 复用主模型 --</option>}
            {LLM_PROVIDERS.map(p => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        {/* 模型选择：有预设时显示下拉，否则显示输入框 */}
        {(!allowReuse || currentProvider) && (
          <div style={fieldGroup}>
            <label style={labelStyle}>{modelLabel}</label>
            {presets.length > 0 ? (
              <select
                style={selectStyle}
                value={isCustom ? '__custom__' : currentModel}
                onChange={e => {
                  if (e.target.value === '__custom__') {
                    updateField(modelKey, '');
                  } else {
                    updateField(modelKey, e.target.value);
                  }
                }}
              >
                {presets.map(m => <option key={m} value={m}>{m}</option>)}
                <option value="__custom__">✏️ 自定义...</option>
              </select>
            ) : (
              <input
                style={inputStyle}
                value={currentModel}
                placeholder={LLM_PROVIDERS.find(p => p.value === currentProvider)?.defaultModel || '输入模型名'}
                onChange={e => updateField(modelKey, e.target.value)}
              />
            )}
            {/* 自定义输入框（仅当选择了"自定义"时显示） */}
            {presets.length > 0 && isCustom && (
              <input
                style={{ ...inputStyle, marginTop: '0.4rem' }}
                value={currentModel}
                placeholder="输入自定义模型名"
                onChange={e => updateField(modelKey, e.target.value)}
                autoFocus
              />
            )}
          </div>
        )}
      </div>
    );
  };

  // ---- 部署模式切换按钮样式 ----
  const modeButtonStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    padding: '0.55rem 0.8rem',
    fontSize: '0.82rem',
    fontWeight: active ? 700 : 400,
    color: active ? '#fff' : theme.colors.text.secondary,
    backgroundColor: active ? theme.colors.primary.accent : 'rgba(255,255,255,0.04)',
    border: `1px solid ${active ? theme.colors.primary.accent : theme.colors.border.default}`,
    borderRadius: theme.borderRadius.sm,
    cursor: 'pointer',
    transition: 'all 0.2s',
    textAlign: 'center' as const,
  });

  const sectionTitleStyle: React.CSSProperties = {
    fontSize: '0.8rem',
    color: theme.colors.text.muted,
    fontWeight: 600,
    letterSpacing: '0.05em',
    padding: '0.6rem 0.8rem',
    margin: '1rem 0 0.6rem',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: theme.borderRadius.sm,
    borderLeft: `3px solid ${theme.colors.primary.accent}`,
  };

  // ---- 标签页内容 ----
  const renderModelsTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🤖 LLM 模型配置</h4>

      {/* ═══ 部署模式切换 ═══ */}
      <div style={{ marginBottom: '1.2rem' }}>
        <label style={{ ...labelStyle, marginBottom: '0.5rem' }}>部署模式</label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            style={modeButtonStyle(deployMode === 'api')}
            onClick={() => {
              setDeployMode('api');
              localStorage.setItem('soultuner_deploy_mode', 'api');
              // API 模式下清空单独的 intent/hyde provider（复用主模型）
              updateField('intent_llm_provider', '');
              updateField('intent_llm_model', '');
              updateField('hyde_llm_provider', '');
              updateField('hyde_llm_model', '');
            }}
          >
            ☁️ API 部署
            <div style={{ fontSize: '0.7rem', fontWeight: 400, opacity: 0.7, marginTop: '2px' }}>
              意图分析 + HyDE 共用一个模型
            </div>
          </button>
          <button
            style={modeButtonStyle(deployMode === 'local')}
            onClick={() => {
              setDeployMode('local');
              localStorage.setItem('soultuner_deploy_mode', 'local');
            }}
          >
            💻 本地部署
            <div style={{ fontSize: '0.7rem', fontWeight: 400, opacity: 0.7, marginTop: '2px' }}>
              分别配置意图分析 / HyDE 模型
            </div>
          </button>
        </div>
      </div>

      {/* ═══ API 模式：只配一个模型 ═══ */}
      {deployMode === 'api' && (
        <>
          <div style={sectionTitleStyle}>☁️ 推荐模型（意图分析 + HyDE 共用）</div>
          {renderModelPicker('llm_default_provider', 'llm_default_model', '提供商', '模型')}
        </>
      )}

      {/* ═══ 本地模式：分开配置 ═══ */}
      {deployMode === 'local' && (
        <>
          <div style={sectionTitleStyle}>🧠 意图分析模型</div>
          {renderModelPicker('intent_llm_provider', 'intent_llm_model', '提供商', '模型')}
          {renderInput('intent_model_path', '微调模型路径（可选）', '/path/to/intent-sft-model')}

          <div style={sectionTitleStyle}>📝 HyDE 描述模型</div>
          {renderModelPicker('hyde_llm_provider', 'hyde_llm_model', '提供商', '模型')}
          {renderInput('hyde_model_path', '微调模型路径（可选）', '/path/to/hyde-grpo-model')}
        </>
      )}

      {/* ═══ 解释生成模型（通用，两种模式都可独立配置）═══ */}
      <div style={sectionTitleStyle}>💬 解释生成模型（通用）</div>
      <div style={{ fontSize: '0.73rem', color: theme.colors.text.muted, marginBottom: '0.6rem', lineHeight: 1.5 }}>
        负责生成推荐理由和最终解释文本（流式输出给用户），建议使用表达能力强的模型
      </div>
      {renderModelPicker('explain_llm_provider', 'explain_llm_model', '提供商', '模型', true)}

      {/* ═══ 上下文压缩（通用设置）═══ */}
      <div style={sectionTitleStyle}>🗜️ 上下文压缩（通用）</div>
      {renderModelPicker('compress_llm_provider', 'compress_llm_model', '提供商', '模型', true)}

      {/* ═══ 超时 ═══ */}
      {renderSlider('llm_timeout', 'LLM 超时', 10, 120, 5, '秒')}
    </>
  );

  const renderRetrievalTab = () => (
    <>
      <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🔍 检索 & 排序参数</h4>

      {/* ═══ 检索数量 ═══ */}
      {renderSlider('graph_search_limit', '图谱检索数量（仅图谱模式）', 3, 30, 1)}
      {renderSlider('semantic_search_limit', '向量检索数量（仅向量模式）', 3, 30, 1)}
      {renderSlider('mixed_retrieval_limit', '混合检索数量（每引擎各返回）', 3, 30, 1)}
      {renderSlider('hybrid_retrieval_limit', '歌单输出数量（最终展示）', 3, 30, 1)}
      {renderSlider('web_search_max_results', '联网搜索数量', 1, 10, 1)}

      {/* ═══ 粗排 & 探索 ═══ */}
      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>粗排 & 探索（Graph Affinity + Thompson Sampling）</span>
      </div>
      {renderToggle('graph_affinity_enabled', '启用图距离粗排 + TS 探索')}
      {settings.graph_affinity_enabled && (
        <>
          {renderSlider('coarse_cut_ratio', '粗排保留比例', 0.3, 1, 0.05)}
          <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
            例: 0.65 = 保留 65% 候选歌曲进入精排，其余淘汰
          </div>
          {renderSlider('exploration_ratio', '小众歌曲曝光度', 0, 0.5, 0.05)}
          <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
            从淘汰歌曲中按此比例捞回冷门歌（Thompson Sampling 采样）
          </div>
          {renderSlider('graph_affinity_max_hops', '最大跳数', 2, 8, 1)}
        </>
      )}

      {/* ═══ 三锚精排权重 ═══ */}
      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>三锚精排权重（语义 + 声学 + 个性化）</span>
      </div>
      {renderSlider('tri_anchor_w_semantic', '语义相关性（M2D-CLAP）', 0, 1, 0.05)}
      {renderSlider('tri_anchor_w_acoustic', '声学风格（OMAR-RQ）', 0, 1, 0.05)}
      {renderSlider('tri_anchor_w_personal', '个性化偏好（图距离+Jaccard）', 0, 1, 0.05)}
      <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
        权重会自动归一化，无需手动凑和为 1
      </div>

      {/* ═══ 多样性 ═══ */}
      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>多样性控制</span>
      </div>
      {renderSlider('max_songs_per_artist', '每歌手最多曲数', 1, 5, 1)}
      {renderSlider('mmr_lambda', 'MMR 相关性偏好', 0.3, 1, 0.05)}
      <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
        越高越偏向相关性，越低越偏向多样性
      </div>
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
      {renderSlider('context_total_budget', '上下文窗口预算', 2000, 16000, 500, ' tokens')}
      <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
        越大保留越多历史对话，但增加 LLM 调用成本和延迟
      </div>
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
