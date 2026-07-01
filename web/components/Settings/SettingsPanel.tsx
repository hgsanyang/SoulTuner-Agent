'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { theme } from '@/styles/theme';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';

// ---- LLM 提供商预设列表 ----
const LLM_PROVIDERS = [
  { value: 'dashscope', label: '通义千问 DashScope (API)', defaultModel: 'qwen3.7-plus' },
  { value: 'siliconflow', label: 'SiliconFlow (API)', defaultModel: 'deepseek-ai/DeepSeek-V3.2' },
  { value: 'volcengine', label: '火山引擎 / 豆包 (字节跳动)', defaultModel: 'ep-20260405142751-x4jm6' },
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
  dashscope: ['qwen3.7-plus', 'qwen3.7-max', 'qwen3.6-flash', 'qwen3.5-flash', 'deepseek-v3.2'],
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
  const [showAdvancedModels, setShowAdvancedModels] = useState(false);

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
  const renderModelsTab = () => {
    const dashscopeModels = MODEL_PRESETS.dashscope;
    const currentProvider = String(settings.llm_default_provider || 'dashscope');
    const currentModel = String(settings.llm_default_model || 'qwen3.7-plus');

    return (
      <>
        <h4 style={{ color: theme.colors.text.primary, margin: '0 0 1rem', fontSize: '0.95rem' }}>🤖 LLM 模型配置</h4>

        <div style={{
          padding: '0.95rem 1rem',
          border: `1px solid ${theme.colors.border.default}`,
          borderRadius: theme.borderRadius.md,
          background: 'rgba(29,185,84,0.06)',
          marginBottom: '1rem',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center' }}>
            <div>
              <div style={{ color: theme.colors.text.primary, fontWeight: 700, fontSize: '0.9rem' }}>
                DashScope API 部署
              </div>
              <div style={{ color: theme.colors.text.muted, fontSize: '0.74rem', lineHeight: 1.5, marginTop: '0.25rem' }}>
                默认由通义千问驱动意图分析、HyDE 与调音师异步回应。Key 请放在项目 .env 中，前端不会展示密钥。
              </div>
            </div>
            <span style={{
              padding: '0.28rem 0.55rem',
              borderRadius: theme.borderRadius.full,
              color: theme.colors.primary.accent,
              border: '1px solid rgba(29,185,84,0.28)',
              background: 'rgba(29,185,84,0.1)',
              fontSize: '0.72rem',
              whiteSpace: 'nowrap',
            }}>
              {currentProvider === 'dashscope' ? '当前默认' : '已自定义'}
            </span>
          </div>
        </div>

        <div style={fieldGroup}>
          <label style={labelStyle}>主模型</label>
          <select
            style={selectStyle}
            value={dashscopeModels.includes(currentModel) ? currentModel : '__custom__'}
            onChange={e => {
              updateField('llm_default_provider', 'dashscope');
              updateField('intent_llm_provider', 'dashscope');
              updateField('llm_default_model', e.target.value === '__custom__' ? '' : e.target.value);
              updateField('intent_llm_model', e.target.value === '__custom__' ? '' : e.target.value);
            }}
          >
            {dashscopeModels.map(model => <option key={model} value={model}>{model}</option>)}
            <option value="__custom__">自定义 DashScope 模型...</option>
          </select>
          {!dashscopeModels.includes(currentModel) && (
            <input
              style={{ ...inputStyle, marginTop: '0.45rem' }}
              value={currentModel}
              placeholder="例如 qwen3.7-plus"
              onChange={e => {
                updateField('llm_default_provider', 'dashscope');
                updateField('intent_llm_provider', 'dashscope');
                updateField('llm_default_model', e.target.value);
                updateField('intent_llm_model', e.target.value);
              }}
            />
          )}
        </div>

        {renderSelect('explanation_mode', '推荐后回应方式', [
          { value: 'tuner_async', label: '调音师异步回应（默认）' },
          { value: 'off', label: '关闭 LLM 文本，只返回歌单' },
          { value: 'song_detail', label: '旧版逐首推荐解释' },
        ])}
        <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.7rem', marginBottom: '1rem', lineHeight: 1.5 }}>
          默认先返回歌单，再异步生成一段调音师式对话和可选方向；不再默认逐首编写听感解释。
        </div>

        <button
          type="button"
          onClick={() => setShowAdvancedModels(prev => !prev)}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.7rem 0.85rem',
            borderRadius: theme.borderRadius.sm,
            border: `1px solid ${theme.colors.border.default}`,
            background: 'rgba(255,255,255,0.04)',
            color: theme.colors.text.secondary,
            cursor: 'pointer',
            fontSize: '0.82rem',
            marginBottom: showAdvancedModels ? '0.8rem' : 0,
          }}
        >
          <span>高级选项</span>
          <span style={{ color: theme.colors.text.muted }}>{showAdvancedModels ? '收起' : '展开'}</span>
        </button>

        {showAdvancedModels && (
          <div style={{
            border: `1px solid ${theme.colors.border.default}`,
            borderRadius: theme.borderRadius.md,
            padding: '0.8rem 0.9rem',
            background: 'rgba(255,255,255,0.025)',
          }}>
            <div style={sectionTitleStyle}>主模型提供商</div>
            {renderModelPicker('llm_default_provider', 'llm_default_model', '提供商', '模型')}

            <div style={sectionTitleStyle}>意图分析 / HyDE</div>
            {renderModelPicker('intent_llm_provider', 'intent_llm_model', '意图模型提供商', '意图模型', true)}
            {renderModelPicker('hyde_llm_provider', 'hyde_llm_model', 'HyDE 提供商', 'HyDE 模型', true)}

            <div style={sectionTitleStyle}>解释与上下文压缩</div>
            {renderModelPicker('explain_llm_provider', 'explain_llm_model', '调音师回应模型提供商', '调音师回应模型', true)}
            {renderModelPicker('compress_llm_provider', 'compress_llm_model', '压缩模型提供商', '压缩模型', true)}

            <div style={sectionTitleStyle}>调用预算</div>
            {renderToggle('explanation_fast_mode', '兼容低延迟模式（覆盖为关闭文本）')}
            {renderSlider('llm_timeout', 'LLM 超时', 10, 120, 5, '秒')}
            {renderSlider('intent_max_tokens', '意图分析最大输出 Token', 512, 4096, 256, ' tokens')}
            {renderSlider('context_total_budget', '上下文窗口预算', 2000, 16000, 500, ' tokens')}
          </div>
        )}
      </>
    );
  };

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

      {/* ═══ 内容双锚精排权重 ═══ */}
      <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, margin: '1.2rem 0', padding: '1rem 0 0' }}>
        <span style={{ fontSize: '0.8rem', color: theme.colors.text.muted }}>内容双锚权重（语义 + 声学）</span>
      </div>
      {renderSlider('tri_anchor_w_semantic', '语义相关性（M2D-CLAP）', 0, 1, 0.05)}
      {renderSlider('tri_anchor_w_acoustic', '声学风格（OMAR-RQ）', 0, 1, 0.05)}
      <div style={{ fontSize: '0.72rem', color: theme.colors.text.muted, marginTop: '-0.5rem', marginBottom: '1rem' }}>
        权重会自动归一化；个性化只在召回后限幅校正层中轻微加减分
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
