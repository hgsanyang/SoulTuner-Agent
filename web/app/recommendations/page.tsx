'use client';

// 此页面使用 useSearchParams，不能静态预渲染
export const dynamic = 'force-dynamic';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import MainLayout from '@/components/Layout/MainLayout';
import WelcomeScreen from '@/components/Content/WelcomeScreen';
import ThinkingIndicator from '@/components/Content/ThinkingIndicator';
import SongCard from '@/components/Content/SongCard';
import {
  fetchCatalogDiagnostics,
  sendSlateFeedback,
  streamRecommendations,
  type CatalogDiagnostics,
  type RefinementOption,
  type SSEEvent,
  type SlateFeedbackRating,
} from '@/lib/api';
import { theme } from '@/styles/theme';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  songs?: any[];
  thinkingMessage?: string;
  clarificationOptions?: string[];
  refinementOptions?: RefinementOption[];
  intentConfidence?: number;
  exposureId?: string;
  error?: string;
}

// 可选模型配置
// 模型选择已统一由设置面板（SettingsPanel）管理，不再在聊天页快捷切换

const STORAGE_KEY = 'music_chat_history';
const DIALOG_STATE_KEY = 'music_dialog_state';

const SLATE_FEEDBACK_OPTIONS: { rating: SlateFeedbackRating; label: string }[] = [
  { rating: 'great', label: '整体很准' },
  { rating: 'partial', label: '有几首不错' },
  { rating: 'off', label: '不太对味' },
  { rating: 'too_familiar', label: '太像旧歌单' },
  { rating: 'more_discovery', label: '想发现更多' },
];

const SLATE_REASON_OPTIONS = [
  '太像我的旧歌单',
  '想发现更多新歌',
  '太吵了',
  '太安静了',
  '场景不贴合',
  '联网多一点',
  '少推荐已收藏',
  '语言/年代不准',
];

export default function RecommendationsPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [dialogState, setDialogState] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const { playSong } = usePlayer();
  const { showToast } = useLibrary();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const seedPrompt = searchParams?.get('prompt');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const autoFollowMessagesRef = useRef(true);
  const songListRef = useRef<HTMLDivElement>(null);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [slateFeedbackStatus, setSlateFeedbackStatus] = useState<Record<string, string>>({});
  const [slateFeedbackReasons, setSlateFeedbackReasons] = useState<Record<string, string[]>>({});
  const [slateFeedbackNote, setSlateFeedbackNote] = useState('');
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [diagnostics, setDiagnostics] = useState<CatalogDiagnostics | null>(null);

  // 联网搜索开关状态（持久化到 localStorage）
  const [webSearchEnabled, setWebSearchEnabled] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('music_web_search_enabled');
      return saved !== null ? saved === 'true' : true;
    }
    return true;
  });



  // 持久化联网搜索开关
  useEffect(() => {
    localStorage.setItem('music_web_search_enabled', String(webSearchEnabled));
  }, [webSearchEnabled]);



  // ── 持久化：从 localStorage 加载聊天记录 ──
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed: ChatMessage[] = JSON.parse(saved);
        // 清除 thinkingMessage（上次可能是加载中态），保留 error
        const cleaned = parsed.map(m => ({ ...m, thinkingMessage: undefined }));
        setMessages(cleaned);
      }
      const savedState = localStorage.getItem(DIALOG_STATE_KEY);
      if (savedState) setDialogState(JSON.parse(savedState));
    } catch { /* 忽略解析错误 */ }
  }, []);

  // ── 持久化：保存聊天记录到 localStorage（排除正在思考的消息）──
  useEffect(() => {
    if (messages.length === 0) return;
    try {
      // 只保存已完成的消息（thinkingMessage 已清除）
      const toSave = messages.filter(m => !m.thinkingMessage || m.error);
      if (toSave.length > 0) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
      }
    } catch { /* 忽略 */ }
  }, [messages]);

  const scrollMessagesToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior, block: 'end' });
  }, []);

  const handleMessagesScroll = useCallback(() => {
    const el = messagesScrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom < 120;
    autoFollowMessagesRef.current = nearBottom;
    setShowJumpToLatest(!nearBottom);
  }, []);

  const jumpToLatest = useCallback(() => {
    autoFollowMessagesRef.current = true;
    setShowJumpToLatest(false);
    scrollMessagesToBottom();
  }, [scrollMessagesToBottom]);

  // ── 自动滚动到最新消息：用户主动上滑后暂停跟随 ──
  useEffect(() => {
    if (!autoFollowMessagesRef.current) return;
    requestAnimationFrame(() => scrollMessagesToBottom());
  }, [messages, scrollMessagesToBottom]);

  // ── 只显示最新一条 assistant 消息的歌曲 ──
  const latestAssistantWithSongs = [...messages]
    .reverse()
    .find(m => m.role === 'assistant' && m.songs && m.songs.length > 0);
  const allSongs = latestAssistantWithSongs
    ? (latestAssistantWithSongs.songs || []).map((song: any, idx: number) => ({ song, msgId: latestAssistantWithSongs.id, idx }))
    : [];
  const queueSongs = allSongs
    .filter(s => s.song.preview_url)
    .map(s => ({
      title: s.song.title,
      artist: s.song.artist,
      genre: s.song.genre,
      preview_url: s.song.preview_url,
      coverUrl: s.song.cover_url,
      lrc_url: s.song.lrc_url,
      exposure_id: s.song.exposure_id,
      exposure_rank: s.song.exposure_rank,
    }));
  const latestExposureId = latestAssistantWithSongs?.exposureId
    || latestAssistantWithSongs?.songs?.find((song: any) => song?.exposure_id)?.exposure_id
    || '';

  // 当最新推荐歌曲变化时，右侧面板自动滚到顶部
  useEffect(() => {
    if (allSongs.length > 0 && songListRef.current) {
      songListRef.current.scrollTop = 0;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestAssistantWithSongs?.id, allSongs.length]);

  const handleSubmit = useCallback(async (value: string) => {

    const newMessageId = Date.now().toString();
    const userMsgId = `user-${newMessageId}`;
    const assistantMsgId = `assistant-${newMessageId}`;

    autoFollowMessagesRef.current = true;
    setShowJumpToLatest(false);

    // ① 先取当前历史（在更新 state 之前读）
    const chatHistorySnapshot = messages
      .filter(m => !m.error && !m.thinkingMessage)
      .map(m => ({ role: m.role, content: m.content }));

    // ② 先中止旧搜索，再启动新搜索
    if (cancelRef.current) {
      cancelRef.current();
      cancelRef.current = null;
    }

    setLoading(true);
    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: value },
      { id: assistantMsgId, role: 'assistant', content: '', songs: [], thinkingMessage: '开始分析你的需求...' }
    ]);

    // ③ 启动新的 SSE 流
    const cancel = streamRecommendations(
      {
        query: value,
        chatHistory: chatHistorySnapshot,
        dialogState,
        webSearchEnabled,
      },
      (event: SSEEvent) => {
        setMessages((prev) => {
          const newMessages = [...prev];
          const lastIdx = newMessages.findLastIndex(m => m.id === assistantMsgId);
          if (lastIdx < 0) return prev;

          const currentMsg = { ...newMessages[lastIdx] };

          switch (event.type) {
            case 'start':
            case 'thinking':
              currentMsg.thinkingMessage = event.message || '正在思考...';
              break;
            case 'response':
              if (event.text) {
                currentMsg.content = event.text;
                if (event.is_complete) currentMsg.thinkingMessage = undefined;
              }
              break;
            case 'recommendations_start':
              currentMsg.thinkingMessage = '正在获取推荐歌曲...';
              currentMsg.songs = [];
              currentMsg.exposureId = event.exposure_id;
              break;
            case 'song':
              if (event.song) {
                const prevSongs = currentMsg.songs || [];
                const exists = prevSongs.some(s => s.title === event.song?.title && s.artist === event.song?.artist);
                if (!exists) {
                  currentMsg.songs = [
                    ...prevSongs,
                    {
                      ...event.song,
                      exposure_id: event.exposure_id || currentMsg.exposureId,
                      exposure_rank: typeof event.index === 'number' ? event.index + 1 : prevSongs.length + 1,
                    },
                  ];
                }
              }
              break;
            case 'recommendations_complete':
              break;
            case 'clarification_required':
              currentMsg.content = event.text || '我需要再确认一下你的意思。';
              currentMsg.clarificationOptions = event.clarification_options || [];
              currentMsg.thinkingMessage = undefined;
              break;
            case 'complete':
              currentMsg.thinkingMessage = undefined;
              if (event.type === 'complete') {
                if (event.dialog_state) {
                  setDialogState(event.dialog_state);
                  try { localStorage.setItem(DIALOG_STATE_KEY, JSON.stringify(event.dialog_state)); } catch { /* ignore */ }
                }
                if (event.exposure_id) currentMsg.exposureId = event.exposure_id;
                if (typeof event.intent_confidence === 'number') {
                  currentMsg.intentConfidence = event.intent_confidence;
                }
                if (event.clarification_options && event.clarification_options.length > 0) {
                  currentMsg.content = currentMsg.content || '我需要再确认一下你的意思。';
                  currentMsg.clarificationOptions = event.clarification_options;
                }
                if (event.refinement_options && event.refinement_options.length > 0) {
                  currentMsg.refinementOptions = event.refinement_options;
                }
                setLoading(false);
              }
              break;
            case 'error':
              currentMsg.error = event.error || '发生未知错误';
              currentMsg.thinkingMessage = undefined;
              setLoading(false);
              break;
          }
          newMessages[lastIdx] = currentMsg;
          return newMessages;
        });
      }
    );

    cancelRef.current = cancel;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, webSearchEnabled, dialogState]);

  /** 中止当前搜索，立即允许新搜索 */
  const handleAbort = useCallback(() => {
    if (cancelRef.current) {
      cancelRef.current();
      cancelRef.current = null;
    }
    setLoading(false);
    setMessages(prev => {
      const newMsgs = [...prev];
      const lastAssistantIdx = [...newMsgs].reverse().findIndex(m => m.role === 'assistant');
      if (lastAssistantIdx >= 0) {
        const realIdx = newMsgs.length - 1 - lastAssistantIdx;
        newMsgs[realIdx] = {
          ...newMsgs[realIdx],
          thinkingMessage: undefined,
          content: newMsgs[realIdx].content || '搜索已被中止',
        };
      }
      return newMsgs;
    });
  }, []);

  /** 新建聊天：清空当前会话（localStorage 也清除） */
  const handleNewChat = useCallback(() => {
    if (cancelRef.current) {
      cancelRef.current();
      cancelRef.current = null;
    }
    setLoading(false);
    setMessages([]);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
    setDialogState({});
    try { localStorage.removeItem(DIALOG_STATE_KEY); } catch { /* ignore */ }
  }, []);

  /** 从某条 assistant 消息中删除指定索引的歌曲 */
  const handleRemoveSong = useCallback((msgId: string, songIndex: number) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id !== msgId || !msg.songs) return msg;
      return { ...msg, songs: msg.songs.filter((_, i) => i !== songIndex) };
    }));
  }, []);

  const toggleSlateReason = useCallback((exposureId: string, reason: string) => {
    if (!exposureId) return;
    setSlateFeedbackReasons(prev => {
      const current = prev[exposureId] || [];
      const next = current.includes(reason)
        ? current.filter(item => item !== reason)
        : [...current, reason];
      return { ...prev, [exposureId]: next };
    });
  }, []);

  const handleSlateFeedback = useCallback(async (rating: SlateFeedbackRating) => {
    if (!latestExposureId) {
      showToast('还没有可评价的推荐批次');
      return;
    }
    try {
      await sendSlateFeedback({
        exposureId: latestExposureId,
        rating,
        reasons: slateFeedbackReasons[latestExposureId] || [],
        note: slateFeedbackNote,
        extra: {
          song_count: allSongs.length,
          web_search_enabled: webSearchEnabled,
        },
      });
      setSlateFeedbackStatus(prev => ({ ...prev, [latestExposureId]: rating }));
      setSlateFeedbackNote('');
      showToast('✅ 已记录这组推荐的反馈');
    } catch (err: any) {
      showToast(`⚠️ 反馈记录失败：${err?.message || '未知错误'}`);
    }
  }, [allSongs.length, latestExposureId, showToast, slateFeedbackNote, slateFeedbackReasons, webSearchEnabled]);

  const loadDiagnostics = useCallback(async () => {
    setDiagnosticsOpen(true);
    setDiagnosticsLoading(true);
    try {
      const report = await fetchCatalogDiagnostics(50);
      setDiagnostics(report);
    } catch (err: any) {
      setDiagnostics({ success: false, error: err?.message || '曲库诊断失败' });
    } finally {
      setDiagnosticsLoading(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (cancelRef.current) cancelRef.current();
    };
  }, []);

  // 从 sessionStorage 读取预设 prompt（由首页写入）
  // React 18 StrictMode 会 mount→unmount→remount，不能在 cleanup 中 clearTimeout
  const seedExecutedRef = useRef(false);
  useEffect(() => {
    if (seedExecutedRef.current) return;
    try {
      const prompt = sessionStorage.getItem('seed_prompt');
      if (!prompt) return;
      sessionStorage.removeItem('seed_prompt');
      seedExecutedRef.current = true;
      setTimeout(() => {
        handleSubmit(prompt);
      }, 300);
    } catch { /* SSR 环境无 sessionStorage */ }
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const hasMessages = messages.length > 0;

  // 临时返回情绪卡片（不清空聊天记录）
  const [showWelcome, setShowWelcome] = useState(false);

  /** 返回推荐卡片视图（保留聊天历史，下次发送时自动回到对话视图） */
  const handleBackToCards = useCallback(() => {
    setShowWelcome(true);
  }, []);

  /** 包装 handleSubmit：从卡片/输入框发送时自动回到对话视图 */
  const handleSubmitAndHideWelcome = useCallback(async (value: string) => {
    setShowWelcome(false);
    return handleSubmit(value);
  }, [handleSubmit]);

  const handleChipSubmit = useCallback((prompt: string) => {
    if (!prompt.trim() || loading) return;
    setShowWelcome(false);
    handleSubmit(prompt);
  }, [handleSubmit, loading]);

  const toolbar = (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.6rem',
      padding: '0.4rem 0.6rem',
      justifyContent: 'space-between',
      marginBottom: '0.75rem', // Added explicit margin-bottom here to separate from input
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>

        {/* 联网搜索开关 */}
        <button
          onClick={() => setWebSearchEnabled(prev => !prev)}
          style={{
            display: 'flex', alignItems: 'center', gap: '0.4rem',
            padding: '0.35rem 0.8rem', borderRadius: '2rem',
            backgroundColor: webSearchEnabled ? 'rgba(29,185,84,0.12)' : 'rgba(255,255,255,0.06)',
            border: `1px solid ${webSearchEnabled ? 'rgba(29,185,84,0.35)' : 'rgba(255,255,255,0.1)'}`,
            color: webSearchEnabled ? theme.colors.primary.accent : 'rgba(255,255,255,0.45)',
            fontSize: '0.82rem', fontWeight: 500, cursor: 'pointer', transition: 'all 0.2s', whiteSpace: 'nowrap',
          }}
          onMouseEnter={e => (e.currentTarget.style.opacity = '0.8')}
          onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
          title={webSearchEnabled ? '关闭联网搜索' : '开启联网搜索'}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" /><line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          联网搜索
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: webSearchEnabled ? theme.colors.primary.accent : 'rgba(255,255,255,0.25)', transition: 'background-color 0.2s', display: 'inline-block' }} />
        </button>
      </div>

      {/* 右侧按钮组 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {/* 返回推荐卡片按钮（仅在对话视图时显示） */}
        {hasMessages && !showWelcome && (
          <button
            onClick={handleBackToCards}
            title="返回场景推荐卡片"
            style={{
              display: 'flex', alignItems: 'center', gap: '0.4rem',
              padding: '0.35rem 0.8rem', borderRadius: '2rem',
              backgroundColor: 'rgba(99, 102, 241, 0.1)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              color: 'rgba(165, 165, 255, 0.85)', fontSize: '0.82rem',
              cursor: 'pointer', transition: 'all 0.2s', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.2)'; e.currentTarget.style.color = '#fff'; }}
            onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'rgba(99, 102, 241, 0.1)'; e.currentTarget.style.color = 'rgba(165, 165, 255, 0.85)'; }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            推荐卡片
          </button>
        )}
        {/* 返回对话按钮（在卡片视图且有历史消息时显示） */}
        {hasMessages && showWelcome && (
          <button
            onClick={() => setShowWelcome(false)}
            title="返回对话记录"
            style={{
              display: 'flex', alignItems: 'center', gap: '0.4rem',
              padding: '0.35rem 0.8rem', borderRadius: '2rem',
              backgroundColor: 'rgba(29, 185, 84, 0.1)',
              border: '1px solid rgba(29, 185, 84, 0.25)',
              color: 'rgba(29, 185, 84, 0.85)', fontSize: '0.82rem',
              cursor: 'pointer', transition: 'all 0.2s', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { e.currentTarget.style.backgroundColor = 'rgba(29, 185, 84, 0.2)'; e.currentTarget.style.color = '#fff'; }}
            onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'rgba(29, 185, 84, 0.1)'; e.currentTarget.style.color = 'rgba(29, 185, 84, 0.85)'; }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            返回对话
          </button>
        )}
        {/* 新建聊天按钮 */}
        {hasMessages && (
          <button
            onClick={handleNewChat}
            title="清空当前对话，开始新的聊天"
            style={{
              display: 'flex', alignItems: 'center', gap: '0.4rem',
              padding: '0.35rem 0.8rem', borderRadius: '2rem',
              backgroundColor: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              color: 'rgba(255,255,255,0.6)', fontSize: '0.82rem',
              cursor: 'pointer', transition: 'all 0.2s', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { e.currentTarget.style.backgroundColor = 'rgba(255,80,60,0.15)'; e.currentTarget.style.color = '#fff'; e.currentTarget.style.borderColor = 'rgba(255,80,60,0.35)'; }}
            onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)'; e.currentTarget.style.color = 'rgba(255,255,255,0.6)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            新建聊天
          </button>
        )}
      </div>
    </div>
  );

  return (
    <MainLayout
      onInputSubmit={handleSubmitAndHideWelcome}
      onInputAbort={handleAbort}
      inputPlaceholder="例如：想运动，来点劲爆的"
      inputIsLoading={loading}
      toolbar={toolbar}
    >
      {(!hasMessages || showWelcome) && !loading && <WelcomeScreen onPromptClick={handleSubmitAndHideWelcome} />}

      {hasMessages && !showWelcome && (() => {

        return (
          <div style={{
            display: 'flex',
            gap: '1.25rem',
            width: '100%',
            minHeight: 0,
            flex: 1,
            overflow: 'hidden',
          }}>
            {/* ── 左栏：对话记录（玻璃容器 + 可滚动） ── */}
            <div style={{
              flex: 1,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: 'rgba(36, 36, 36, 0.5)',
              backdropFilter: 'blur(16px)',
              borderRadius: '1rem',
              border: `1px solid ${theme.colors.border.default}`,
              overflow: 'hidden',
              minHeight: 0,
              position: 'relative',
            }}>
              {/* 对话标题栏 */}
              <div style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '0.85rem 1.25rem',
                borderBottom: `1px solid ${theme.colors.border.default}`,
                flexShrink: 0,
              }}>
                <h2 style={{ fontSize: '1rem', fontWeight: 600, color: theme.colors.text.primary, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(29,185,84,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
                  对话记录
                </h2>
              </div>
              {/* 对话滚动区 */}
              <div
                ref={messagesScrollRef}
                onScroll={handleMessagesScroll}
                style={{
                flex: 1,
                overflowY: 'auto',
                padding: '1rem 1.25rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem',
              }}>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex', flexDirection: 'column',
                    alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
                  }}
                >
                  {msg.role === 'user' ? (
                    <div style={{
                      backgroundColor: '#2a2a2a', padding: '0.85rem 1.2rem',
                      borderRadius: '1.1rem 1.1rem 0.3rem 1.1rem',
                      maxWidth: '85%', color: '#fff', lineHeight: '1.6',
                      border: '1px solid rgba(255,255,255,0.08)',
                    }}>
                      {msg.content}
                    </div>
                  ) : (
                    <div style={{ width: '100%' }}>
                      {msg.thinkingMessage && <ThinkingIndicator message={msg.thinkingMessage} />}

                      {msg.error && (
                        <div style={{
                          padding: '1rem', margin: '1rem 0',
                          backgroundColor: 'rgba(255,50,50,0.08)',
                          color: '#ff6b6b', borderRadius: '0.75rem',
                          border: '1px solid rgba(255,50,50,0.2)',
                        }}>
                          {msg.error}
                        </div>
                      )}

                      {/* 只显示 AI 文字回复，歌曲列表移到右栏 */}
                      {msg.content && (
                        <div style={{
                          padding: '1.25rem',
                          backgroundColor: theme.colors.background.card,
                          borderRadius: theme.borderRadius.md,
                          border: `1px solid ${theme.colors.border.default}`,
                          boxShadow: theme.shadows.sm,
                        }}>
                          <p style={{ color: theme.colors.text.primary, lineHeight: '1.75', whiteSpace: 'pre-wrap', margin: 0 }}>
                            {msg.content}
                          </p>
                        </div>
                      )}

                      {(msg.clarificationOptions?.length || msg.refinementOptions?.length) && (
                        <div style={{
                          marginTop: '0.75rem',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '0.6rem',
                          maxWidth: '100%',
                        }}>
                          {msg.clarificationOptions && msg.clarificationOptions.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                              {msg.clarificationOptions.map((option) => (
                                <button
                                  key={option}
                                  onClick={() => handleChipSubmit(option)}
                                  disabled={loading}
                                  style={{
                                    padding: '0.45rem 0.75rem',
                                    borderRadius: '999px',
                                    border: '1px solid rgba(29,185,84,0.35)',
                                    backgroundColor: 'rgba(29,185,84,0.12)',
                                    color: 'rgba(245,255,248,0.92)',
                                    fontSize: '0.82rem',
                                    cursor: loading ? 'not-allowed' : 'pointer',
                                    transition: 'all 0.18s ease',
                                    whiteSpace: 'nowrap',
                                  }}
                                  onMouseEnter={e => { if (!loading) e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.2)'; }}
                                  onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.12)'; }}
                                >
                                  {option}
                                </button>
                              ))}
                            </div>
                          )}

                          {msg.refinementOptions && msg.refinementOptions.length > 0 && (
                            <div style={{
                              display: 'flex',
                              flexWrap: 'wrap',
                              gap: '0.5rem',
                              paddingLeft: '0.1rem',
                            }}>
                              {msg.refinementOptions.map((option) => (
                                <button
                                  key={`${option.label}-${option.prompt}`}
                                  onClick={() => handleChipSubmit(option.prompt || option.label)}
                                  disabled={loading}
                                  title={option.reason || option.prompt}
                                  style={{
                                    padding: '0.44rem 0.78rem',
                                    borderRadius: '999px',
                                    border: '1px solid rgba(91, 214, 170, 0.34)',
                                    backgroundColor: 'rgba(42, 170, 130, 0.16)',
                                    color: 'rgba(238, 255, 248, 0.9)',
                                    fontSize: '0.81rem',
                                    fontWeight: 520,
                                    cursor: loading ? 'not-allowed' : 'pointer',
                                    transition: 'all 0.18s ease',
                                    whiteSpace: 'nowrap',
                                    boxShadow: '0 6px 18px rgba(0,0,0,0.12)',
                                  }}
                                  onMouseEnter={e => {
                                    if (!loading) {
                                      e.currentTarget.style.backgroundColor = 'rgba(54, 190, 146, 0.24)';
                                      e.currentTarget.style.borderColor = 'rgba(112, 235, 190, 0.48)';
                                      e.currentTarget.style.color = '#fff';
                                    }
                                  }}
                                  onMouseLeave={e => {
                                    e.currentTarget.style.backgroundColor = 'rgba(42, 170, 130, 0.16)';
                                    e.currentTarget.style.borderColor = 'rgba(91, 214, 170, 0.34)';
                                    e.currentTarget.style.color = 'rgba(238, 255, 248, 0.9)';
                                  }}
                                >
                                  {option.label}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
              </div>
              {showJumpToLatest && (
                <button
                  onClick={jumpToLatest}
                  style={{
                    position: 'absolute',
                    right: '1rem',
                    bottom: '1rem',
                    zIndex: 4,
                    padding: '0.45rem 0.75rem',
                    borderRadius: '999px',
                    border: '1px solid rgba(29,185,84,0.3)',
                    backgroundColor: 'rgba(20,20,20,0.86)',
                    color: 'rgba(245,255,248,0.9)',
                    fontSize: '0.78rem',
                    cursor: 'pointer',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.28)',
                    backdropFilter: 'blur(10px)',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.18)')}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(20,20,20,0.86)')}
                >
                  跳到最新
                </button>
              )}
            </div>

            {/* ── 右栏：歌曲列表（可滚动） ── */}
            {allSongs.length > 0 && (
              <div style={{
                width: '560px',
                flexShrink: 0,
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: 'rgba(36, 36, 36, 0.5)',
                backdropFilter: 'blur(16px)',
                borderRadius: '1rem',
                border: `1px solid ${theme.colors.border.default}`,
                overflow: 'hidden',
                minHeight: 0,
              }}>
                {/* 歌曲面板标题栏 */}
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '1rem 1.25rem',
                  borderBottom: `1px solid ${theme.colors.border.default}`,
                  flexShrink: 0,
                }}>
                  <h2 style={{
                    fontSize: '1rem', fontWeight: 600,
                    color: theme.colors.text.primary, margin: 0,
                  }}>
                    推荐歌曲 <span style={{ fontSize: '0.82rem', color: theme.colors.text.muted, fontWeight: 400 }}>({allSongs.length})</span>
                  </h2>
                  <button
                    onClick={() => {
                      if (queueSongs.length === 0) return;
                      playSong(queueSongs[0], queueSongs);
                      showToast(`▶ 已设置 ${queueSongs.length} 首歌为播放列表`);
                    }}
                    title="全部播放"
                    style={{
                      display: 'flex', alignItems: 'center', gap: '0.4rem',
                      padding: '0.3rem 0.7rem', borderRadius: '2rem',
                      border: '1px solid rgba(255,255,255,0.15)',
                      backgroundColor: 'rgba(255,255,255,0.06)',
                      color: 'rgba(255,255,255,0.7)', fontSize: '0.78rem', fontWeight: 500,
                      cursor: 'pointer', transition: 'all 0.2s', whiteSpace: 'nowrap',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)'; }}
                    onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)'; }}
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    全部播放
                  </button>
                </div>

                {/* 歌曲滚动列表 */}
                <div
                  ref={songListRef}
                  style={{
                    flex: 1,
                    overflowY: 'auto',
                    padding: '0.75rem',
                  }}
                >
                  {allSongs.map(({ song, msgId, idx }, i) => (
                    <SongCard
                      key={`${song.title}_${song.artist}_${i}`}
                      {...song}
                      queueContext={queueSongs}
                      onRemove={() => handleRemoveSong(msgId, idx)}
                    />
                  ))}
                </div>
                <div style={{
                  flexShrink: 0,
                  borderTop: `1px solid ${theme.colors.border.default}`,
                  padding: '0.85rem 1rem',
                  backgroundColor: 'rgba(18,18,18,0.72)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.65rem' }}>
                    <div>
                      <div style={{ color: theme.colors.text.primary, fontWeight: 650, fontSize: '0.9rem' }}>这组推荐怎么样？</div>
                      <div style={{ color: theme.colors.text.muted, fontSize: '0.72rem', marginTop: '0.15rem' }}>
                        用于离线重放和排序权重学习，不会写入 GraphZep。
                      </div>
                    </div>
                    <button
                      onClick={loadDiagnostics}
                      style={{
                        padding: '0.38rem 0.7rem',
                        borderRadius: '999px',
                        border: '1px solid rgba(91,214,170,0.28)',
                        backgroundColor: diagnosticsOpen ? 'rgba(42,170,130,0.18)' : 'rgba(255,255,255,0.06)',
                        color: 'rgba(238,255,248,0.88)',
                        fontSize: '0.76rem',
                        cursor: 'pointer',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      曲库诊断
                    </button>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem', marginBottom: '0.55rem' }}>
                    {SLATE_FEEDBACK_OPTIONS.map(option => {
                      const active = latestExposureId && slateFeedbackStatus[latestExposureId] === option.rating;
                      return (
                        <button
                          key={option.rating}
                          onClick={() => handleSlateFeedback(option.rating)}
                          disabled={!latestExposureId}
                          style={{
                            padding: '0.38rem 0.64rem',
                            borderRadius: '999px',
                            border: active ? '1px solid rgba(29,185,84,0.55)' : '1px solid rgba(255,255,255,0.12)',
                            backgroundColor: active ? 'rgba(29,185,84,0.20)' : 'rgba(255,255,255,0.06)',
                            color: active ? '#fff' : 'rgba(255,255,255,0.76)',
                            fontSize: '0.76rem',
                            cursor: latestExposureId ? 'pointer' : 'not-allowed',
                          }}
                        >
                          {option.label}
                        </button>
                      );
                    })}
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.6rem' }}>
                    {SLATE_REASON_OPTIONS.map(reason => {
                      const selected = Boolean(latestExposureId && (slateFeedbackReasons[latestExposureId] || []).includes(reason));
                      return (
                        <button
                          key={reason}
                          onClick={() => toggleSlateReason(latestExposureId, reason)}
                          disabled={!latestExposureId}
                          title="作为这组推荐的补充原因"
                          style={{
                            padding: '0.32rem 0.58rem',
                            borderRadius: '999px',
                            border: selected ? '1px solid rgba(120,210,255,0.44)' : '1px solid rgba(255,255,255,0.10)',
                            backgroundColor: selected ? 'rgba(65,145,210,0.18)' : 'rgba(255,255,255,0.045)',
                            color: selected ? 'rgba(238,250,255,0.95)' : 'rgba(255,255,255,0.58)',
                            fontSize: '0.72rem',
                            cursor: latestExposureId ? 'pointer' : 'not-allowed',
                          }}
                        >
                          {reason}
                        </button>
                      );
                    })}
                  </div>
                  <input
                    value={slateFeedbackNote}
                    onChange={event => setSlateFeedbackNote(event.target.value)}
                    placeholder="可选：哪里不对？例如太像我的收藏、年代不准、想更冷门..."
                    maxLength={240}
                    style={{
                      width: '100%',
                      boxSizing: 'border-box',
                      padding: '0.55rem 0.7rem',
                      borderRadius: '0.55rem',
                      border: '1px solid rgba(255,255,255,0.10)',
                      backgroundColor: 'rgba(0,0,0,0.22)',
                      color: theme.colors.text.primary,
                      outline: 'none',
                      fontSize: '0.76rem',
                    }}
                  />
                  {diagnosticsOpen && (
                    <div style={{
                      marginTop: '0.75rem',
                      padding: '0.75rem',
                      borderRadius: '0.7rem',
                      border: '1px solid rgba(255,255,255,0.10)',
                      backgroundColor: 'rgba(0,0,0,0.20)',
                      color: 'rgba(255,255,255,0.72)',
                      fontSize: '0.76rem',
                      lineHeight: 1.55,
                    }}>
                      {diagnosticsLoading ? (
                        <div>正在读取曲库分布...</div>
                      ) : diagnostics?.success ? (
                        <div>
                          <div style={{ color: '#fff', fontWeight: 650, marginBottom: '0.4rem' }}>
                            曲库 {diagnostics.catalog?.total_songs || 0} 首 · 可播放 {diagnostics.catalog?.playable_songs || 0} 首 · 最近曝光 {diagnostics.recent_recommendations?.exposures || 0} 批
                          </div>
                          <div>语言：{(diagnostics.catalog?.top?.languages || []).slice(0, 4).map(item => `${item.label} ${Math.round(item.ratio * 100)}%`).join(' / ') || '暂无'}</div>
                          <div>流派：{(diagnostics.catalog?.top?.genres || []).slice(0, 5).map(item => `${item.label} ${item.count}`).join(' / ') || '暂无'}</div>
                          <div>最近来源：{(diagnostics.recent_recommendations?.top_recall_sources || []).slice(0, 5).map(item => `${item.label} ${item.count}`).join(' / ') || '暂无曝光'}</div>
                          {(diagnostics.warnings || []).length > 0 && (
                            <div style={{ marginTop: '0.45rem', color: 'rgba(255,220,160,0.92)' }}>
                              {(diagnostics.warnings || []).slice(0, 3).map(w => w.message).join('；')}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{ color: '#ff8f8f' }}>{diagnostics?.error || '曲库诊断失败'}</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })()}
      {/* 用于自动滚动到底的锚点 */}
    </MainLayout>
  );
}
