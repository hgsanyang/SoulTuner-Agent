'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import MainLayout from '@/components/Layout/MainLayout';
import WelcomeScreen from '@/components/Content/WelcomeScreen';
import ThinkingIndicator from '@/components/Content/ThinkingIndicator';
import ResultsDisplay from '@/components/Content/ResultsDisplay';
import { streamRecommendations, type SSEEvent } from '@/lib/api';
import { theme } from '@/styles/theme';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  songs?: any[];
  thinkingMessage?: string;
  error?: string;
}

// 可选模型配置
const MODEL_OPTIONS = [
  { provider: 'siliconflow', label: 'SiliconFlow', icon: '⚡' },
  { provider: 'dashscope', label: '通义千问', icon: '🔮' },
  { provider: 'google', label: 'Gemini', icon: '✨' },
  { provider: 'deepseek', label: 'DeepSeek', icon: '🧠' },
];

const STORAGE_KEY = 'music_chat_history';

export default function RecommendationsPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const cancelRef = useRef<(() => void) | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const seedPrompt = searchParams?.get('prompt');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 模型切换和联网开关状态
  const [selectedProvider, setSelectedProvider] = useState('siliconflow');
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const selectedModel = MODEL_OPTIONS.find(m => m.provider === selectedProvider) || MODEL_OPTIONS[0];

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

  // ── 自动滚动到最新消息 ──
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const handleSubmit = useCallback(async (value: string) => {
    setShowModelMenu(false);

    const newMessageId = Date.now().toString();
    const userMsgId = `user-${newMessageId}`;
    const assistantMsgId = `assistant-${newMessageId}`;

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
        llmProvider: selectedProvider,
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
              break;
            case 'song':
              if (event.song) {
                const prevSongs = currentMsg.songs || [];
                const exists = prevSongs.some(s => s.title === event.song?.title && s.artist === event.song?.artist);
                if (!exists) currentMsg.songs = [...prevSongs, event.song];
              }
              break;
            case 'recommendations_complete':
            case 'complete':
              currentMsg.thinkingMessage = undefined;
              if (event.type === 'complete') setLoading(false);
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
  }, [messages, selectedProvider, webSearchEnabled]);

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
  }, []);

  /** 从某条 assistant 消息中删除指定索引的歌曲 */
  const handleRemoveSong = useCallback((msgId: string, songIndex: number) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id !== msgId || !msg.songs) return msg;
      return { ...msg, songs: msg.songs.filter((_, i) => i !== songIndex) };
    }));
  }, []);

  useEffect(() => {
    return () => {
      if (cancelRef.current) cancelRef.current();
    };
  }, []);

  useEffect(() => {
    if (!seedPrompt) return;
    handleSubmit(seedPrompt);
    router.replace(pathname);
  }, [seedPrompt, handleSubmit, router, pathname]);

  const hasMessages = messages.length > 0;

  // 搜索栏上方工具栏
  const toolbar = (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.6rem',
      padding: '0.4rem 0.6rem',
      justifyContent: 'space-between',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
        {/* 模型切换按钮 */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowModelMenu(prev => !prev)}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.4rem',
              padding: '0.35rem 0.8rem',
              borderRadius: '2rem',
              backgroundColor: showModelMenu ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.06)',
              border: `1px solid ${showModelMenu ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.1)'}`,
              color: '#fff', fontSize: '0.82rem', fontWeight: 500,
              cursor: 'pointer', transition: 'all 0.15s', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)')}
            onMouseLeave={e => !showModelMenu && (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
          >
            <span>{selectedModel.icon}</span>
            <span>{selectedModel.label}</span>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ opacity: 0.6, transform: showModelMenu ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {/* 模型下拉菜单 */}
          {showModelMenu && (
            <div style={{
              position: 'absolute', bottom: 'calc(100% + 8px)', left: 0,
              backgroundColor: 'rgba(20,20,20,0.97)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '12px', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
              minWidth: '180px', zIndex: 200, overflow: 'hidden', backdropFilter: 'blur(16px)',
            }}>
              <div style={{ padding: '0.5rem 0.8rem', fontSize: '0.72rem', color: 'rgba(255,255,255,0.4)', borderBottom: '1px solid rgba(255,255,255,0.06)', fontWeight: 600, letterSpacing: '0.06em' }}>
                选择模型
              </div>
              {MODEL_OPTIONS.map(m => (
                <button key={m.provider}
                  onClick={() => { setSelectedProvider(m.provider); setShowModelMenu(false); }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.6rem',
                    width: '100%', padding: '0.6rem 0.8rem',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: m.provider === selectedProvider ? theme.colors.primary.accent : '#fff',
                    fontSize: '0.88rem', textAlign: 'left', transition: 'background-color 0.15s',
                    fontWeight: m.provider === selectedProvider ? 600 : 400,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.07)')}
                  onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                >
                  <span style={{ fontSize: '1rem' }}>{m.icon}</span>
                  <span>{m.label}</span>
                  {m.provider === selectedProvider && (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ marginLeft: 'auto' }}>
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

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

      {/* 右侧：新建聊天按钮（仅在有聊天记录时显示） */}
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
  );

  return (
    <MainLayout
      onInputSubmit={handleSubmit}
      onInputAbort={handleAbort}
      inputPlaceholder="例如：想运动，来点劲爆的"
      inputIsLoading={loading}
      toolbar={toolbar}
    >
      {!hasMessages && !loading && <WelcomeScreen />}

      {hasMessages && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', paddingBottom: '2rem' }}>
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

                  {(msg.content || (msg.songs && msg.songs.length > 0)) && (
                    <ResultsDisplay
                      response={msg.content}
                      songs={msg.songs}
                      onRemoveSong={(songIndex) => handleRemoveSong(msg.id, songIndex)}
                    />
                  )}
                </div>
              )}
            </div>
          ))}
          {/* 用于自动滚动到底的锚点 */}
          <div ref={messagesEndRef} />
        </div>
      )}
    </MainLayout>
  );
}
