'use client';

import { useEffect, useRef, useState } from 'react';
import MainLayout from '@/components/Layout/MainLayout';
import JourneyBuilder from '@/components/Journey/JourneyBuilder';
import JourneySummary from '@/components/Journey/JourneySummary';
import JourneyTimeline from '@/components/Journey/JourneyTimeline';
import JourneySegments, {
  JourneySegmentState,
} from '@/components/Journey/JourneySegments';
import {
  JourneyRequest,
  SSEEvent,
  streamJourney,
  JourneySegment,
} from '@/lib/api';
import { theme } from '@/styles/theme';

function createSegmentState(segment: JourneySegment, status: JourneySegmentState['status']): JourneySegmentState {
  return {
    ...segment,
    songs: segment.songs ?? [],
    status,
  };
}

export default function JourneyPage() {
  const [loading, setLoading] = useState(false);
  const [thinkingMessage, setThinkingMessage] = useState('');
  const [meta, setMeta] = useState<{
    total_segments?: number;
    total_duration?: number;
    total_songs?: number;
  } | null>(null);
  const [journeyTitle, setJourneyTitle] = useState<string | null>(null);
  const [segments, setSegments] = useState<JourneySegmentState[]>([]);
  const [activeSegmentId, setActiveSegmentId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const cleanup = () => {
    if (cancelRef.current) {
      cancelRef.current();
      cancelRef.current = null;
    }
  };

  useEffect(() => cleanup, []);

  const handleSegmentClick = (segmentId: number) => {
    const el = document.getElementById(`journey-segment-${segmentId}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const handleJourneyEvent = (event: SSEEvent) => {
    switch (event.type) {
      case 'journey_start':
        setThinkingMessage(event.message || '正在准备音乐旅程...');
        setLoading(true);
        break;
      case 'thinking':
        setThinkingMessage(event.message || '正在分析故事与意图...');
        break;
      case 'journey_info':
        setMeta({
          total_segments: event.total_segments,
          total_duration: event.total_duration,
          total_songs: event.total_songs,
        });
        break;
      case 'segment_start':
        if (event.segment) {
          setSegments((prev) => {
            const existing = prev.find((seg) => seg.segment_id === event.segment?.segment_id);
            const updatedSegment = createSegmentState(event.segment!, 'active');
            if (existing) {
              return prev.map((seg) =>
                seg.segment_id === updatedSegment.segment_id
                  ? { ...seg, ...updatedSegment }
                  : seg
              );
            }
            return [...prev, updatedSegment].sort((a, b) => a.segment_id - b.segment_id);
          });
          setActiveSegmentId(event.segment.segment_id);
          setThinkingMessage(`正在生成「${event.segment.mood}」阶段...`);
        }
        break;
      case 'song':
        if (typeof event.segment_id === 'number' && event.song) {
          setSegments((prev) =>
            prev.map((segment) => {
              if (segment.segment_id !== event.segment_id) return segment;
              const exists = segment.songs?.some(
                (s) => s.title === event.song?.title && s.artist === event.song?.artist
              );
              if (exists) return segment;
              return {
                ...segment,
                songs: [...(segment.songs || []), event.song],
              };
            })
          );
        }
        break;
      case 'segment_complete':
        if (typeof event.segment_id === 'number') {
          setSegments((prev) =>
            prev.map((segment) =>
              segment.segment_id === event.segment_id
                ? { ...segment, status: 'complete' }
                : segment
            )
          );
          setThinkingMessage('正在准备下一个阶段...');
        }
        break;
      case 'transition_point':
        if (typeof event.to_segment === 'number') {
          setActiveSegmentId(event.to_segment);
        }
        break;
      case 'journey_complete':
      case 'complete':
        setLoading(false);
        setThinkingMessage('旅程生成完成 ✅');
        if (event.result) {
          setMeta((prev) => ({
            total_segments: event.result?.segments?.length || prev?.total_segments,
            total_duration: event.result?.total_duration ?? prev?.total_duration,
            total_songs: event.result?.total_songs ?? prev?.total_songs,
          }));
          if (event.result?.segments?.length) {
            setSegments(
              event.result.segments.map((seg) => ({
                ...seg,
                songs: seg.songs || [],
                status: 'complete',
              }))
            );
          }
        }
        cleanup();
        break;
      case 'error':
        setError(event.error || '旅程生成失败，请稍后重试。');
        setLoading(false);
        setThinkingMessage('');
        cleanup();
        break;
      default:
        break;
    }
  };

  const handleGenerate = (payload: JourneyRequest) => {
    setError(null);
    setMeta(null);
    setJourneyTitle(null);
    setSegments([]);
    setActiveSegmentId(null);
    setLoading(true);
    setThinkingMessage('正在排队等待生成...');

    if (!payload.story && !payload.mood_transitions?.length) {
      setError('请提供故事情节或至少一个情绪节点');
      setLoading(false);
      return;
    }

    if (payload.story) {
      const raw = payload.story.trim();
      const firstStage =
        raw.split(/→|->|－|—/)[0]?.trim() ||
        (raw.length > 16 ? `${raw.slice(0, 16)}…` : raw);
      setJourneyTitle(firstStage ? `${firstStage} 的音乐旅程` : '你的音乐旅程');
    } else if (payload.mood_transitions?.length) {
      setJourneyTitle('情绪曲线驱动的音乐旅程');
    }

    cleanup();
    const cancel = streamJourney(payload, handleJourneyEvent);
    cancelRef.current = cancel;
  };

  const hasResults = segments.length > 0 || loading || meta !== null || error !== null;

  return (
    <MainLayout>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(360px, 35%) 1fr',
          gap: '1.25rem',
          height: '100%',
          minHeight: 0,
        }}
      >
        {/* Left panel — sticky input controls */}
        <div
          style={{
            overflowY: 'auto',
            padding: '1.25rem',
            borderRadius: theme.borderRadius.lg,
            border: `1px solid ${theme.colors.border.default}`,
            backgroundColor: theme.colors.background.card,
          }}
        >
          <JourneyBuilder loading={loading} onGenerate={handleGenerate} />
        </div>

        {/* Right panel — results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto', minHeight: 0 }}>
          {hasResults ? (
            <>
              <JourneySummary
                loading={loading}
                thinkingMessage={thinkingMessage}
                journeyTitle={journeyTitle || undefined}
                meta={meta}
                error={error}
              />
              <JourneyTimeline
                segments={segments}
                activeSegmentId={activeSegmentId}
                onSegmentClick={handleSegmentClick}
              />
              <JourneySegments segments={segments} activeSegmentId={activeSegmentId} />
            </>
          ) : (
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: theme.borderRadius.lg,
                border: `1px dashed ${theme.colors.border.default}`,
                backgroundColor: 'rgba(255,255,255,0.02)',
                gap: '0.75rem',
                padding: '3rem',
              }}
            >
              <div style={{ fontSize: '3.5rem', opacity: 0.3 }}>🗺️</div>
              <h3 style={{ margin: 0, color: theme.colors.text.secondary, fontSize: '1.15rem', fontWeight: 600 }}>
                你的音乐旅程将在这里展开
              </h3>
              <p style={{ margin: 0, color: theme.colors.text.muted, fontSize: '0.88rem', textAlign: 'center', maxWidth: '360px' }}>
                在左侧输入一段故事情节，或切换到情绪曲线模式设置节点，然后点击生成旅程
              </p>
            </div>
          )}
        </div>
      </div>
    </MainLayout>
  );
}

