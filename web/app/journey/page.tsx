'use client';

import { useEffect, useRef, useState } from 'react';
import MainLayout from '@/components/Layout/MainLayout';
import JourneyBuilder from '@/components/Journey/JourneyBuilder';
import JourneySummary from '@/components/Journey/JourneySummary';
import JourneySegments, {
  JourneySegmentState,
} from '@/components/Journey/JourneySegments';
import {
  JourneyRequest,
  SSEEvent,
  streamJourney,
  JourneySegment,
} from '@/lib/api';

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
    setLoading(true);  // ← 立即设为 loading，不等 SSE 第一条事件
    setThinkingMessage('正在排队等待生成...');

    if (!payload.story && !payload.mood_transitions?.length) {
      setError('请提供故事情节或至少一个情绪节点');
      setLoading(false);
      return;
    }

    // 根据输入生成一个简单的旅程标题，让体验更有“故事感”
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

  return (
    <MainLayout>
      <JourneyBuilder loading={loading} onGenerate={handleGenerate} />
      <JourneySummary
        loading={loading}
        thinkingMessage={thinkingMessage}
        journeyTitle={journeyTitle || undefined}
        meta={meta}
        error={error}
      />
      <JourneySegments segments={segments} activeSegmentId={activeSegmentId} />
    </MainLayout>
  );
}

