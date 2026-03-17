'use client';

import { theme } from '@/styles/theme';
import SongCard from './SongCard';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { useState } from 'react';

interface Song {
  title: string;
  artist: string;
  genre?: string;
  mood?: string;
  reason?: string;
  preview_url?: string;
  cover_url?: string;
  lrc_url?: string;
}

interface ResultsDisplayProps {
  response?: string;
  songs?: Song[];
  onRemoveSong?: (index: number) => void;  // 从结果中删除某首歌
}

export default function ResultsDisplay({ response, songs, onRemoveSong }: ResultsDisplayProps) {
  const { addAllToQueue, playSong } = usePlayer();
  const { showToast } = useLibrary();
  const [addedAll, setAddedAll] = useState(false);

  const handleAddAllToQueue = () => {
    if (!songs || songs.length === 0) return;
    const songsWithUrl = songs.filter(s => s.preview_url);
    addAllToQueue(songsWithUrl.map(s => ({ title: s.title, artist: s.artist, genre: s.genre, preview_url: s.preview_url, coverUrl: s.cover_url, lrc_url: s.lrc_url })));
    showToast(`✚ 已将 ${songsWithUrl.length} 首歌加入播放列表`);
    setAddedAll(true);
    // 如果这些歌有 preview_url，自动播放第一首
    if (songsWithUrl.length > 0 && songsWithUrl[0].preview_url) {
      playSong(
        { title: songsWithUrl[0].title, artist: songsWithUrl[0].artist, genre: songsWithUrl[0].genre, preview_url: songsWithUrl[0].preview_url, coverUrl: songsWithUrl[0].cover_url, lrc_url: songsWithUrl[0].lrc_url },
        songsWithUrl.map(s => ({ title: s.title, artist: s.artist, genre: s.genre, preview_url: s.preview_url, coverUrl: s.cover_url, lrc_url: s.lrc_url }))
      );
    }
  };

  return (
    <div style={{ padding: '1.5rem' }}>
      {/* AI 回复文本 */}
      {response && (
        <div style={{
          padding: '1.25rem',
          marginBottom: '1.5rem',
          backgroundColor: theme.colors.background.card,
          borderRadius: theme.borderRadius.md,
          border: `1px solid ${theme.colors.border.default}`,
          boxShadow: theme.shadows.sm,
        }}>
          <p style={{ color: theme.colors.text.primary, lineHeight: '1.75', whiteSpace: 'pre-wrap', margin: 0 }}>
            {response}
          </p>
        </div>
      )}

      {/* 歌曲列表 */}
      {songs && songs.length > 0 && (
        <div>
          {/* 标题行 + 全部加入播放列表按钮 */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
            <h2 style={{
              fontSize: '1.1rem', fontWeight: 600,
              color: theme.colors.text.primary, margin: 0,
            }}>
              推荐歌曲 <span style={{ fontSize: '0.82rem', color: theme.colors.text.muted, fontWeight: 400 }}>({songs.length})</span>
            </h2>

            {/* 全部加入播放列表按钮 */}
            <button
              onClick={handleAddAllToQueue}
              title="将所有推荐加入播放列表并开始播放"
              style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                padding: '0.35rem 0.85rem',
                borderRadius: '2rem',
                border: `1px solid ${addedAll ? 'rgba(29,185,84,0.5)' : 'rgba(255,255,255,0.15)'}`,
                backgroundColor: addedAll ? 'rgba(29,185,84,0.1)' : 'rgba(255,255,255,0.06)',
                color: addedAll ? theme.colors.primary.accent : 'rgba(255,255,255,0.7)',
                fontSize: '0.8rem', fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s',
                whiteSpace: 'nowrap',
              }}
              onMouseEnter={e => { e.currentTarget.style.backgroundColor = addedAll ? 'rgba(29,185,84,0.18)' : 'rgba(255,255,255,0.1)'; }}
              onMouseLeave={e => { e.currentTarget.style.backgroundColor = addedAll ? 'rgba(29,185,84,0.1)' : 'rgba(255,255,255,0.06)'; }}
            >
              {addedAll ? (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
                  已在播放列表
                </>
              ) : (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  全部播放
                </>
              )}
            </button>
          </div>

          {/* 歌曲卡片列表 */}
          {songs.map((song, index) => (
            <SongCard
              key={`${song.title}_${song.artist}_${index}`}
              {...song}
              onRemove={onRemoveSong ? () => onRemoveSong(index) : undefined}
            />
          ))}
        </div>
      )}
    </div>
  );
}
