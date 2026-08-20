'use client';

/**
 * 全局音乐播放器状态上下文 (PlayerContext)
 * 
 * 作用:
 * 1. 抽离了原本写死在各个 SongCard 中的 <audio> 播放逻辑，使得音频可以在全局（后台）连续播放，即使切换页面也不中断。
 * 2. 维护了全局的播放列表 (queue)、播放模式 (playMode: 单曲循环/顺序/随机) 以及当前播放状态 (isPlaying, currentTime 等)。
 * 3. 供 GlobalPlayer 吸底全局控制器和全屏歌词卡片调用与订阅状态变化。
 */
import React, { createContext, useContext, useState, useRef, useEffect, ReactNode } from 'react';
import { sendUserEvent } from '@/lib/api';
import { useAppSession } from '@/context/AppSessionContext';
import { SessionRequestContext } from '@/lib/app-session';
import { resolveOptionalMediaUrl } from '@/lib/runtime-url';

export interface Song {
    title: string;
    artist: string;
    genre?: string;
    preview_url?: string;
    coverUrl?: string;
    lrc_url?: string;
    exposure_id?: string;
    exposure_rank?: number;
}

export type PlayMode = 'sequence' | 'random' | 'loop';

interface PlayerContextType {
    currentSong: Song | null;
    isPlaying: boolean;
    volume: number;
    duration: number;
    currentTime: number;
    playMode: PlayMode;
    queue: Song[];
    isExpanded: boolean;

    playSong: (song: Song, newQueue?: Song[]) => void;
    togglePlay: () => void;
    playNext: () => void;
    playPrev: () => void;
    setVolume: (v: number) => void;
    seek: (time: number) => void;
    setPlayMode: (mode: PlayMode) => void;
    setExpanded: (expanded: boolean | ((prev: boolean) => boolean)) => void;
    addToQueue: (song: Song) => void;
    removeFromQueue: (title: string, artist: string) => void;
    addAllToQueue: (songs: Song[]) => void;
    replaceQueue: (songs: Song[]) => void;
}

const PlayerContext = createContext<PlayerContextType | undefined>(undefined);

const isSameSong = (a?: Song | null, b?: Song | null) =>
    Boolean(a && b && a.title === b.title && a.artist === b.artist);

const normalizeSongMedia = (song: Song): Song => ({
    ...song,
    preview_url: resolveOptionalMediaUrl(song.preview_url),
    coverUrl: resolveOptionalMediaUrl(song.coverUrl),
    lrc_url: resolveOptionalMediaUrl(song.lrc_url),
});

const MIN_SKIP_LISTEN_MS = 1_000;
const MAX_SKIP_LISTEN_MS = 30_000;
const MAX_SKIP_PROGRESS_RATIO = 0.5;

export function PlayerProvider({ children }: { children: ReactNode }) {
    const { activeProfile, interactionMode, sessionId } = useAppSession();
    const [currentSong, setCurrentSong] = useState<Song | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [volume, setVolumeState] = useState(0.8);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [playMode, setPlayMode] = useState<PlayMode>('sequence');
    const [queue, setQueue] = useState<Song[]>([]);
    const [isExpanded, setExpanded] = useState(false);

    const audioRef = useRef<HTMLAudioElement | null>(null);
    const playbackContextRef = useRef<SessionRequestContext>({
        profileId: activeProfile.profile_id,
        interactionMode,
        sessionId,
    });

    const playbackMetrics = (audio: HTMLAudioElement | null) => {
        const playedSeconds = Math.max(0, Number(audio?.currentTime || 0));
        const durationSeconds = Number(audio?.duration || 0);
        const progressRatio = Number.isFinite(durationSeconds) && durationSeconds > 0
            ? Math.max(0, Math.min(1, playedSeconds / durationSeconds))
            : 0;
        return {
            playDurationMs: Math.round(playedSeconds * 1000),
            progressRatio,
            sessionId: playbackContextRef.current.sessionId,
        };
    };

    const reportPlayback = (
        eventType: 'play_start' | 'skip' | 'full_play' | 'repeat',
        song: Song,
        audio: HTMLAudioElement | null,
    ) => {
        const metrics = playbackMetrics(audio);
        sendUserEvent(eventType, song.title, song.artist, {
            exposureId: song.exposure_id,
            position: song.exposure_rank,
            ...metrics,
            requestContext: playbackContextRef.current,
        });
    };

    const handlePlaybackError = (error: unknown) => {
        console.warn('Audio playback was blocked or failed:', error);
        setIsPlaying(false);
    };

    const startAudio = (audio: HTMLAudioElement) => {
        audio.play().catch(handlePlaybackError);
    };

    useEffect(() => {
        audioRef.current = new Audio();
        audioRef.current.volume = volume;

        const audio = audioRef.current;

        const updateTime = () => setCurrentTime(audio.currentTime);
        const updateDuration = () => setDuration(audio.duration);
        audio.addEventListener('timeupdate', updateTime);
        audio.addEventListener('loadedmetadata', updateDuration);

        return () => {
            audio.removeEventListener('timeupdate', updateTime);
            audio.removeEventListener('loadedmetadata', updateDuration);
            audio.pause();
        };
    }, []); // Only init once

    // Playback events belong to the profile/mode that started the queue. Stop
    // and clear on a context switch so a late "ended" event cannot be written
    // into the newly selected profile.
    useEffect(() => {
        const audio = audioRef.current;
        if (audio) {
            audio.pause();
            audio.removeAttribute('src');
            audio.load();
        }
        setCurrentSong(null);
        setQueue([]);
        setIsPlaying(false);
        setCurrentTime(0);
        setDuration(0);
        playbackContextRef.current = {
            profileId: activeProfile.profile_id,
            interactionMode,
            sessionId,
        };
    }, [activeProfile.profile_id, interactionMode, sessionId]);

    // Reattach ended handler to capture latest playMode state closure
    const handlePlayNext = (isAuto: boolean = false) => {
        if (!currentSong || queue.length === 0) return;

        if (isAuto && playMode === 'loop') {
            if (audioRef.current) {
                audioRef.current.currentTime = 0;
                startAudio(audioRef.current);
            }
            return;
        }

        let nextIndex = queue.findIndex(s => s.title === currentSong.title && s.artist === currentSong.artist);

        if (playMode === 'random') {
            nextIndex = Math.floor(Math.random() * queue.length);
        } else {
            nextIndex = (nextIndex + 1) % queue.length;
        }

        playSong(queue[nextIndex]);
    };

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        audio.onended = () => {
            if (currentSong) {
                reportPlayback(playMode === 'loop' ? 'repeat' : 'full_play', currentSong, audio);
            }
            handlePlayNext(true);
        };
    }, [currentSong, queue, playMode]);

    const playSong = (song: Song, newQueue?: Song[]) => {
        const playableSong = normalizeSongMedia(song);
        const audio = audioRef.current;
        if (isSameSong(currentSong, playableSong) && audio && !audio.ended) {
            setCurrentSong(playableSong);
            if (newQueue) {
                setQueue(newQueue.map(normalizeSongMedia));
            }
            if (!isPlaying && playableSong.preview_url) {
                startAudio(audio);
                setIsPlaying(true);
            }
            return;
        }

        if (currentSong && audio && !audio.ended) {
            const metrics = playbackMetrics(audio);
            if (
                metrics.playDurationMs >= MIN_SKIP_LISTEN_MS
                && metrics.playDurationMs < MAX_SKIP_LISTEN_MS
                && metrics.progressRatio < MAX_SKIP_PROGRESS_RATIO
            ) {
                reportPlayback('skip', currentSong, audio);
            }
        }

        setCurrentSong(playableSong);
        playbackContextRef.current = {
            profileId: activeProfile.profile_id,
            interactionMode,
            sessionId,
        };
        setIsPlaying(true);

        if (newQueue) {
            setQueue(newQueue.map(normalizeSongMedia));
        } else if (queue.length === 0) {
            setQueue([playableSong]);
        }

        if (audioRef.current && playableSong.preview_url) {
            audioRef.current.preload = 'metadata';
            audioRef.current.src = playableSong.preview_url;
            startAudio(audioRef.current);
            reportPlayback('play_start', playableSong, audioRef.current);
        } else if (audioRef.current && !playableSong.preview_url) {
            // Stop audio if no preview
            audioRef.current.pause();
            setIsPlaying(false);
        }
    };

    const addToQueue = (song: Song) => {
        const normalized = normalizeSongMedia(song);
        setQueue(prev => {
            const exists = prev.some(s => s.title === normalized.title && s.artist === normalized.artist);
            if (exists) return prev;
            return [...prev, normalized];
        });
    };

    const removeFromQueue = (title: string, artist: string) => {
        setQueue(prev => prev.filter(s => !(s.title === title && s.artist === artist)));
    };

    const addAllToQueue = (songs: Song[]) => {
        setQueue(prev => {
            const existing = new Set(prev.map(s => `${s.title}_${s.artist}`));
            const newSongs = songs
                .map(normalizeSongMedia)
                .filter(s => !existing.has(`${s.title}_${s.artist}`));
            return [...prev, ...newSongs];
        });
    };

    const replaceQueue = (songs: Song[]) => {
        setQueue(songs.map(normalizeSongMedia));
    };

    const togglePlay = () => {
        if (!audioRef.current || !currentSong || !currentSong.preview_url) return;
        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            startAudio(audioRef.current);
            setIsPlaying(true);
        }
    };

    const playNext = () => handlePlayNext(false);

    const playPrev = () => {
        if (!currentSong || queue.length === 0) return;
        let prevIndex = queue.findIndex(s => s.title === currentSong.title && s.artist === currentSong.artist);

        if (playMode === 'random') {
            prevIndex = Math.floor(Math.random() * queue.length);
        } else {
            prevIndex = (prevIndex - 1 + queue.length) % queue.length;
        }

        playSong(queue[prevIndex]);
    };

    const setVolume = (v: number) => {
        setVolumeState(v);
        if (audioRef.current) {
            audioRef.current.volume = v;
        }
    };

    const seek = (time: number) => {
        if (audioRef.current) {
            audioRef.current.currentTime = time;
            setCurrentTime(time);
        }
    };

    return (
        <PlayerContext.Provider
            value={{
                currentSong,
                isPlaying,
                volume,
                duration,
                currentTime,
                playMode,
                queue,
                isExpanded,
                playSong,
                togglePlay,
                playNext,
                playPrev,
                setVolume,
                seek,
                setPlayMode,
                setExpanded,
                addToQueue,
                removeFromQueue,
                addAllToQueue,
                replaceQueue,
            }}
        >
            {children}
        </PlayerContext.Provider>
    );
}

export const usePlayer = () => {
    const context = useContext(PlayerContext);
    if (context === undefined) {
        throw new Error('usePlayer must be used within a PlayerProvider');
    }
    return context;
};
