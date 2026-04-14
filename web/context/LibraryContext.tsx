'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { sendUserEvent, fetchLikedSongs, fetchDislikedSongs, removeDislike as apiRemoveDislike } from '@/lib/api';

export interface LikedSong {
    id: string;
    title: string;
    artist: string;
    genre?: string;
    preview_url?: string;
    coverUrl?: string;
    lrc_url?: string;
    addedAt: number;
}

export interface DislikedSong {
    id: string;
    title: string;
    artist: string;
    coverUrl?: string;
    dislikedAt: number;
}

export interface CollectionSong extends LikedSong { }

export interface Collection {
    id: number;
    name: string;
    coverColor: string;
    songs: CollectionSong[];
}

interface LibraryContextType {
    likedSongs: LikedSong[];
    dislikedSongs: DislikedSong[];
    toggleLike: (song: Omit<LikedSong, 'id' | 'addedAt'>) => void;
    toggleDislike: (song: { title: string; artist: string; coverUrl?: string }) => void;
    undoDislike: (title: string, artist: string) => void;
    isLiked: (title: string, artist: string) => boolean;
    isDisliked: (title: string, artist: string) => boolean;
    // Collections
    collections: Collection[];
    addCollection: (name: string) => number;
    addToCollection: (collectionId: number, song: Omit<CollectionSong, 'id' | 'addedAt'>) => void;
    removeFromCollection: (collectionId: number, songId: string) => void;
    isInCollection: (collectionId: number, title: string, artist: string) => boolean;
    // Sync
    syncFromBackend: () => Promise<void>;
    isSyncing: boolean;
    // Toast
    toast: string | null;
    showToast: (msg: string) => void;
}

const INITIAL_COLLECTIONS: Collection[] = [
    { id: 1, name: '深夜敲代码专用', coverColor: '#5B8DEF', songs: [] },
    { id: 2, name: '雨天治愈系', coverColor: '#EF5B9C', songs: [] },
];

const LibraryContext = createContext<LibraryContextType | undefined>(undefined);

export function LibraryProvider({ children }: { children: React.ReactNode }) {
    const [likedSongs, setLikedSongs] = useState<LikedSong[]>([]);
    const [dislikedSongs, setDislikedSongs] = useState<DislikedSong[]>([]);
    const [collections, setCollections] = useState<Collection[]>(INITIAL_COLLECTIONS);
    const [toast, setToast] = useState<string | null>(null);
    const [isSyncing, setIsSyncing] = useState(false);

    // Load from local storage first (fast startup)
    useEffect(() => {
        try {
            const stored = localStorage.getItem('music_likes');
            if (stored) setLikedSongs(JSON.parse(stored));
            const storedCols = localStorage.getItem('music_collections');
            if (storedCols) setCollections(JSON.parse(storedCols));
            const storedDislikes = localStorage.getItem('music_dislikes');
            if (storedDislikes) setDislikedSongs(JSON.parse(storedDislikes));
        } catch (e) {
            console.error('Failed to load from local storage', e);
        }
    }, []);

    // Then sync from backend (authoritative source)
    // Backend (Neo4j) is the single source of truth for LIKES/DISLIKES.
    // On sync, we REPLACE local state with backend data to ensure
    // unlike/undislike operations are properly reflected after refresh/restart.
    const syncFromBackend = useCallback(async () => {
        setIsSyncing(true);
        try {
            const [backendLiked, backendDisliked] = await Promise.all([
                fetchLikedSongs(50),
                fetchDislikedSongs(50),
            ]);

            // ★ Replace liked songs with backend data (authoritative)
            // Previously this only appended, causing unliked songs to persist in localStorage
            const backendLikedSongs: LikedSong[] = backendLiked.map(bl => ({
                id: `${bl.song.title}_${bl.song.artist}`,
                title: bl.song.title,
                artist: bl.song.artist,
                genre: bl.song.genre,
                preview_url: bl.song.audio_url,
                coverUrl: bl.song.cover_url,
                lrc_url: bl.song.lrc_url,
                addedAt: Date.now(),
            }));
            setLikedSongs(backendLikedSongs);
            console.log(`[Sync] 后端权威同步: ${backendLikedSongs.length} 首喜欢的歌曲`);

            // ★ Replace disliked songs with backend data (authoritative)
            const backendDislikedSongs: DislikedSong[] = backendDisliked.map(d => ({
                id: `${d.title}_${d.artist}`,
                title: d.title,
                artist: d.artist,
                coverUrl: d.cover_url,
                dislikedAt: d.disliked_at || Date.now(),
            }));
            setDislikedSongs(backendDislikedSongs);
            console.log(`[Sync] 后端权威同步: ${backendDislikedSongs.length} 首不喜欢的歌曲`);
        } catch (err) {
            console.warn('[Sync] 后端同步失败，保留本地缓存:', err);
        } finally {
            setIsSyncing(false);
        }
    }, []);

    // Auto-sync on mount
    useEffect(() => {
        syncFromBackend();
    }, [syncFromBackend]);

    // Save likes to local storage
    useEffect(() => {
        try { localStorage.setItem('music_likes', JSON.stringify(likedSongs)); }
        catch (e) { console.error('Failed to save liked songs', e); }
    }, [likedSongs]);

    // Save dislikes to local storage
    useEffect(() => {
        try { localStorage.setItem('music_dislikes', JSON.stringify(dislikedSongs)); }
        catch (e) { console.error('Failed to save disliked songs', e); }
    }, [dislikedSongs]);

    // Save collections to local storage
    useEffect(() => {
        try { localStorage.setItem('music_collections', JSON.stringify(collections)); }
        catch (e) { console.error('Failed to save collections', e); }
    }, [collections]);

    const showToast = (msg: string) => {
        setToast(msg);
        setTimeout(() => setToast(null), 2500);
    };

    const isLiked = (title: string, artist: string) => {
        const id = `${title}_${artist}`;
        return likedSongs.some(song => song.id === id);
    };

    const isDisliked = (title: string, artist: string) => {
        const id = `${title}_${artist}`;
        return dislikedSongs.some(song => song.id === id);
    };

    const toggleLike = (song: Omit<LikedSong, 'id' | 'addedAt'>) => {
        const id = `${song.title}_${song.artist}`;
        setLikedSongs(prev => {
            const existing = prev.find(s => s.id === id);
            if (existing) {
                sendUserEvent('unlike', song.title, song.artist);
                return prev.filter(s => s.id !== id);
            }
            showToast(`♥ 已添加到「我的喜欢」`);
            sendUserEvent('like', song.title, song.artist);
            return [{ ...song, id, addedAt: Date.now() }, ...prev];
        });
    };

    const toggleDislike = (song: { title: string; artist: string; coverUrl?: string }) => {
        const id = `${song.title}_${song.artist}`;
        setDislikedSongs(prev => {
            const existing = prev.find(s => s.id === id);
            if (existing) {
                // 已经是不喜欢 → 撤销
                apiRemoveDislike(song.title, song.artist);
                showToast(`已撤销「不喜欢」`);
                return prev.filter(s => s.id !== id);
            }
            // 标记为不喜欢
            sendUserEvent('dislike', song.title, song.artist);
            showToast(`👎 已标记为「不喜欢」，后续推荐将过滤此歌曲`);
            // 同时从 likes 中移除
            setLikedSongs(lp => lp.filter(s => s.id !== id));
            return [{ ...song, id, coverUrl: song.coverUrl, dislikedAt: Date.now() }, ...prev];
        });
    };

    const undoDislike = async (title: string, artist: string) => {
        const id = `${title}_${artist}`;
        const ok = await apiRemoveDislike(title, artist);
        if (ok) {
            setDislikedSongs(prev => prev.filter(s => s.id !== id));
            showToast(`✓ 已从「不喜欢」列表中移除`);
        }
    };

    const addCollection = (name: string) => {
        const id = Date.now();
        setCollections(prev => [...prev, { id, name, coverColor: '#242424', songs: [] }]);
        return id;
    };

    const addToCollection = (collectionId: number, song: Omit<CollectionSong, 'id' | 'addedAt'>) => {
        const id = `${song.title}_${song.artist}`;
        setCollections(prev => prev.map(c => {
            if (c.id !== collectionId) return c;
            if (c.songs.some(s => s.id === id)) return c; // already in
            showToast(`✓ 已添加到「${c.name}」`);
            sendUserEvent('save', song.title, song.artist);
            return { ...c, songs: [{ ...song, id, addedAt: Date.now() }, ...c.songs] };
        }));
    };

    const removeFromCollection = (collectionId: number, songId: string) => {
        setCollections(prev => prev.map(c =>
            c.id === collectionId ? { ...c, songs: c.songs.filter(s => s.id !== songId) } : c
        ));
    };

    const isInCollection = (collectionId: number, title: string, artist: string) => {
        const col = collections.find(c => c.id === collectionId);
        const id = `${title}_${artist}`;
        return col ? col.songs.some(s => s.id === id) : false;
    };

    return (
        <LibraryContext.Provider value={{
            likedSongs, dislikedSongs,
            toggleLike, toggleDislike, undoDislike,
            isLiked, isDisliked,
            collections, addCollection, addToCollection, removeFromCollection, isInCollection,
            syncFromBackend, isSyncing,
            toast, showToast,
        }}>
            {children}
            {/* 全局 Toast 通知 */}
            {toast && (
                <div style={{
                    position: 'fixed',
                    top: '1.5rem',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    backgroundColor: 'rgba(30,30,30,0.95)',
                    color: '#fff',
                    padding: '0.75rem 1.5rem',
                    borderRadius: '2rem',
                    fontSize: '0.9rem',
                    fontWeight: 500,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
                    border: '1px solid rgba(255,255,255,0.12)',
                    zIndex: 9999,
                    animation: 'fadeInDown 0.2s ease',
                    letterSpacing: '0.01em',
                    backdropFilter: 'blur(10px)',
                }}>
                    {toast}
                </div>
            )}
        </LibraryContext.Provider>
    );
}

export function useLibrary() {
    const context = useContext(LibraryContext);
    if (context === undefined) {
        throw new Error('useLibrary must be used within a LibraryProvider');
    }
    return context;
}
