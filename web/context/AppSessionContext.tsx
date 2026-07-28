'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import {
  abortActiveSessionStreams,
  apiFetch,
  createSessionId,
  InteractionMode,
  scopedStorageKey,
  setActiveRequestContext,
} from '@/lib/app-session';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8501';
const PROFILE_CACHE_KEY = 'soultuner:profiles';
const ACTIVE_PROFILE_KEY = 'soultuner:active-profile';
const ACTIVE_MODE_KEY = 'soultuner:active-mode';

export interface AppProfile {
  profile_id: string;
  display_name: string;
  profile_type: 'personal' | 'test' | string;
  status: 'active' | 'deleted' | string;
}

interface AppSessionContextValue {
  profiles: AppProfile[];
  activeProfile: AppProfile;
  interactionMode: InteractionMode;
  sessionId: string;
  hydrated: boolean;
  profileLoading: boolean;
  profileError: string;
  switchProfile: (profileId: string) => void;
  setInteractionMode: (mode: InteractionMode) => void;
  createProfile: (displayName: string) => Promise<AppProfile>;
  storageKey: (key: string) => string;
}

const DEFAULT_PROFILE: AppProfile = {
  profile_id: 'local_admin',
  display_name: '默认档案',
  profile_type: 'personal',
  status: 'active',
};

const AppSessionContext = createContext<AppSessionContextValue | null>(null);

function normalizeProfiles(payload: unknown): AppProfile[] {
  if (!payload || typeof payload !== 'object') return [];
  const record = payload as Record<string, unknown>;
  const raw = Array.isArray(record.profiles)
    ? record.profiles
    : Array.isArray(record.items)
      ? record.items
      : Array.isArray(payload)
        ? payload
        : [];
  return raw.filter((item): item is AppProfile => {
    if (!item || typeof item !== 'object') return false;
    const profile = item as Record<string, unknown>;
    return typeof profile.profile_id === 'string' && typeof profile.display_name === 'string';
  });
}

function readCachedProfiles(): AppProfile[] {
  try {
    const raw = localStorage.getItem(PROFILE_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function AppSessionProvider({ children }: { children: React.ReactNode }) {
  const [profiles, setProfiles] = useState<AppProfile[]>([DEFAULT_PROFILE]);
  const [activeProfileId, setActiveProfileId] = useState(DEFAULT_PROFILE.profile_id);
  const [interactionMode, setMode] = useState<InteractionMode>('personal');
  const [sessionId, setSessionId] = useState('');
  const [hydrated, setHydrated] = useState(false);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileError, setProfileError] = useState('');

  useEffect(() => {
    const cachedProfiles = readCachedProfiles();
    const cachedProfileId = localStorage.getItem(ACTIVE_PROFILE_KEY) || DEFAULT_PROFILE.profile_id;
    const cachedMode = localStorage.getItem(ACTIVE_MODE_KEY);
    const initialMode: InteractionMode = cachedMode === 'developer' ? 'developer' : 'personal';
    const initialProfiles = cachedProfiles.length > 0 ? cachedProfiles : [DEFAULT_PROFILE];
    const initialProfile = initialProfiles.some(profile => profile.profile_id === cachedProfileId)
      ? cachedProfileId
      : initialProfiles[0].profile_id;
    const nextSessionId = createSessionId();

    setProfiles(initialProfiles);
    setActiveProfileId(initialProfile);
    setMode(initialMode);
    setSessionId(nextSessionId);
    setActiveRequestContext({
      profileId: initialProfile,
      interactionMode: initialMode,
      sessionId: nextSessionId,
    });
    setHydrated(true);

    apiFetch(`${API_URL}/api/profiles`)
      .then(async response => {
        if (!response.ok) throw new Error(`档案读取失败: ${response.status}`);
        const remoteProfiles = normalizeProfiles(await response.json());
        if (remoteProfiles.length === 0) return;
        setProfiles(remoteProfiles);
        setActiveProfileId(current => (
          remoteProfiles.some(profile => profile.profile_id === current)
            ? current
            : remoteProfiles[0].profile_id
        ));
        localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(remoteProfiles));
        setProfileError('');
      })
      .catch(error => {
        setProfileError(error instanceof Error ? error.message : '档案服务暂不可用');
      })
      .finally(() => setProfileLoading(false));
  }, []);

  useEffect(() => {
    if (!hydrated || !sessionId) return;
    setActiveRequestContext({
      profileId: activeProfileId,
      interactionMode,
      sessionId,
    });
    localStorage.setItem(ACTIVE_PROFILE_KEY, activeProfileId);
    localStorage.setItem(ACTIVE_MODE_KEY, interactionMode);
  }, [activeProfileId, hydrated, interactionMode, sessionId]);

  const startFreshSession = useCallback((
    profileId: string,
    mode: InteractionMode,
  ) => {
    abortActiveSessionStreams();
    const nextSessionId = createSessionId();
    setActiveRequestContext({
      profileId,
      interactionMode: mode,
      sessionId: nextSessionId,
    });
    setSessionId(nextSessionId);
  }, []);

  const switchProfile = useCallback((profileId: string) => {
    if (profileId === activeProfileId) return;
    setActiveProfileId(profileId);
    startFreshSession(profileId, interactionMode);
  }, [activeProfileId, interactionMode, startFreshSession]);

  const setInteractionMode = useCallback((mode: InteractionMode) => {
    if (mode === interactionMode) return;
    setMode(mode);
    startFreshSession(activeProfileId, mode);
  }, [activeProfileId, interactionMode, startFreshSession]);

  const createProfile = useCallback(async (displayName: string): Promise<AppProfile> => {
    const cleaned = displayName.trim();
    if (!cleaned) throw new Error('请输入档案名称');
    setProfileLoading(true);
    setProfileError('');
    try {
      const response = await apiFetch(`${API_URL}/api/profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          display_name: cleaned,
          profile_type: 'personal',
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || `创建失败: ${response.status}`);
      }
      const profile = (payload.profile || payload) as AppProfile;
      if (!profile.profile_id || !profile.display_name) {
        throw new Error('档案服务返回了无效数据');
      }
      setProfiles(current => {
        const next = [...current.filter(item => item.profile_id !== profile.profile_id), profile];
        localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(next));
        return next;
      });
      setActiveProfileId(profile.profile_id);
      startFreshSession(profile.profile_id, interactionMode);
      return profile;
    } catch (error) {
      const message = error instanceof Error ? error.message : '创建档案失败';
      setProfileError(message);
      throw error;
    } finally {
      setProfileLoading(false);
    }
  }, [interactionMode, startFreshSession]);

  const activeProfile = profiles.find(profile => profile.profile_id === activeProfileId)
    || profiles[0]
    || DEFAULT_PROFILE;

  const storageKey = useCallback((key: string) => (
    scopedStorageKey(key, activeProfile.profile_id, interactionMode)
  ), [activeProfile.profile_id, interactionMode]);

  const value = useMemo<AppSessionContextValue>(() => ({
    profiles,
    activeProfile,
    interactionMode,
    sessionId,
    hydrated,
    profileLoading,
    profileError,
    switchProfile,
    setInteractionMode,
    createProfile,
    storageKey,
  }), [
    activeProfile,
    createProfile,
    hydrated,
    interactionMode,
    profileError,
    profileLoading,
    profiles,
    sessionId,
    setInteractionMode,
    storageKey,
    switchProfile,
  ]);

  return (
    <AppSessionContext.Provider value={value}>
      {children}
    </AppSessionContext.Provider>
  );
}

export function useAppSession(): AppSessionContextValue {
  const context = useContext(AppSessionContext);
  if (!context) {
    throw new Error('useAppSession must be used within AppSessionProvider');
  }
  return context;
}
