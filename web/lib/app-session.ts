import { resolveBackendUrl } from '@/lib/runtime-url';

export type InteractionMode = 'personal' | 'developer';

export interface SessionRequestContext {
  profileId: string;
  interactionMode: InteractionMode;
  sessionId: string;
}

const DEFAULT_CONTEXT: SessionRequestContext = {
  profileId: 'local_admin',
  interactionMode: 'personal',
  sessionId: '',
};

let activeContext: SessionRequestContext = { ...DEFAULT_CONTEXT };
const activeStreamControllers = new Set<AbortController>();

export function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return `session-${Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')}`;
  }
  throw new Error('当前环境不支持安全会话标识，请使用 HTTPS 或 localhost');
}

export function setActiveRequestContext(context: SessionRequestContext): void {
  activeContext = { ...context };
}

export function getActiveRequestContext(): SessionRequestContext {
  return { ...activeContext };
}

export function scopedStorageKey(
  key: string,
  profileId: string = activeContext.profileId,
  mode: InteractionMode = activeContext.interactionMode,
): string {
  return `soultuner:${encodeURIComponent(profileId)}:${mode}:${key}`;
}

export function sessionHeaders(headers?: HeadersInit): Headers {
  return sessionHeadersFor(activeContext, headers);
}

export function sessionHeadersFor(
  context: SessionRequestContext,
  headers?: HeadersInit,
): Headers {
  const merged = new Headers(headers);
  merged.set('X-SoulTuner-Profile', context.profileId);
  merged.set('X-SoulTuner-Mode', context.interactionMode);
  merged.set('X-SoulTuner-Session', context.sessionId);
  return merged;
}

let visitorBootstrap: Promise<void> | undefined;
let visitorExpired = false;
const VISITOR_EXPIRED_MESSAGE = '访客会话已过期。请刷新页面创建新会话；旧会话的记忆不会自动转移。';

async function ensureVisitorSession(): Promise<void> {
  if (visitorExpired) throw new Error(VISITOR_EXPIRED_MESSAGE);
  if (!visitorBootstrap) {
    visitorBootstrap = fetch(resolveBackendUrl('http://localhost:8501/api/anonymous-session'), {
      credentials: 'include', cache: 'no-store',
    }).then(response => {
      if (!response.ok) throw new Error('访客会话初始化失败，请稍后重试');
    }).catch(error => { visitorBootstrap = undefined; throw error; });
  }
  await visitorBootstrap;
}

async function checkVisitorResponse(response: Response): Promise<Response> {
  if (response.status === 401) {
    const data = await response.clone().json().catch(() => ({}));
    if (data.error === 'anonymous_session_required') {
      visitorExpired = true;
      abortActiveSessionStreams();
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('soultuner:visitor-expired', { detail: VISITOR_EXPIRED_MESSAGE }));
      }
      // Never automatically replay a feedback/memory POST under a new identity.
      throw new Error(VISITOR_EXPIRED_MESSAGE);
    }
  }
  return response;
}

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  await ensureVisitorSession();
  const resolved = typeof input === 'string' || input instanceof URL
    ? resolveBackendUrl(input)
    : input;
  return checkVisitorResponse(await fetch(resolved, {
    ...init,
    credentials: 'include',
    headers: sessionHeaders(init.headers),
  }));
}

export async function apiFetchFor(
  context: SessionRequestContext,
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  await ensureVisitorSession();
  const resolved = typeof input === 'string' || input instanceof URL
    ? resolveBackendUrl(input)
    : input;
  return checkVisitorResponse(await fetch(resolved, {
    ...init,
    credentials: 'include',
    headers: sessionHeadersFor(context, init.headers),
  }));
}

export function registerActiveStream(controller: AbortController): () => void {
  activeStreamControllers.add(controller);
  return () => activeStreamControllers.delete(controller);
}

export function abortActiveSessionStreams(): void {
  for (const controller of activeStreamControllers) {
    controller.abort();
  }
  activeStreamControllers.clear();
}
