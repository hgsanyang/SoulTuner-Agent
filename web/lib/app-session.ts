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
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  return fetch(input, {
    ...init,
    headers: sessionHeaders(init.headers),
  });
}

export function apiFetchFor(
  context: SessionRequestContext,
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  return fetch(input, {
    ...init,
    headers: sessionHeadersFor(context, init.headers),
  });
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
