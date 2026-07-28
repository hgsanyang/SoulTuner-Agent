'use client';

import { useState } from 'react';
import { useLang } from '@/context/LanguageContext';
import { useAppSession } from '@/context/AppSessionContext';
import { theme } from '@/styles/theme';

export default function ProfileSwitcher() {
  const { t } = useLang();
  const {
    profiles,
    activeProfile,
    interactionMode,
    profileLoading,
    profileError,
    switchProfile,
    setInteractionMode,
    createProfile,
  } = useAppSession();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [displayName, setDisplayName] = useState('');

  const submitProfile = async () => {
    try {
      await createProfile(displayName);
      setDisplayName('');
      setCreating(false);
      setOpen(false);
    } catch {
      // The provider exposes the actionable error below.
    }
  };

  return (
    <div style={{ display: 'grid', gap: '0.55rem' }}>
      <div
        aria-label={t("交互模式")}
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          padding: '3px',
          borderRadius: theme.borderRadius.sm,
          border: `1px solid ${theme.colors.border.default}`,
          background: 'rgba(255,255,255,0.025)',
        }}
      >
        {([
          { value: 'personal' as const, label: t('日常') },
          { value: 'developer' as const, label: t('开发') },
        ]).map(option => {
          const active = interactionMode === option.value;
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => setInteractionMode(option.value)}
              aria-pressed={active}
              style={{
                minHeight: '30px',
                border: 'none',
                borderRadius: '4px',
                background: active
                  ? option.value === 'developer'
                    ? 'rgba(96,165,250,0.18)'
                    : 'rgba(29,185,84,0.18)'
                  : 'transparent',
                color: active
                  ? option.value === 'developer' ? '#93c5fd' : '#86efac'
                  : theme.colors.text.muted,
                fontSize: '0.76rem',
                fontWeight: 700,
                cursor: 'pointer',
              }}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      {/* 档案是「日常模式」的概念：它决定这次交互算在谁头上、参与谁的个性化学习。
          开发模式本来就把数据隔离出去、不参与学习，再让人选档案只会让人以为
          「选错了会污染数据」。所以开发模式下不显示选择器，只说明当前状态。 */}
      {interactionMode === 'developer' ? (
        <div
          style={{
            minHeight: '38px',
            padding: '0.45rem 0.6rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            borderRadius: theme.borderRadius.sm,
            border: `1px dashed ${theme.colors.border.default}`,
            background: 'rgba(96,165,250,0.06)',
            color: theme.colors.text.muted,
            fontSize: '0.7rem',
            lineHeight: 1.45,
          }}
        >
          开发模式不使用用户档案，本次交互的数据独立存放、不参与个性化学习。
        </div>
      ) : (
      <div style={{ position: 'relative' }}>
        <button
          type="button"
          onClick={() => setOpen(value => !value)}
          aria-expanded={open}
          aria-label={t("切换用户档案")}
          style={{
            width: '100%',
            minHeight: '38px',
            padding: '0.45rem 0.55rem',
            display: 'grid',
            gridTemplateColumns: '28px minmax(0,1fr) 16px',
            alignItems: 'center',
            gap: '0.5rem',
            borderRadius: theme.borderRadius.sm,
            border: `1px solid ${theme.colors.border.default}`,
            background: 'rgba(255,255,255,0.025)',
            color: theme.colors.text.primary,
            cursor: 'pointer',
            textAlign: 'left',
          }}
        >
          <span
            aria-hidden="true"
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '50%',
              display: 'grid',
              placeItems: 'center',
              // 这个分支只在日常模式下渲染，开发模式走上面的说明块。
              background: 'rgba(29,185,84,0.18)',
              color: '#86efac',
              fontSize: '0.72rem',
              fontWeight: 800,
            }}
          >
            {activeProfile.display_name.trim().slice(0, 1).toUpperCase() || 'U'}
          </span>
          <span style={{ minWidth: 0 }}>
            <span
              style={{
                display: 'block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontSize: '0.78rem',
                fontWeight: 700,
              }}
            >
              {activeProfile.display_name}
            </span>
            <span style={{ display: 'block', color: theme.colors.text.muted, fontSize: '0.66rem' }}>
              参与个性化学习
            </span>
          </span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>

        {open && (
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: 'calc(100% + 6px)',
              zIndex: 30,
              padding: '0.4rem',
              borderRadius: theme.borderRadius.sm,
              border: `1px solid ${theme.colors.border.focus}`,
              background: '#171717',
              boxShadow: theme.shadows.md,
            }}
          >
            <div style={{ maxHeight: '180px', overflowY: 'auto', display: 'grid', gap: '2px' }}>
              {profiles.filter(profile => profile.status !== 'deleted').map(profile => (
                <button
                  key={profile.profile_id}
                  type="button"
                  onClick={() => {
                    switchProfile(profile.profile_id);
                    setOpen(false);
                  }}
                  style={{
                    width: '100%',
                    padding: '0.5rem 0.55rem',
                    border: 'none',
                    borderRadius: '4px',
                    background: profile.profile_id === activeProfile.profile_id
                      ? 'rgba(255,255,255,0.08)'
                      : 'transparent',
                    color: theme.colors.text.primary,
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: '0.76rem',
                  }}
                >
                  {profile.display_name}
                </button>
              ))}
            </div>

            <div style={{ borderTop: `1px solid ${theme.colors.border.default}`, marginTop: '0.35rem', paddingTop: '0.35rem' }}>
              {creating ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 30px', gap: '0.35rem' }}>
                  <input
                    autoFocus
                    value={displayName}
                    onChange={event => setDisplayName(event.target.value)}
                    onKeyDown={event => {
                      if (event.key === 'Enter') submitProfile();
                      if (event.key === 'Escape') setCreating(false);
                    }}
                    placeholder={t("新档案名称")}
                    maxLength={32}
                    style={{
                      minWidth: 0,
                      height: '30px',
                      padding: '0 0.5rem',
                      borderRadius: '4px',
                      border: `1px solid ${theme.colors.border.focus}`,
                      background: '#0f0f0f',
                      color: theme.colors.text.primary,
                      outline: 'none',
                    }}
                  />
                  <button
                    type="button"
                    onClick={submitProfile}
                    disabled={profileLoading || !displayName.trim()}
                    title={t("创建档案")}
                    style={{
                      width: '30px',
                      height: '30px',
                      borderRadius: '4px',
                      border: 'none',
                      background: theme.colors.primary.accent,
                      color: '#07140b',
                      fontSize: '1rem',
                      cursor: 'pointer',
                    }}
                  >
                    +
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setCreating(true)}
                  style={{
                    width: '100%',
                    padding: '0.5rem 0.55rem',
                    border: 'none',
                    borderRadius: '4px',
                    background: 'transparent',
                    color: '#86efac',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: '0.76rem',
                    fontWeight: 700,
                  }}
                >
                  ＋ 新建档案
                </button>
              )}
            </div>
            {profileError && (
              <div style={{ padding: '0.4rem 0.5rem 0.15rem', color: '#fca5a5', fontSize: '0.66rem', lineHeight: 1.4 }}>
                {profileError}
              </div>
            )}
          </div>
        )}
      </div>
      )}
    </div>
  );
}
