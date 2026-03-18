'use client';

import { ReactNode, useState, useEffect } from 'react';
import Header from './Header';
import Sidebar from '../Navigation/Sidebar';
import ChatInput from '../Input/ChatInput';
import { theme } from '@/styles/theme';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { motion, AnimatePresence } from 'framer-motion';
import { usePathname } from 'next/navigation';

interface MainLayoutProps {
  children: ReactNode;
  onInputSubmit?: (value: string) => void;
  onInputAbort?: () => void;         // 中止当前搜索
  inputPlaceholder?: string;
  inputDisabled?: boolean;            // 保留向后兼容
  inputIsLoading?: boolean;           // 搜索进行中（用于中止按钮切换）
  toolbar?: ReactNode;
}

export default function MainLayout({
  children,
  onInputSubmit,
  onInputAbort,
  inputPlaceholder,
  inputDisabled = false,
  inputIsLoading = false,
  toolbar,
}: MainLayoutProps) {
  const isMobile = useMediaQuery('(max-width: 960px)');
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const sidebarWidth = isMobile ? 0 : theme.layout.sidebarWidth;
  const containerPadding = isMobile ? '0' : '1.5rem 2rem';
  const pathname = usePathname();

  // Removed: window.scrollTo no longer needed as page uses height:100vh + overflow:hidden

  return (
    <div
      style={{
        height: '100vh',
        backgroundColor: 'transparent',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {isMobile && isSidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            zIndex: 9,
            backdropFilter: 'blur(4px)',
          }}
        />
      )}
      <Sidebar
        isMobile={isMobile}
        isOpen={isMobile ? isSidebarOpen : true}
        onClose={() => setSidebarOpen(false)}
      />
      <div
        style={{
          marginLeft: isMobile ? 0 : `${sidebarWidth}px`,
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          padding: containerPadding,
          paddingBottom: '90px',
          gap: '1.25rem',
          overflow: 'hidden',
        }}
      >
        {isMobile && (
          <button
            type="button"
            aria-label="打开导航"
            onClick={() => setSidebarOpen(true)}
            style={{
              position: 'absolute',
              top: '1rem',
              left: '1rem',
              width: '40px',
              height: '40px',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              backgroundColor: 'rgba(36, 36, 36, 0.4)',
              backdropFilter: 'blur(16px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              zIndex: 10,
            }}
          >
            <span
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '5px',
              }}
            >
              {[0, 1, 2].map((line) => (
                <span
                  key={line}
                  style={{
                    width: '22px',
                    height: '2px',
                    borderRadius: '9999px',
                    backgroundColor: theme.colors.text.primary,
                  }}
                />
              ))}
            </span>
          </button>
        )}
        <main
          style={{
            flex: 1,
            width: '100%',
            alignSelf: 'center',
            maxWidth: `${theme.layout.contentMaxWidth}px`,
            margin: '0 auto',
            padding: 0,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
          }}
        >
          {/* Framer Motion for Page Transitions */}
          <AnimatePresence mode="wait">
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.35, ease: "easeInOut" }}
              style={{
                flex: 1,
                width: '100%',
                backgroundColor: 'rgba(36, 36, 36, 0.4)',
                backdropFilter: 'blur(16px)',
                borderRadius: isMobile ? '0' : '1.15rem',
                border: isMobile ? 'none' : `1px solid ${theme.colors.border.default}`,
                padding: isMobile ? '1.25rem' : '2.5rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '1.25rem',
                overflowY: 'auto',
                minHeight: 0,
              }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* 工具栏 + 搜索输入（卡片外部，紧随其后，文档流定位） */}
        {onInputSubmit && (
          <div style={{
            maxWidth: `${theme.layout.contentMaxWidth}px`,
            margin: '0 auto',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            flexShrink: 0,
            paddingBottom: isMobile ? '0.5rem' : '1rem',
          }}>
            {toolbar && (
              <div style={{
                margin: isMobile ? '0 auto 0.5rem' : '0 auto 0.5rem',
                padding: isMobile ? '0 0.25rem' : '0 1rem',
                maxWidth: isMobile ? '520px' : '640px',
                width: '100%',
                position: 'relative',
                zIndex: 100,
              }}>
                {toolbar}
              </div>
            )}
            <ChatInput
              onSubmit={onInputSubmit!}
              onAbort={onInputAbort}
              placeholder={inputPlaceholder}
              disabled={inputDisabled && !inputIsLoading}
              isLoading={inputIsLoading}
              isMobile={isMobile}
            />
          </div>
        )}
      </div>
    </div>
  );
}


