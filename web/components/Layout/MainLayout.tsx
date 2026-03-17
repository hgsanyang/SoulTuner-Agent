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

  useEffect(() => {
    // 每次路径变化时重置滚动条
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: 'transparent',
        position: 'relative',
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
          minHeight: '100vh',
          padding: containerPadding,
          gap: '1.75rem',
        }}
      >
        <Header
          onMenuToggle={isMobile ? () => setSidebarOpen((prev) => !prev) : undefined}
          isMobile={isMobile}
        />
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
              }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
        {onInputSubmit && (
          <div style={{ maxWidth: `${theme.layout.contentMaxWidth}px`, margin: '0 auto', width: '100%' }}>
            {toolbar && <div style={{ marginBottom: '0.5rem' }}>{toolbar}</div>}
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


