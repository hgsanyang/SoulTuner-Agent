'use client';

import { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { theme } from '@/styles/theme';

interface NavItemProps {
  href: string;
  label: string;
  description: string;
  icon: ReactNode;
}

export default function NavItem({ href, label, description, icon }: NavItemProps) {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <Link href={href}>
      <div
        style={{
          padding: '0.95rem 1.1rem',
          borderRadius: theme.borderRadius.md,
          backgroundColor: isActive ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.03)',
          color: '#fff',
          cursor: 'pointer',
          fontWeight: 500,
          display: 'flex',
          alignItems: 'center',
          gap: '0.85rem',
          border: `1px solid ${isActive ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.08)'}`,
          transition: 'border-color 0.2s ease, background-color 0.2s ease',
        }}
      >
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: theme.borderRadius.md,
            backgroundColor: isActive ? 'rgba(255,255,255,0.24)' : 'rgba(255,255,255,0.08)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: isActive ? '#fff' : 'rgba(255,255,255,0.7)',
          }}
        >
          {icon}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
          <span style={{ fontSize: '0.95rem' }}>{label}</span>
          <span
            style={{
              fontSize: '0.78rem',
              color: isActive ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.6)',
            }}
          >
            {description}
          </span>
        </div>
      </div>
    </Link>
  );
}

