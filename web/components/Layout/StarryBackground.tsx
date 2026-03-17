'use client';

/**
 * ✨ StarryBackground.tsx (沉浸式深空背景动效组件)
 * 作用：在页面最底层渲染动态的、缓慢移动的漫天星辰视觉效果。
 * 功能特性：
 * 1. 核心算法利用 CSS 的多重 `box-shadow` 以及关键帧动画 (`keyframes animStar`) 大批量低性能开销地生成繁星效果。
 * 2. 使用 `useEffect` 强制要求仅在客户端挂载(Mounted)后生成随机像素系的星星，彻底避开 Next.js 服务端 SSR 与客户端渲染因 `Math.random()` 计算不同导致的 Hydration (水合) 报错机制。
 */

import { useEffect, useState } from 'react';

// 生成指定数量的随机星空 box-shadow 字符串
const generateStars = (count: number, size: number, width: number, height: number) => {
  let value = '';
  for (let i = 0; i < count; i++) {
    const x = Math.floor(Math.random() * width);
    const y = Math.floor(Math.random() * height);
    value += `${x}px ${y}px #FFF${i === count - 1 ? '' : ','}`;
  }
  return value;
};

export default function StarryBackground() {
  const [shadowsSmall, setShadowsSmall] = useState('');
  const [shadowsMedium, setShadowsMedium] = useState('');
  const [shadowsLarge, setShadowsLarge] = useState('');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // 扩大范围至宽 4000px，高 3000px，适配超宽带鱼屏/4K全屏
    setShadowsSmall(generateStars(1200, 1, 4000, 3000));
    setShadowsMedium(generateStars(400, 2, 4000, 3000));
    setShadowsLarge(generateStars(150, 3, 4000, 3000));
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: -1,
          pointerEvents: 'none',
          background: 'radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%)',
        }}
      />
    );
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: -1,
        pointerEvents: 'none',
        background: 'radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%)',
        overflow: 'hidden',
      }}
    >
      <div className="stars-small" />
      <div className="stars-medium" />
      <div className="stars-large" />
      <style key="starry-styles">{`
        .stars-small, .stars-medium, .stars-large {
           border-radius: 50%;
           position: absolute;
           top: 0; left: 0;
        }
        
        .stars-small { width: 1px; height: 1px; background: transparent; box-shadow: ${shadowsSmall}; }
        .stars-small::after { content: " "; position: absolute; top: 3000px; width: 1px; height: 1px; background: transparent; box-shadow: ${shadowsSmall}; }
        
        .stars-medium { width: 2px; height: 2px; background: transparent; box-shadow: ${shadowsMedium}; }
        .stars-medium::after { content: " "; position: absolute; top: 3000px; width: 2px; height: 2px; background: transparent; box-shadow: ${shadowsMedium}; }
        
        .stars-large { width: 3px; height: 3px; background: transparent; box-shadow: ${shadowsLarge}; }
        .stars-large::after { content: " "; position: absolute; top: 3000px; width: 3px; height: 3px; background: transparent; box-shadow: ${shadowsLarge}; }

        .stars-small {
          animation: animStar 100s linear infinite;
        }
        .stars-medium {
          animation: animStar 200s linear infinite;
        }
        .stars-large {
          animation: animStar 300s linear infinite;
        }
        @keyframes animStar {
          from { transform: translateY(0px); }
          to { transform: translateY(-3000px); }
        }
      `}</style>
    </div>
  );
}
