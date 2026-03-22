# ============================================================
# SoulTuner-Agent Next.js 前端 Dockerfile
# 多阶段构建：builder 编译 → runner 仅运行产物
# ============================================================

# ---- Stage 1: 构建 ----
FROM node:22-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# 构建时需要的环境变量（API 地址在运行时可覆盖）
ENV NEXT_PUBLIC_API_URL=http://localhost:8501
RUN npm run build

# ---- Stage 2: 运行 ----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# 仅复制必要文件
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./

EXPOSE 3003

CMD ["npx", "next", "start", "-p", "3003"]
