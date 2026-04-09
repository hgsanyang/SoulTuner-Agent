# syntax=docker/dockerfile:1.9
FROM node:22-alpine as builder

WORKDIR /app

# Install system dependencies for building
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    curl \
    ca-certificates

# Copy package files for core library
COPY package*.json ./
COPY tsconfig.json ./

# Copy source code for core library
COPY src/ ./src/

# Install dependencies and build core library
RUN npm ci --include=dev
RUN npm run build

# Build server stage
FROM node:22-alpine as server-builder

WORKDIR /app

# Copy core library source (needed for TypeScript compilation with rootDir="..")
COPY --from=builder /app/src ./src/
COPY --from=builder /app/dist ./dist/
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/tsconfig.json ./tsconfig.json

# Copy server files
COPY server/package*.json ./server/
COPY server/tsconfig.json ./server/
COPY server/src/ ./server/src/

# Install server dependencies
# Install server dependencies
WORKDIR /app/server
RUN npm ci --include=dev

# Install core library's type dependencies at /app level
# (../src/ files resolve modules via /app/node_modules/, NOT /app/server/node_modules/)
WORKDIR /app
RUN npm install --no-save zod openai uuid @types/uuid dotenv neo4j-driver

# Build server (compiles both server/src and ../src together, rootDir="..")
WORKDIR /app/server
RUN npm run build

# Runtime stage
FROM node:22-alpine

# Install curl for healthcheck
RUN apk add --no-cache curl

# Create non-root user
RUN addgroup -g 1001 -S app && \
    adduser -S app -u 1001

# Set up the application directory
WORKDIR /app

# Copy built server application (includes both server and core compiled output)
COPY --from=server-builder /app/server/dist ./dist/
COPY --from=server-builder /app/server/package*.json ./
COPY --from=server-builder /app/server/node_modules ./node_modules/

# Copy core library compiled output to node_modules for runtime resolution
COPY --from=builder /app/dist ./node_modules/graphzep/dist/
COPY --from=builder /app/package.json ./node_modules/graphzep/

# Install production dependencies only
RUN npm ci --only=production && \
    npm install --no-save neo4j-driver openai zod uuid && \
    npm cache clean --force

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/healthcheck || exit 1

# Start the server (path preserves rootDir=".." structure: dist/server/src/standalone-main.js)
CMD ["node", "dist/server/src/standalone-main.js"]