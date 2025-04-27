FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies
COPY ui/package*.json ./
RUN npm install --legacy-peer-deps

# Copy all files
COPY ui/ .

# Make prebuild.sh executable and run it
RUN chmod +x prebuild.sh || echo "No prebuild.sh script found, creating directory manually"
RUN mkdir -p /app/src/lib/ && ls -la /app/src/lib/

# Run prebuild script for environment setup
RUN sh -c "./prebuild.sh"

# Log the contents of the lib directory
RUN echo "Contents of /app/src/lib after prebuild:" && ls -la /app/src/lib/

# Build the Next.js app with increased memory
RUN NODE_OPTIONS=--max-old-space-size=4096 npm run build

# Runner stage
FROM node:22-alpine AS runner

WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080
ENV NEXT_TELEMETRY_DISABLED=1

# Set up non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    chown -R nextjs:nodejs /app

# Copy built app
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Create directory for GCS mount and set permissions
RUN mkdir -p /gcs && chown nextjs:nodejs /gcs

USER nextjs

# Expose port
EXPOSE 8080

# Start server
CMD ["node", "server.js"]