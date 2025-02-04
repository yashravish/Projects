# Use the official Node.js image.
FROM node:14 as builder
WORKDIR /app
COPY ../node-graphql-api/package.json ../node-graphql-api/package-lock.json* ./
RUN npm install
COPY ../node-graphql-api/ ./
# (Optional build step if using a build tool)
RUN npm run build || echo "No build step required"

# Final image.
FROM node:14
WORKDIR /app
COPY --from=builder /app /app
EXPOSE 5000
CMD ["npm", "start"]
