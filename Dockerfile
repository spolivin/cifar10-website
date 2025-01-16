FROM node:18-alpine
COPY . /code
WORKDIR /code
RUN npm ci
RUN npm install http-server --global
RUN npm run build
EXPOSE 8080
CMD http-server ./dist
