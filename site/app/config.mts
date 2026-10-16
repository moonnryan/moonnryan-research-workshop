import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Moonnryan's Research Workshop",
  description: '研究、实践与思考。',
  lang: 'zh-CN',
  appearance: true,
  base: process.env.MRW_BASE || '/',
  cleanUrls: false,
  outDir: '../dist',
  markdown: {
    math: true,
    headers: { level: [1, 2, 3] },
  },
  vite: {
    server: {
      watch: {
        usePolling: true,
        interval: 350,
        ignored: ['**/node_modules/**', '**/.cache/**'],
      },
    },
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${(process.env.MRW_BASE || '/').replace(/\/?$/, '/')}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#f8f7f3' }],
  ],
})
