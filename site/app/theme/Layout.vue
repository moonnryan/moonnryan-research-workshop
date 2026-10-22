<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, watch, ref } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import RelatedArticles from './components/RelatedArticles.vue'

const { page, frontmatter, isDark } = useData()
const route = useRoute()
const isArticle = computed(() => frontmatter.value.layout === 'article')
const isAbout = computed(() => frontmatter.value.layout === 'about')
const headings = ref<{ title: string; id: string; depth: number }[]>([])
const mobileOpen = ref(false)
const activeHeading = ref('')
let headingElements: HTMLElement[] = []
let scrollFrame = 0
let bodyObserver: ResizeObserver | undefined

function updateActiveHeading() {
  scrollFrame = 0
  if (!headingElements.length) { activeHeading.value = ''; return }
  let current = headingElements[0]
  for (const element of headingElements) {
    if (element.getBoundingClientRect().top <= 110) current = element
    else break
  }
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 8) {
    current = headingElements[headingElements.length - 1]
  }
  activeHeading.value = current.id
}

function scheduleHeadingUpdate() {
  if (!scrollFrame) scrollFrame = window.requestAnimationFrame(updateActiveHeading)
}

async function updateHeadings() {
  await nextTick()
  bodyObserver?.disconnect()
  const elements = [...document.querySelectorAll<HTMLElement>('.article-body h1[id], .article-body h2[id], .article-body h3[id]')].slice(1)
  headingElements = elements
  const baseLevel = Math.min(...elements.map(element => Number(element.tagName.slice(1))))
  headings.value = elements.map(element => ({
      id: element.id,
      title: (element.textContent ?? '').replace(/\u200b/g, '').trim(),
      depth: Number(element.tagName.slice(1)) - baseLevel,
    }))
  const body = document.querySelector('.article-body')
  if (body) bodyObserver?.observe(body)
  scheduleHeadingUpdate()
}
onMounted(() => {
  bodyObserver = new ResizeObserver(scheduleHeadingUpdate)
  updateHeadings()
  window.addEventListener('scroll', scheduleHeadingUpdate, { passive: true })
  window.addEventListener('resize', scheduleHeadingUpdate)
  watch(isDark, dark => {
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#1b1e1b' : '#f8f7f3')
  }, { immediate: true })
})
onUnmounted(() => {
  bodyObserver?.disconnect()
  window.cancelAnimationFrame(scrollFrame)
  window.removeEventListener('scroll', scheduleHeadingUpdate)
  window.removeEventListener('resize', scheduleHeadingUpdate)
})
watch(() => page.value.relativePath, () => { mobileOpen.value = false; updateHeadings() }, { flush: 'post' })
watch(activeHeading, async () => {
  await nextTick()
  const outline = document.querySelector<HTMLElement>('.article-outline')
  const active = outline?.querySelector<HTMLElement>('[aria-current="location"]')
  if (!outline || !active) return
  const parent = outline.getBoundingClientRect()
  const child = active.getBoundingClientRect()
  const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'
  if (child.bottom > parent.bottom) outline.scrollBy({ top: child.bottom - parent.bottom + 12, behavior })
  else if (child.top < parent.top) outline.scrollBy({ top: child.top - parent.top - 12, behavior })
})
</script>

<template>
  <div class="workshop-shell" :class="{ 'is-home': route.path === withBase('/'), 'is-listing': route.path === withBase('/articles.html') }">
    <header class="site-header">
      <a class="brand" :href="withBase('/')" aria-label="Moonnryan's Research Workshop 首页">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">moonnryan<span class="brand-subtitle">research workshop</span></span>
      </a>
      <button class="menu-toggle" :aria-expanded="mobileOpen" aria-controls="main-navigation" @click="mobileOpen = !mobileOpen">菜单</button>
      <nav id="main-navigation" class="main-nav" :class="{ 'is-open': mobileOpen }" aria-label="主导航">
        <a :class="{ active: route.path === withBase('/') }" :href="withBase('/')">首页</a>
        <a :class="{ active: isArticle || route.path.includes('/articles') }" :href="withBase('/articles.html')">文章</a>
        <a :class="{ active: frontmatter.layout === 'topics' }" :href="withBase('/topics.html')">主题</a>
        <a :class="{ active: isAbout }" :href="withBase('/about.html')">关于</a>
        <a class="github-link" href="https://github.com/moonnryan" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a>
        <button class="appearance-toggle" type="button" aria-label="切换深浅色模式" title="切换深浅色模式" @click="isDark = !isDark">
          <svg class="theme-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z" /></svg>
          <svg class="theme-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></svg>
        </button>
      </nav>
    </header>

    <main id="main-content">
      <div v-if="isArticle" class="article-layout">
        <article class="article-main">
          <a class="back-link" :href="withBase('/articles.html')">← 所有文章</a>
          <div class="article-meta">
            <span>{{ frontmatter.categoryName }}</span><span class="meta-dot">·</span>
            <span>约 {{ frontmatter.minutes }} 分钟</span>
          </div>
          <div v-if="frontmatter.tags?.length" class="article-tags article-page-tags" aria-label="文章标签">
            <a v-for="tag in frontmatter.tags" :key="tag" :href="withBase('/articles.html?tag=' + encodeURIComponent(tag))">{{ tag }}</a>
          </div>
          <Content :key="page.relativePath" class="vp-doc article-body" />
        </article>
        <aside v-if="headings.length" class="article-outline" aria-label="文章目录">
          <div class="outline-label">本文目录</div>
          <a v-for="heading in headings" :key="heading.id" :href="'#' + heading.id" :class="['outline-depth-' + heading.depth, { 'is-current': heading.id === activeHeading }]" :aria-current="heading.id === activeHeading ? 'location' : undefined">{{ heading.title }}</a>
        </aside>
        <RelatedArticles :url="route.path" />
      </div>
      <div v-else-if="isAbout" class="about-layout"><div class="section-eyebrow">ABOUT THE WORKSHOP</div><Content class="vp-doc" /></div>
      <Content v-else :key="page.relativePath" />
    </main>

    <footer class="site-footer">
      <span>Copyright © 2026. Mengran.Yang.</span>
      <a class="footer-note" href="https://vitepress.dev/" target="_blank" rel="noreferrer">Powered by VitePress ↗</a>
    </footer>
  </div>
</template>
