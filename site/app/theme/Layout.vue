<script setup lang="ts">
import { computed, nextTick, onMounted, watch, ref } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import { isPreview } from './content'

const { page, frontmatter } = useData()
const route = useRoute()
const isArticle = computed(() => frontmatter.value.layout === 'article')
const isAbout = computed(() => frontmatter.value.layout === 'about')
const headings = ref<{ title: string; id: string }[]>([])
const mobileOpen = ref(false)

async function updateHeadings() {
  await nextTick()
  headings.value = [...document.querySelectorAll('.article-body h1[id], .article-body h2[id], .article-body h3[id]')]
    .slice(1).map(element => ({
      id: element.id,
      title: (element.textContent ?? '').replace(/\u200b/g, '').trim(),
    }))
}
onMounted(updateHeadings)
watch(() => route.path, () => { mobileOpen.value = false; updateHeadings() })
</script>

<template>
  <div class="workshop-shell" :class="{ 'is-home': route.path === withBase('/'), 'is-listing': route.path === withBase('/articles.html') }">
    <a class="skip-link" href="#main-content">跳到正文</a>
    <header class="site-header">
      <a class="brand" :href="withBase('/')" aria-label="Moonnryan's Research Workshop 首页">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">moonnryan<span class="brand-subtitle">research workshop</span></span>
      </a>
      <button class="menu-toggle" :aria-expanded="mobileOpen" aria-controls="main-navigation" @click="mobileOpen = !mobileOpen">菜单</button>
      <nav id="main-navigation" class="main-nav" :class="{ 'is-open': mobileOpen }" aria-label="主导航">
        <a :class="{ active: route.path === withBase('/') }" :href="withBase('/')">首页</a>
        <a :class="{ active: isArticle || route.path.includes('/articles') }" :href="withBase('/articles.html')">文章</a>
        <a :class="{ active: isAbout }" :href="withBase('/about.html')">关于</a>
        <a class="github-link" href="https://github.com/moonnryan" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a>
      </nav>
    </header>

    <main id="main-content">
      <div v-if="isArticle" class="article-layout">
        <article class="article-main">
          <a class="back-link" :href="withBase('/articles.html')">← 所有文章</a>
          <div class="article-meta">
            <span>{{ frontmatter.categoryName }}</span><span class="meta-dot">·</span>
            <span>约 {{ frontmatter.minutes }} 分钟</span>
            <span v-if="isPreview && frontmatter.status !== 'ready'" class="draft-label">草稿</span>
          </div>
          <div v-if="frontmatter.tags?.length" class="article-tags article-page-tags" aria-label="文章标签">
            <a v-for="tag in frontmatter.tags" :key="tag" :href="withBase('/articles.html?tag=' + encodeURIComponent(tag))">{{ tag }}</a>
          </div>
          <Content :key="page.relativePath" class="vp-doc article-body" />
          <div class="article-end"><span>读到这里，感谢你的时间。</span><a :href="withBase('/articles.html')">继续阅读 ↗</a></div>
        </article>
        <aside v-if="headings.length" class="article-outline" aria-label="文章目录">
          <div class="outline-label">本文目录</div>
          <a v-for="heading in headings" :key="heading.id" :href="'#' + heading.id">{{ heading.title }}</a>
        </aside>
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
