<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { articles, categories, isPreview } from '../content'
const query = ref('')
const selected = ref('all')
const filtered = computed(() => articles.filter(article => {
  const matchesCategory = selected.value === 'all' || article.category === selected.value
  const term = query.value.trim().toLowerCase()
  const searchable = [article.title, article.excerpt, article.categoryName, ...article.tags].join(' ').toLowerCase()
  return matchesCategory && (!term || searchable.includes(term))
}))
</script>

<template>
  <section class="listing-page" aria-labelledby="listing-title">
    <div class="listing-header"><h1 id="listing-title" class="heading-count"><span class="heading-count-number">{{ String(articles.length).padStart(2, '0') }}</span><span class="heading-count-label">total notes</span></h1></div>
    <div class="list-toolbar">
      <div class="category-filters" aria-label="按主题筛选"><button :class="{ selected: selected === 'all' }" @click="selected = 'all'">全部</button><button v-for="[key, name] in categories" :key="key" :class="{ selected: selected === key }" @click="selected = key">{{ name }}</button></div>
      <div class="search-field"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></svg><input v-model="query" type="search" aria-label="搜索文章" placeholder="搜索标题或关键词"></div>
    </div>
    <div v-if="filtered.length" class="all-articles">
      <a v-for="article in filtered" :key="article.url" class="list-article" :href="withBase(article.url)">
        <div class="list-article-main"><div class="card-meta"><span>{{ article.categoryName }}</span><span class="meta-dot">·</span><span>约 {{ article.minutes }} 分钟</span><span v-if="isPreview && article.draft" class="draft-label">草稿</span></div><h2>{{ article.title }}</h2><p>{{ article.excerpt }}</p><div class="article-tags"><span v-for="tag in article.tags.slice(0, 4)" :key="tag">{{ tag }}</span></div></div>
        <div class="list-article-side"><span>{{ article.updated }}</span><span class="row-arrow" aria-hidden="true">↗</span></div>
      </a>
    </div>
    <div v-else class="empty-state"><span>{{ articles.length ? '没有找到匹配的文章。' : '文章正在整理中。' }}</span><p>{{ articles.length ? '试试其他关键词或主题。' : '完成的记录会陆续出现在这里。' }}</p><button v-if="articles.length" class="text-link" @click="query = ''; selected = 'all'">清除筛选 ↗</button></div>
  </section>
</template>
