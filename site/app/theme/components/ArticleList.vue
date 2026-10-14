<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import { articles, categories } from '../content'
const query = ref('')
const selected = ref('all')
const selectedTag = ref('')
const pageSize = 10
const currentPage = ref(1)

function readTag() {
  selectedTag.value = new URLSearchParams(window.location.search).get('tag') ?? ''
}
onMounted(() => {
  readTag()
  window.addEventListener('workshop-route-change', readTag)
})
onUnmounted(() => window.removeEventListener('workshop-route-change', readTag))

const filtered = computed(() => articles.filter(article => {
  const matchesCategory = selected.value === 'all' || article.category === selected.value
  const matchesTag = !selectedTag.value || article.tags.includes(selectedTag.value)
  const term = query.value.trim().toLowerCase()
  const searchable = [article.title, article.excerpt, article.categoryName, ...article.tags].join(' ').toLowerCase()
  return matchesCategory && matchesTag && (!term || searchable.includes(term))
}))

const pageCount = computed(() => Math.max(1, Math.ceil(filtered.value.length / pageSize)))
const pageArticles = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return filtered.value.slice(start, start + pageSize)
})
watch([query, selected, selectedTag], () => { currentPage.value = 1 }, { flush: 'sync' })
watch(pageCount, count => { currentPage.value = Math.min(currentPage.value, count) }, { flush: 'sync' })

function goToPage(page: number) {
  currentPage.value = Math.min(Math.max(page, 1), pageCount.value)
  document.getElementById('listing-title')?.scrollIntoView({ block: 'start' })
}

function clearFilters() {
  query.value = ''
  selected.value = 'all'
  selectedTag.value = ''
  window.history.replaceState({}, '', withBase('/articles.html'))
}
</script>

<template>
  <section class="listing-page" aria-labelledby="listing-title">
    <div class="listing-header"><h1 id="listing-title" class="heading-count"><span class="heading-count-number">{{ String(articles.length).padStart(2, '0') }}</span><span class="heading-count-label">total notes</span></h1></div>
    <div class="list-toolbar">
      <div class="category-filters" aria-label="按主题筛选"><button :class="{ selected: selected === 'all' }" @click="selected = 'all'">全部</button><button v-for="[key, name] in categories" :key="key" :class="{ selected: selected === key }" @click="selected = key">{{ name }}</button></div>
      <div class="search-field"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></svg><input v-model="query" type="search" aria-label="搜索文章" placeholder="搜索标题或关键词"></div>
    </div>
    <div v-if="selectedTag" class="active-tag">标签 <strong>#{{ selectedTag }}</strong><button @click="clearFilters">清除</button></div>
    <div v-if="filtered.length" class="all-articles">
      <article v-for="article in pageArticles" :key="article.url" class="list-article">
        <div class="list-article-main"><div class="card-meta"><span>{{ article.categoryName }}</span><span class="meta-dot">·</span><span>约 {{ article.minutes }} 分钟</span></div><h2><a :href="withBase(article.url)">{{ article.title }}</a></h2><p><a :href="withBase(article.url)">{{ article.excerpt }}</a></p><div class="article-tags"><a v-for="tag in article.tags" :key="tag" :href="withBase('/articles.html?tag=' + encodeURIComponent(tag))">{{ tag }}</a></div></div>
        <div class="list-article-side"><span>{{ article.updated }}</span><a class="row-arrow" :href="withBase(article.url)" :aria-label="'阅读' + article.title">↗</a></div>
      </article>
    </div>
    <div v-else class="empty-state"><span>{{ articles.length ? '没有找到匹配的文章。' : '文章正在整理中。' }}</span><p>{{ articles.length ? '试试其他关键词、主题或标签。' : '完成的记录会陆续出现在这里。' }}</p><button v-if="articles.length" class="text-link" @click="clearFilters">清除筛选 ↗</button></div>
    <nav v-if="pageCount > 1" class="article-pagination" aria-label="文章分页">
      <button :disabled="currentPage === 1" @click="goToPage(currentPage - 1)">← 上一页</button>
      <div class="pagination-pages">
        <button v-for="page in pageCount" :key="page" :class="{ selected: currentPage === page }" :aria-current="currentPage === page ? 'page' : undefined" :aria-label="'第 ' + page + ' 页'" @click="goToPage(page)">{{ page }}</button>
      </div>
      <button :disabled="currentPage === pageCount" @click="goToPage(currentPage + 1)">下一页 →</button>
    </nav>
  </section>
</template>
