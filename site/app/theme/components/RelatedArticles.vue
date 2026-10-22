<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { articles } from '../content'

const props = defineProps<{ url: string }>()
const current = computed(() => articles.find(article => decodeURI(withBase(article.url)) === decodeURI(props.url)))
const tagFrequency = new Map<string, number>()
for (const article of articles) for (const tag of new Set(article.tags)) tagFrequency.set(tag, (tagFrequency.get(tag) ?? 0) + 1)
const related = computed(() => {
  const source = current.value
  if (!source) return []
  return articles.filter(article => article.url !== source.url).map(article => {
    // A specific shared tag matters more than a platform tag used across many articles.
    const shared = article.tags.filter(tag => source.tags.includes(tag))
      .reduce((score, tag) => score + 1 / Math.sqrt(tagFrequency.get(tag) ?? 1), 0)
    return { article, score: shared + Number(article.category === source.category) }
  }).filter(item => item.score > 0).sort((a, b) => b.score - a.score).slice(0, 3).map(item => item.article)
})
</script>

<template>
  <section v-if="current && related.length" class="related-section" aria-labelledby="related-title">
    <div class="related-heading">
      <h2 id="related-title">继续阅读</h2>
      <a class="subtle-link" :href="withBase('/articles.html?topic=' + current.category)">更多{{ current.categoryName }} →</a>
    </div>
    <div class="related-grid">
      <a v-for="article in related" :key="article.url" class="related-card" :href="withBase(article.url)">
        <div class="card-meta"><time :datetime="article.updated">{{ article.updated }}</time><span class="related-category">{{ article.categoryName }}</span></div>
        <h3>{{ article.title }}</h3>
        <span class="related-read">阅读全文 <span aria-hidden="true">→</span></span>
      </a>
    </div>
  </section>
</template>
