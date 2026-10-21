<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { articles, categories } from '../content'

const groups = computed(() => categories.map(([key, name]) => {
  const items = articles.filter(article => article.category === key)
  const counts = new Map<string, number>()
  for (const article of items) for (const tag of new Set(article.tags)) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  const tags = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh-CN'))
  return { key, name, count: items.length, tags }
}))
const largest = computed(() => Math.max(1, ...groups.value.map(group => group.count)))
</script>

<template>
  <section class="topics-page" aria-labelledby="topics-title">
    <div class="topics-header"><h1 id="topics-title">主题</h1><span class="topics-count">{{ groups.length }} topics · {{ articles.length }} notes</span></div>
    <nav class="topic-cloud" aria-label="主题导航">
      <a v-for="group in groups" :key="group.key" :href="'#' + group.key" :style="{ '--topic-size': `${24 + 20 * group.count / largest}px` }">{{ group.name }}<sup>{{ group.count }}</sup></a>
    </nav>
    <div class="topic-groups">
      <section v-for="group in groups" :id="group.key" :key="group.key" class="topic-group" :aria-labelledby="'topic-' + group.key">
        <div class="topic-group-heading"><h2 :id="'topic-' + group.key">{{ group.name }}</h2><a class="subtle-link" :href="withBase('/articles.html?topic=' + group.key)">{{ group.count }} 篇文章 →</a></div>
        <div class="topic-tags">
          <a v-for="[tag, count] in group.tags" :key="tag" :href="withBase('/articles.html?topic=' + group.key + '&tag=' + encodeURIComponent(tag))"><span>#{{ tag }}</span><span class="tag-count">{{ count }}</span></a>
        </div>
      </section>
    </div>
  </section>
</template>
