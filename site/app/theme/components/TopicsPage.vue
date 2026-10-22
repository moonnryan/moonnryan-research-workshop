<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { articles, categories, countTags, filterUrl, normalizeTopic } from '../content'

const groups = computed(() => categories.map(([key, name]) => {
  const items = articles.filter(article => article.category === key)
  return { key, name, count: items.length, items, tags: countTags(items) }
}))
const largest = computed(() => Math.max(1, ...groups.value.map(group => group.count)))
const activeKey = ref('')
const selectedTags = ref<string[]>([])
const dialog = ref<HTMLDialogElement>()
const active = computed(() => groups.value.find(group => group.key === activeKey.value))
const matchingCount = computed(() => active.value?.items.filter(article => !selectedTags.value.length || selectedTags.value.some(tag => article.tags.includes(tag))).length ?? 0)
const positions = [[20, 29], [50, 23], [79, 32], [31, 72], [69, 74]]
const mobilePositions = [[24, 16], [77, 21], [48, 47], [23, 76], [78, 80]]

function bubbleStyle(count: number, index: number) {
  return {
    '--bubble-diameter': `${240 * Math.sqrt(count / largest.value)}px`,
    '--bubble-x': `${positions[index % positions.length][0]}%`,
    '--bubble-y': `${positions[index % positions.length][1]}%`,
    '--bubble-mobile-x': `${mobilePositions[index % mobilePositions.length][0]}%`,
    '--bubble-mobile-y': `${mobilePositions[index % mobilePositions.length][1]}%`,
    '--float-duration': `${11 + index * 1.7}s`,
    '--float-delay': `${-index * 2.3}s`,
  }
}

async function openGroup(key: string) {
  if (!groups.value.some(group => group.key === key)) return
  activeKey.value = key
  selectedTags.value = []
  await nextTick()
  if (!dialog.value?.open) dialog.value?.showModal()
  window.history.replaceState({}, '', window.location.pathname + window.location.search + '#' + key)
}

function closeGroup() {
  dialog.value?.close()
}

function afterClose() {
  activeKey.value = ''
  selectedTags.value = []
  if (window.location.pathname.endsWith('/topics.html')) {
    window.history.replaceState({}, '', window.location.pathname + window.location.search)
  }
}

function toggleTag(tag: string) {
  selectedTags.value = selectedTags.value.includes(tag) ? selectedTags.value.filter(item => item !== tag) : [...selectedTags.value, tag]
}

function readHash() {
  const key = normalizeTopic(decodeURIComponent(window.location.hash.slice(1)))
  if (key) openGroup(key)
}
onMounted(() => {
  readHash()
  window.addEventListener('hashchange', readHash)
})
onUnmounted(() => {
  dialog.value?.close()
  window.removeEventListener('hashchange', readHash)
})
</script>

<template>
  <section class="topics-page" aria-labelledby="topics-title">
    <div class="topics-header"><h1 id="topics-title">主题</h1><span class="topics-count">{{ groups.length }} topics · {{ articles.length }} notes</span></div>
    <p class="topics-intro">选择一个主题，按标签继续探索</p>
    <div class="bubble-stage" aria-label="主题泡泡">
      <div v-for="(group, index) in groups" :key="group.key" class="topic-bubble-position" :style="bubbleStyle(group.count, index)">
        <button class="topic-bubble" aria-haspopup="dialog" :aria-label="`${group.name}，${group.count} 篇文章`" @click="openGroup(group.key)"><span class="bubble-name">{{ group.name }}</span><span class="bubble-count">{{ group.count }} notes</span><span class="bubble-open" aria-hidden="true">↗</span></button>
      </div>
    </div>
    <dialog ref="dialog" class="topic-dialog" aria-labelledby="bubble-dialog-title" @close="afterClose" @click="event => { if (event.target === dialog) closeGroup() }">
      <div v-if="active" class="expanded-bubble">
        <button class="bubble-close" aria-label="关闭主题" autofocus @click="closeGroup">×</button>
        <div class="expanded-bubble-heading"><h2 id="bubble-dialog-title">{{ active.name }}</h2><p>{{ active.count }} 篇文章<span> · {{ selectedTags.length ? '匹配任一已选标签' : '选择标签，可多选' }}</span></p></div>
        <div class="tag-orbit" aria-label="选择标签">
          <button v-for="([tag, count], index) in active.tags" :key="tag" class="tag-bubble" :class="{ selected: selectedTags.includes(tag) }" :aria-pressed="selectedTags.includes(tag)" :style="{ '--tag-delay': `${-index * .43}s`, '--tag-duration': `${5 + index % 5}s` }" @click="toggleTag(tag)"><span>#{{ tag }}</span><span class="tag-bubble-count">{{ selectedTags.includes(tag) ? '✓ ' : '' }}{{ count }}</span></button>
        </div>
        <div class="bubble-destination">
          <button v-if="selectedTags.length" class="bubble-clear" @click="selectedTags = []">清除 {{ selectedTags.length }} 个标签</button>
          <a class="topic-go-bubble" :href="filterUrl(active.key, selectedTags)"><span>前往 →</span><span>{{ matchingCount }} 篇文章</span></a>
        </div>
      </div>
    </dialog>
  </section>
</template>
