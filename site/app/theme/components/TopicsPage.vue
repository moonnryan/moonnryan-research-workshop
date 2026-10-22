<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { articles, categories, countTags, filterUrl, normalizeTopic } from '../content'
import BubbleField from './BubbleField.vue'

const groups = computed(() => categories.map(([key, name]) => {
  const items = articles.filter(article => article.category === key)
  return { key, name, count: items.length, items, tags: countTags(items) }
}))
const activeKey = ref('')
const selectedTags = ref<string[]>([])
const dialog = ref<HTMLDialogElement>()
const active = computed(() => groups.value.find(group => group.key === activeKey.value))
const matchingCount = computed(() => active.value?.items.filter(article => !selectedTags.value.length || selectedTags.value.some(tag => article.tags.includes(tag))).length ?? 0)
const topicBubbles = computed(() => groups.value.map(group => ({ id: group.key, label: group.name, count: group.count })))
const tagBubbles = computed(() => active.value?.tags.map(([tag, count]) => ({ id: tag, label: tag, count, selected: selectedTags.value.includes(tag) })) ?? [])

function dismissOutside(event: PointerEvent) {
  const bubble = dialog.value?.querySelector<HTMLElement>('.expanded-bubble')
  if (!bubble) return
  const rect = bubble.getBoundingClientRect()
  const x = (event.clientX - rect.left - rect.width / 2) / (rect.width / 2)
  const y = (event.clientY - rect.top - rect.height / 2) / (rect.height / 2)
  if (x * x + y * y > 1) closeGroup()
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
  // A queued close event may arrive after the user has already opened another topic.
  if (dialog.value?.open) return
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
    <BubbleField :items="topicBubbles" kind="topics" :paused="!!activeKey" @activate="openGroup" />
    <dialog ref="dialog" class="topic-dialog" tabindex="-1" autofocus aria-labelledby="bubble-dialog-title" aria-describedby="bubble-dismiss-hint" @close="afterClose" @pointerdown="dismissOutside">
      <div v-if="active" class="expanded-bubble">
        <span id="bubble-dismiss-hint" class="visually-hidden">点击圆圈外或按 Escape 关闭</span>
        <div class="expanded-bubble-heading"><h2 id="bubble-dialog-title">{{ active.name }}</h2><p>{{ active.count }} 篇文章<span> · {{ selectedTags.length ? '匹配任一已选标签' : '选择标签，可多选' }}</span></p></div>
        <div class="tag-orbit" aria-label="选择标签">
          <BubbleField :items="tagBubbles" kind="tags" @activate="toggleTag" />
        </div>
        <div class="bubble-destination">
          <button v-if="selectedTags.length" class="bubble-clear" @click="selectedTags = []">清除 {{ selectedTags.length }} 个标签</button>
          <a class="topic-go-bubble" :href="filterUrl(active.key, selectedTags)"><span>前往 →</span><span>{{ matchingCount }} 篇文章</span></a>
        </div>
      </div>
    </dialog>
  </section>
</template>
