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
const compactFace = ref<HTMLElement>()
const hiddenSource = ref('')
const phase = ref<'closed' | 'opening' | 'open' | 'closing'>('closed')
const active = computed(() => groups.value.find(group => group.key === activeKey.value))
const matchingCount = computed(() => active.value?.items.filter(article => !selectedTags.value.length || selectedTags.value.some(tag => article.tags.includes(tag))).length ?? 0)
const topicBubbles = computed(() => groups.value.map(group => ({ id: group.key, label: group.name, count: group.count })))
const tagBubbles = computed(() => active.value?.tags.map(([tag, count]) => ({ id: tag, label: tag, count, selected: selectedTags.value.includes(tag) })) ?? [])
const easing = 'cubic-bezier(.16, 1, .3, 1)'
let version = 0
let morph: Animation | undefined
let contentAnimations: Animation[] = []
let faceAnimation: Animation | undefined
let faceFrame = 0

function prepareFace(source?: HTMLElement) {
  const face = compactFace.value
  const button = source?.querySelector('button')
  if (!face || !source || !button) return
  const rect = source.getBoundingClientRect()
  face.style.width = `${rect.width}px`
  face.style.height = `${rect.height}px`
  face.style.setProperty('--bubble-diameter', `${rect.width}px`)
  const copy = button.cloneNode(true) as HTMLButtonElement
  copy.tabIndex = -1
  copy.style.cssText = 'background: transparent; border: 0; box-shadow: none; pointer-events: none'
  face.replaceChildren(copy)
}

function animateFace(opening: boolean, currentOpacity = '0') {
  faceAnimation?.cancel()
  cancelAnimationFrame(faceFrame)
  const face = compactFace.value!
  // Keep the compact label at its original size while its surrounding circle morphs.
  const sync = () => {
    const shell = dialog.value?.querySelector<HTMLElement>('.expanded-bubble')
    if (!shell) return
    const matrix = new DOMMatrixReadOnly(getComputedStyle(shell).transform)
    face.style.transform = `translate(-50%, -50%) scale(${1 / matrix.a}, ${1 / matrix.d})`
    faceFrame = requestAnimationFrame(sync)
  }
  sync()
  faceAnimation = face.animate([
    { opacity: opening ? 1 : currentOpacity },
    { opacity: opening ? 0 : 1 },
  ], { duration: opening ? 300 : 240, easing: opening ? 'cubic-bezier(.4, 0, 1, 1)' : easing, fill: 'both' })
}

function stopFace() {
  cancelAnimationFrame(faceFrame)
  faceAnimation?.cancel()
}

function sourceElement(key = activeKey.value) {
  return [...document.querySelectorAll<HTMLElement>('.bubble-stage [data-bubble-id]')].find(element => element.dataset.bubbleId === key)
}
function transformFrom(source: DOMRect, target: DOMRect) {
  return `translate(${source.left + source.width / 2 - target.left - target.width / 2}px, ${source.top + source.height / 2 - target.top - target.height / 2}px) scale(${source.width / target.width}, ${source.height / target.height})`
}
function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}
function animateContent(opening: boolean) {
  const elements = [...dialog.value!.querySelectorAll<HTMLElement>('.expanded-bubble-heading, .tag-orbit, .bubble-destination')]
  const opacities = elements.map(element => getComputedStyle(element).opacity)
  contentAnimations.forEach(animation => animation.cancel())
  contentAnimations = elements.map((element, index) => element.animate([
    { opacity: opening ? 0 : opacities[index], transform: opening ? 'translateY(10px)' : 'none' },
    { opacity: opening ? 1 : 0, transform: opening ? 'none' : 'translateY(-5px)' },
  ], { duration: opening ? 330 : 140, delay: opening ? 140 + index * 35 : 0, easing, fill: 'both' }))
}

function dismissOutside(event: PointerEvent) {
  const bubble = dialog.value?.querySelector<HTMLElement>('.expanded-bubble')
  if (!bubble) return
  const rect = bubble.getBoundingClientRect()
  const x = (event.clientX - rect.left - rect.width / 2) / (rect.width / 2)
  const y = (event.clientY - rect.top - rect.height / 2) / (rect.height / 2)
  if (x * x + y * y > 1) closeGroup()
}

async function openGroup(key: string, origin?: DOMRect) {
  if (!groups.value.some(group => group.key === key)) return
  if (activeKey.value === key && (phase.value === 'opening' || phase.value === 'open')) return
  const ticket = ++version
  const source = origin ?? sourceElement(key)?.getBoundingClientRect()
  const sourceNode = sourceElement(key)
  morph?.cancel()
  stopFace()
  contentAnimations.forEach(animation => animation.cancel())
  activeKey.value = key
  hiddenSource.value = key
  selectedTags.value = []
  phase.value = 'opening'
  await nextTick()
  if (ticket !== version || !dialog.value) return
  if (!dialog.value.open) dialog.value.showModal()
  window.history.replaceState({}, '', window.location.pathname + window.location.search + '#' + key)
  const bubble = dialog.value.querySelector<HTMLElement>('.expanded-bubble')!
  const target = dialog.value.getBoundingClientRect()
  prepareFace(sourceNode)
  if (!reducedMotion()) {
    morph = bubble.animate([
      { transform: source ? transformFrom(source, target) : 'scale(.96)' },
      { transform: 'none' },
    ], { duration: 620, easing, fill: 'both' })
    animateFace(true)
    animateContent(true)
    await morph.finished.catch(() => {})
  }
  if (ticket !== version) return
  morph?.cancel()
  stopFace()
  contentAnimations.forEach(animation => animation.cancel())
  phase.value = 'open'
}

async function closeGroup() {
  if (!dialog.value?.open || phase.value === 'closing') return
  const ticket = ++version
  const bubble = dialog.value.querySelector<HTMLElement>('.expanded-bubble')!
  const current = getComputedStyle(bubble).transform
  const backdrop = getComputedStyle(dialog.value, '::backdrop').opacity
  const target = dialog.value.getBoundingClientRect()
  const destination = sourceElement()?.getBoundingClientRect()
  const faceOpacity = compactFace.value ? getComputedStyle(compactFace.value).opacity : '0'
  morph?.cancel()
  dialog.value.style.setProperty('--backdrop-from', backdrop)
  phase.value = 'closing'
  if (!reducedMotion()) {
    morph = bubble.animate([
      { transform: current },
      { transform: destination ? transformFrom(destination, target) : 'scale(.96)' },
    ], { duration: 420, easing, fill: 'both' })
    animateFace(false, faceOpacity)
    animateContent(false)
    await morph.finished.catch(() => {})
  }
  if (ticket !== version) return
  // Reveal the stationary source before removing the modal from the top layer.
  hiddenSource.value = ''
  await nextTick()
  if (ticket === version) dialog.value?.close()
}

async function afterClose() {
  if (dialog.value?.open) return
  const source = sourceElement()?.querySelector<HTMLButtonElement>('button')
  morph?.cancel()
  stopFace()
  contentAnimations.forEach(animation => animation.cancel())
  activeKey.value = ''
  hiddenSource.value = ''
  phase.value = 'closed'
  selectedTags.value = []
  if (window.location.pathname.endsWith('/topics.html')) {
    window.history.replaceState({}, '', window.location.pathname + window.location.search)
    await nextTick()
    source?.focus({ preventScroll: true })
  }
}

function toggleTag(tag: string) {
  if (phase.value !== 'open') return
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
  version++
  morph?.cancel()
  stopFace()
  contentAnimations.forEach(animation => animation.cancel())
  dialog.value?.close()
  window.removeEventListener('hashchange', readHash)
})
</script>

<template>
  <section class="topics-page" aria-labelledby="topics-title">
    <div class="topics-header"><p id="topics-title" class="topics-intro">选择一个主题，按标签继续探索</p><span class="topics-count">{{ groups.length }} topics · {{ articles.length }} notes</span></div>
    <BubbleField :items="topicBubbles" kind="topics" :paused="!!activeKey" :hidden-id="hiddenSource" @activate="openGroup" />
    <dialog ref="dialog" class="topic-dialog" :data-phase="phase" autofocus aria-labelledby="bubble-dialog-title" aria-describedby="bubble-dismiss-hint" @close="afterClose" @cancel.prevent="closeGroup()" @pointerdown="dismissOutside">
      <div v-if="active" class="expanded-bubble">
        <div ref="compactFace" class="bubble-morph-face" aria-hidden="true" inert></div>
        <span id="bubble-dismiss-hint" class="visually-hidden">点击圆圈外或按 Escape 关闭</span>
        <div class="expanded-bubble-heading"><h2 id="bubble-dialog-title">{{ active.name }}</h2><p>{{ active.count }} 篇文章<span> · {{ selectedTags.length ? '匹配任一已选标签' : '选择标签，可多选' }}</span></p></div>
        <div class="tag-orbit" aria-label="选择标签" :inert="phase !== 'open'">
          <BubbleField :items="tagBubbles" kind="tags" :paused="phase !== 'open'" @activate="toggleTag" />
        </div>
        <div class="bubble-destination" :inert="phase !== 'open'">
          <button class="bubble-clear" :class="{ 'is-empty': !selectedTags.length }" :disabled="!selectedTags.length" @click="selectedTags = []">清除 {{ selectedTags.length }} 个标签</button>
          <a class="topic-go-bubble" :href="filterUrl(active.key, selectedTags)"><span>前往 →</span><span>{{ matchingCount }} 篇文章</span></a>
        </div>
      </div>
    </dialog>
  </section>
</template>
