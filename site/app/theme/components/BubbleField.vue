<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { createBodies, settleBodies, stepBodies } from '../bubble-physics.mjs'

export type BubbleItem = { id: string; label: string; count: number; selected?: boolean }
const props = defineProps<{ items: BubbleItem[]; kind: 'topics' | 'tags'; paused?: boolean }>()
const emit = defineEmits<{ activate: [id: string] }>()
const field = ref<HTMLElement>()
const canvasHeight = ref(0)
const nodes = new Map<string, HTMLElement>()
let bodies: ReturnType<typeof createBodies> = []
let width = 0, height = 0, frame = 0, lastTime = 0, elapsed = 0
let resizeObserver: ResizeObserver | undefined
let visibility: IntersectionObserver | undefined
let visible = true
let motion: MediaQueryList | undefined
let mounted = false
let centered = false

function fallbackStyle(item: BubbleItem, index: number) {
  const anchors = [[22, 30], [49, 26], [76, 33], [34, 69], [68, 73]]
  const mobile = [[24, 16], [77, 21], [48, 47], [23, 76], [78, 80]]
  const maximum = Math.max(1, ...props.items.map(bubble => bubble.count))
  return props.kind === 'topics' ? {
    '--bubble-diameter': `${240 * Math.sqrt(item.count / maximum)}px`,
    '--bubble-x': `${anchors[index % 5][0]}%`, '--bubble-y': `${anchors[index % 5][1]}%`,
    '--bubble-mobile-x': `${mobile[index % 5][0]}%`, '--bubble-mobile-y': `${mobile[index % 5][1]}%`,
  } : undefined
}
function setNode(id: string, element: unknown) {
  if (element instanceof HTMLElement) nodes.set(id, element)
  else nodes.delete(id)
}
function draw() {
  for (const body of bodies) {
    const element = nodes.get(body.id)
    if (!element) continue
    element.style.left = '0'
    element.style.top = '0'
    element.style.width = `${body.radius * 2}px`
    element.style.height = `${body.radius * 2}px`
    element.style.transform = `translate3d(${body.x - body.radius}px, ${body.y - body.radius}px, 0)`
    element.style.setProperty('--bubble-diameter', `${body.radius * 2}px`)
  }
}
function tick(time: number) {
  frame = 0
  if (!visible || props.paused || motion?.matches || document.hidden) { lastTime = 0; return }
  const dt = lastTime ? (time - lastTime) / 1000 : 1 / 60
  lastTime = time; elapsed += Math.min(dt, .04)
  stepBodies(bodies, width, height, elapsed, dt, props.kind)
  draw()
  frame = requestAnimationFrame(tick)
}
function resume() {
  if (!mounted) return
  if (motion?.matches) {
    for (const body of bodies) { body.radius = body.baseRadius; body.hovered = false }
    settleBodies(bodies, width, height, props.kind === 'tags'); draw()
  }
  if (!frame && visible && !props.paused && !motion?.matches && !document.hidden) frame = requestAnimationFrame(tick)
}
function hover(id: string, hovered: boolean) {
  if (motion?.matches) return
  for (const body of bodies) if (body.id === id) body.hovered = hovered
  resume()
}
function layout() {
  const element = field.value
  if (!element?.clientWidth) return
  width = element.clientWidth
  if (props.kind === 'tags') {
    const unit = width < 500 ? 24 : 26
    const area = props.items.reduce((sum, item) => sum + Math.PI * (unit * Math.sqrt(item.count) + 4) ** 2, 0)
    height = Math.max(element.parentElement?.clientHeight ?? 0, Math.ceil(area / (Math.PI / 4 * width * .57)))
    canvasHeight.value = height
    if (!centered) {
      centered = true
      nextTick(() => {
        const viewport = element.parentElement
        if (mounted && viewport) viewport.scrollTop = Math.max(0, (height - viewport.clientHeight) / 2)
      })
    }
  } else height = element.clientHeight
  if (!height) return
  bodies = createBodies(props.items, width, height, props.kind)
  settleBodies(bodies, width, height, props.kind === 'tags')
  draw(); resume()
}
watch(() => props.items.map(item => `${item.id}:${item.count}`).join('|'), async () => { await nextTick(); layout() })
watch(() => props.paused, resume)
onMounted(() => {
  mounted = true
  motion = window.matchMedia('(prefers-reduced-motion: reduce)')
  motion.addEventListener('change', resume)
  document.addEventListener('visibilitychange', resume)
  resizeObserver = new ResizeObserver(layout)
  resizeObserver.observe(field.value!)
  if (props.kind === 'tags' && field.value?.parentElement) resizeObserver.observe(field.value.parentElement)
  visibility = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; resume() })
  visibility.observe(field.value!)
  layout()
})
onUnmounted(() => {
  mounted = false
  cancelAnimationFrame(frame)
  resizeObserver?.disconnect(); visibility?.disconnect()
  motion?.removeEventListener('change', resume)
  document.removeEventListener('visibilitychange', resume)
})
</script>

<template>
  <div ref="field" :class="kind === 'topics' ? 'bubble-stage' : 'tag-cloud-canvas'" :style="kind === 'tags' ? { height: canvasHeight + 'px' } : undefined">
    <div v-for="(item, index) in items" :key="item.id" :ref="element => setNode(item.id, element)" :data-bubble-id="item.id" :style="fallbackStyle(item, index)" :class="kind === 'topics' ? 'topic-bubble-position' : 'tag-bubble-position'">
      <button :class="[kind === 'topics' ? 'topic-bubble' : 'tag-bubble', { selected: item.selected }]" :aria-label="`${kind === 'tags' ? '#' : ''}${item.label}，${item.count} 篇文章`" :aria-haspopup="kind === 'topics' ? 'dialog' : undefined" :aria-pressed="kind === 'tags' ? !!item.selected : undefined" @pointerenter="hover(item.id, true)" @pointerleave="hover(item.id, false)" @pointercancel="hover(item.id, false)" @focus="hover(item.id, true)" @blur="hover(item.id, false)" @click="emit('activate', item.id)">
        <span :class="kind === 'topics' ? 'bubble-name' : 'tag-bubble-label'">{{ kind === 'tags' ? '#' : '' }}{{ item.label }}</span>
        <span :class="kind === 'topics' ? 'bubble-count' : 'tag-bubble-count'">{{ item.selected ? '✓ ' : '' }}{{ item.count }}{{ kind === 'topics' ? ' notes' : '' }}</span>
        <span v-if="kind === 'topics'" class="bubble-open" aria-hidden="true">↗</span>
      </button>
    </div>
  </div>
</template>
