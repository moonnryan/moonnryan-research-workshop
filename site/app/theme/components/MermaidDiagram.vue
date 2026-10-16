<script setup lang="ts">
import { getCurrentInstance, onMounted, ref, watch } from 'vue'
import { useData } from 'vitepress'
const { isDark } = useData()
const props = defineProps<{ encoded: string }>()
const target = ref<HTMLElement>()
const error = ref('')
let sequence = 0
const instanceId = getCurrentInstance()?.uid ?? 0
async function render() {
  if (!target.value) return
  try {
    error.value = ''
    const { default: mermaid } = await import('mermaid')
    // SVG text avoids HTML labels inheriting article paragraph styles after measurement.
    await document.fonts.ready
    const dark = isDark.value
    mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'base', htmlLabels: false,
      flowchart: { useMaxWidth: true, padding: 18 }, themeVariables: {
      darkMode: dark,
      fontSize: '16px',
      fontFamily: 'system-ui, sans-serif', primaryColor: dark ? '#2b342e' : '#eeefe9', primaryTextColor: dark ? '#e0e6dc' : '#263c34',
      primaryBorderColor: dark ? '#728475' : '#aab6aa', lineColor: dark ? '#a0b09e' : '#6f8074', secondaryColor: dark ? '#242a25' : '#f8f7f3',
      tertiaryColor: dark ? '#242a25' : '#f2f4ec',
      edgeLabelBackground: dark ? '#232923' : '#f8f7f3',
    } })
    const bytes = Uint8Array.from(atob(props.encoded), character => character.charCodeAt(0))
    const code = new TextDecoder().decode(bytes)
    const id = `mrw-diagram-${instanceId}-${sequence++}`
    const { svg } = await mermaid.render(id, code)
    if (target.value) {
      target.value.innerHTML = svg
      const diagram = target.value.querySelector('svg')
      const bounds = diagram?.viewBox.baseVal
      if (diagram && bounds?.width && bounds.height) {
        // Fit the whole diagram into the article while retaining its aspect ratio.
        diagram.style.width = `${Math.ceil(bounds.width)}px`
        diagram.style.height = 'auto'
        diagram.style.maxWidth = '100%'
      }
    }
  } catch {
    error.value = '流程图暂时无法显示。'
  }
}
onMounted(render)
watch([() => props.encoded, isDark], render)
</script>

<template><figure class="mermaid-figure"><div v-if="error" role="status">{{ error }}</div><div v-else ref="target" aria-label="流程图"></div></figure></template>
