import data from '../../data/articles.json'
import { withBase } from 'vitepress'
import { topicLabels, normalizeTopic } from '../topic-groups.mjs'

export const articles = data.articles
export const isPreview = data.preview
export const categories = Object.entries(topicLabels).filter(([key]) => articles.some(article => article.category === key))
export { normalizeTopic }

export function filterUrl(topic = 'all', tags: string[] = []) {
  const params = new URLSearchParams()
  if (topic !== 'all') params.set('topic', topic)
  for (const tag of tags) params.append('tag', tag)
  const query = params.toString()
  return withBase('/articles.html') + (query ? '?' + query : '')
}

export function countTags(items: typeof articles) {
  const counts = new Map<string, number>()
  for (const article of items) for (const tag of new Set(article.tags)) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], 'zh-CN'))
}
