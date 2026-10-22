import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import matter from 'gray-matter'
import { topicLabels, normalizeTopic } from '../app/topic-groups.mjs'

const site = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const repository = path.dirname(site)
const destination = path.join(site, 'content')
const preview = process.argv.includes('--preview') || process.env.MRW_INCLUDE_DRAFTS === '1'

async function files(directory) {
  const result = []
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name)
    if (entry.isDirectory()) result.push(...await files(target))
    else if (entry.name.endsWith('.md')) result.push(target)
  }
  return result
}

function excerpt(content) {
  return content.replace(/```[\s\S]*?```/g, '')
    .split(/\n\s*\n/)
    .find(paragraph => paragraph.trim() && !/^[#|!]/.test(paragraph.trim()))
    ?.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[`*$]/g, '').replace(/\s+/g, ' ').slice(0, 128) ?? ''
}

function date(value) {
  return value instanceof Date ? value.toISOString().slice(0, 10) : String(value ?? '').slice(0, 10)
}

async function writeChanged(file, content) {
  let old
  try { old = await fs.readFile(file, 'utf8') } catch {}
  if (old !== content) await fs.writeFile(file, content)
}

async function copyChanged(source, target) {
  await fs.mkdir(target, { recursive: true })
  for (const entry of await fs.readdir(source, { withFileTypes: true })) {
    if (entry.name.endsWith('.tex') || entry.name === '.DS_Store') continue
    const input = path.join(source, entry.name)
    const output = path.join(target, entry.name)
    if (entry.isDirectory()) { await copyChanged(input, output); continue }
    const bytes = await fs.readFile(input)
    let old
    try { old = await fs.readFile(output) } catch {}
    if (!old || !old.equals(bytes)) await fs.writeFile(output, bytes)
  }
}

await fs.mkdir(destination, { recursive: true })
// Clear stale mode-specific pages when switching between private preview and public output.
let previousMode
try { previousMode = JSON.parse(await fs.readFile(path.join(destination, 'data/articles.json'), 'utf8')).preview } catch {}
if (previousMode !== preview) {
  for (const entry of await fs.readdir(destination)) {
    if (!['.vitepress', 'data'].includes(entry)) await fs.rm(path.join(destination, entry), { recursive: true, force: true })
  }
}
await fs.mkdir(path.join(destination, '.vitepress'), { recursive: true })
await copyChanged(path.join(site, 'app'), path.join(destination, '.vitepress'))
await copyChanged(path.join(site, 'pages'), destination)
await copyChanged(path.join(repository, 'assets', 'media'), path.join(destination, 'assets', 'media'))

const articles = []
const selectedFiles = new Set()
for (const source of await files(path.join(repository, 'topics'))) {
  const text = await fs.readFile(source, 'utf8')
  const { data, content } = matter(text)
  if (!preview && !(data.blog === true && data.status === 'ready')) continue
  const relative = path.relative(repository, source).split(path.sep).join('/')
  const category = normalizeTopic(relative.split('/')[1])
  const body = content.replace(/```mermaid\s*\n([\s\S]*?)```/g, (_, diagram) => {
    const encoded = Buffer.from(diagram.trim()).toString('base64')
    return `<MermaidDiagram encoded="${encoded}" />`
  })
  const readingText = content.replace(/```[\s\S]*?```/g, '')
  const chinese = (readingText.match(/[\u3400-\u9fff]/g) ?? []).length
  const words = (readingText.match(/[A-Za-z]+/g) ?? []).length
  const created = date(data.created)
  const updated = date(data.updated ?? data.created)
  const article = {
    title: String(data.title ?? path.basename(source, '.md')),
    url: '/' + relative.replace(/\.md$/, '.html'),
    category,
    categoryName: topicLabels[category] ?? category,
    excerpt: excerpt(content),
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    created,
    updated,
    minutes: Math.max(2, Math.ceil(chinese / 450 + words / 220)),
    draft: data.status !== 'ready',
    tutorial: category === 'tutorial',
  }
  articles.push(article)
  const output = path.join(destination, relative)
  selectedFiles.add(output)
  await fs.mkdir(path.dirname(output), { recursive: true })
  await writeChanged(output, matter.stringify(body, {
    ...data,
    layout: 'article',
    description: article.excerpt,
    categoryName: article.categoryName,
    minutes: article.minutes,
  }))
  // Retain old URLs after moves and merges without duplicating list entries.
  for (const alias of data.aliases ?? []) {
    if (typeof alias !== 'string' || !/^\/(?:assets\/tutorial|topics\/[a-z-]+)\/[^/]+\.html$/.test(alias)) {
      throw new Error(`Invalid article alias: ${alias}`)
    }
    const aliasFile = path.join(destination, alias.slice(1).replace(/\.html$/, '.md'))
    selectedFiles.add(aliasFile)
    await fs.mkdir(path.dirname(aliasFile), { recursive: true })
    const target = (process.env.MRW_BASE || '/').replace(/\/?$/, '/') + article.url.slice(1)
    await writeChanged(aliasFile, matter.stringify(`[打开文章：${article.title}](${target})\n`, {
      title: article.title,
      layout: false,
      head: [
        ['meta', { 'http-equiv': 'refresh', content: `0;url=${target}` }],
        ['meta', { name: 'robots', content: 'noindex' }],
      ],
    }))
  }
}
articles.sort((a, b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN'))
for (const directory of ['topics', 'assets']) {
  const target = path.join(destination, directory)
  try {
    for (const file of await files(target)) if (!selectedFiles.has(file)) await fs.rm(file)
  } catch (error) { if (error.code !== 'ENOENT') throw error }
}
await fs.mkdir(path.join(destination, 'data'), { recursive: true })
await writeChanged(path.join(destination, 'data', 'articles.json'), JSON.stringify({ preview, articles }, null, 2))
console.log(`${preview ? 'Preview' : 'Public'} content prepared: ${articles.length} articles.`)
