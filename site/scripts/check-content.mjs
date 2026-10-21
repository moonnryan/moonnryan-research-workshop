import fs from 'node:fs/promises'
import path from 'node:path'
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'

const site = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const root = path.join(site, 'content')
const data = JSON.parse(await fs.readFile(path.join(root, 'data/articles.json'), 'utf8'))
if (!data.preview) assert.ok(data.articles.length > 0, 'Public build contains no articles; check blog and status metadata')
assert.equal(new Set(data.articles.map(article => article.url)).size, data.articles.length, 'Duplicate article routes')
for (const article of data.articles) {
  const file = path.join(root, article.url.replace(/^\//, '').replace(/\.html$/, '.md'))
  const text = await fs.readFile(file, 'utf8')
  assert.ok(text.includes('layout: article'), `Missing article layout: ${file}`)
  if (!data.preview) assert.equal(article.draft, false, 'Draft included in public content')
  for (const match of text.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    if (/^https?:/.test(match[1])) continue
    await fs.access(path.resolve(path.dirname(file), decodeURIComponent(match[1])))
  }
  assert.ok(!text.includes('```mermaid'), 'Unprocessed Mermaid code block')
}
console.log(`Validated ${data.articles.length} ${data.preview ? 'preview' : 'public'} articles and their local images.`)
