import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import matter from 'gray-matter'

const repository = path.resolve(fileURLToPath(new URL('../..', import.meta.url)))
async function files(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map(entry => {
    const file = path.join(directory, entry.name)
    return entry.isDirectory() ? files(file) : entry.name.endsWith('.md') ? [file] : []
  }))
  return nested.flat()
}
function date(value) {
  return value instanceof Date ? value.toISOString().slice(0, 10) : String(value).slice(0, 10)
}
const documents = []
for (const directory of ['topics', 'assets/tutorial', 'assets/matrices']) {
  for (const file of await files(path.join(repository, directory))) {
    const { data } = matter(await fs.readFile(file, 'utf8'))
    const relative = path.relative(repository, file).split(path.sep).join('/')
    documents.push({
      file: relative, title: data.title, created: date(data.created),
      updated: date(data.updated ?? data.created), tags: data.tags ?? [],
      type: relative.startsWith('assets/tutorial/') ? '教程' : relative.startsWith('assets/matrices/') ? '矩阵' : '专题',
    })
  }
}
documents.sort((a, b) => a.created.localeCompare(b.created) || a.file.localeCompare(b.file, 'zh-CN'))
let chronology = '# 按工作时间阅读\n\n按首次工作日期排列，后续延伸保存在 updated 字段。教程用于复现，专题用于解释机制，矩阵用于核对契约。跨度较长的材料按问题归并，不按每一天拆文。\n\n原始日期不完整的研究稿使用整理日期并在文内标记，没有补造实验日期。是否公开由 blog 与 status 共同决定。\n'
let month = ''
for (const document of documents) {
  const next = document.created.slice(0, 7)
  if (next !== month) {
    month = next
    chronology += '\n## ' + month + '\n\n| 首次工作 | 更新范围截止 | 类型 | 内容 |\n| --- | --- | --- | --- |\n'
  }
  chronology += '| ' + document.created + ' | ' + document.updated + ' | ' + document.type + ' | [' + document.title + '](../' + document.file + ') |\n'
}
chronology += '\n## 整理范围\n\n归并工作记录中的部署、量化、图改写、训练、数据、视频、运行时与端侧状态管理。重复操作和日常配置归入对应篇章；客户、人员、内部仓库、网络与账号信息不进入文章。未验证设计写明假设，后续推翻的结论保留反证与最终边界。\n\nCV 约束沿用实际归档接口，新工具链和不同导出图需要另行验收。性能数据注明环境与计时条件，不作为跨平台性能承诺。\n'
await fs.mkdir(path.join(repository, 'indexes/hardware'), { recursive: true })
await fs.writeFile(path.join(repository, 'indexes/chronology.md'), chronology)
for (const [name, tag] of [['intel', 'Intel'], ['qualcomm', 'Qualcomm'], ['nvidia', 'NVIDIA'], ['ascend', 'Ascend'], ['rockchip', 'Rockchip'], ['sophgo', 'Sophgo']]) {
  const selected = documents.filter(document => document.tags.includes(tag))
  const index = '# ' + tag + ' 条目索引\n\n按首次工作日期排列。跨平台篇章在多个硬件索引中复用，不复制正文。\n\n| 日期 | 内容 |\n| --- | --- |\n' +
    selected.map(document => '| ' + document.created + ' | [' + document.title + '](../../' + document.file + ') |').join('\n') +
    '\n\n[返回时间索引](../chronology.md)\n'
  await fs.writeFile(path.join(repository, 'indexes/hardware', name + '.md'), index)
}
console.log('Updated chronological and hardware indexes for ' + documents.length + ' documents.')
const readmePath = path.join(repository, 'README.md')
const readme = await fs.readFile(readmePath, 'utf8')
const counts = ['专题', '教程', '矩阵'].map(type => documents.filter(document => document.type === type).length)
await fs.writeFile(readmePath, readme.replace(/共 \d+ 篇：\d+ 篇专题、\d+ 篇教程和 \d+ 份约束矩阵/, '共 ' + documents.length + ' 篇：' + counts[0] + ' 篇专题、' + counts[1] + ' 篇教程和 ' + counts[2] + ' 份约束矩阵'))
