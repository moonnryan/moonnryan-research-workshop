import path from 'node:path'
import fs from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const site = path.resolve(fileURLToPath(new URL('..', import.meta.url)))
const repository = path.dirname(site)
let pending = false
let preparing = false

async function prepare() {
  if (preparing) { pending = true; return }
  preparing = true
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, ['scripts/prepare-content.mjs', '--preview'], { cwd: site, stdio: 'inherit' })
      child.on('error', reject)
      child.on('exit', code => code === 0 ? resolve() : reject(new Error(`Content export exited with ${code}`)))
    })
  } finally {
    preparing = false
  }
  if (pending) { pending = false; await prepare() }
}
const directories = [
  path.join(repository, 'topics'), path.join(repository, 'assets/media'),
  path.join(site, 'app'), path.join(site, 'pages'),
]
async function fingerprint(directory) {
  const parts = []
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    if (entry.name === '.DS_Store') continue
    const file = path.join(directory, entry.name)
    if (entry.isDirectory()) parts.push(...await fingerprint(file))
    else { const stat = await fs.stat(file); parts.push(`${file}:${stat.size}:${stat.mtimeMs}`) }
  }
  return parts
}
async function signature() { return (await Promise.all(directories.map(fingerprint))).flat().sort().join('\n') }
await prepare()
let previous = await signature()
const extraArgs = process.argv.slice(2)
const hasHost = extraArgs.some(argument => argument === '--host' || argument.startsWith('--host='))
const server = spawn(process.execPath, ['node_modules/vitepress/bin/vitepress.js', 'dev', 'content', ...(hasHost ? [] : ['--host', '127.0.0.1']), ...extraArgs], { cwd: site, stdio: 'inherit' })
let checking = false
const timer = setInterval(async () => {
  if (checking) return
  checking = true
  try {
    const next = await signature()
    if (next !== previous) { previous = next; await prepare() }
  } catch (error) { console.error(error.message) }
  finally { checking = false }
}, 750)
function stop() { clearInterval(timer); server.kill('SIGTERM') }
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
server.on('exit', code => { clearInterval(timer); process.exit(code ?? 0) })
