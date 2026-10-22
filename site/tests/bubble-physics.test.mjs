import assert from 'node:assert/strict'
import test from 'node:test'
import { createBodies, settleBodies, stepBodies } from '../app/theme/bubble-physics.mjs'

function checkSpacing(bodies, width, height) {
  for (let i = 0; i < bodies.length; i++) {
    const a = bodies[i]
    assert.ok(Number.isFinite(a.x) && Number.isFinite(a.y))
    assert.ok(a.x >= a.radius && a.x <= width - a.radius)
    assert.ok(a.y >= a.radius && a.y <= height - a.radius)
    for (let j = i + 1; j < bodies.length; j++) {
      const b = bodies[j]
      assert.ok(Math.hypot(a.x - b.x, a.y - b.y) >= a.radius + b.radius - 1, `${a.id} overlaps ${b.id}`)
    }
  }
}

for (const width of [320, 690]) {
  test(`48 weighted tags remain separated inside a ${width}px field`, () => {
    const items = Array.from({ length: 48 }, (_, i) => ({ id: String(i), count: i === 0 ? 5 : i < 3 ? 3 : i < 10 ? 2 : 1 }))
    const unit = width < 500 ? 24 : 26
    const area = items.reduce((sum, item) => sum + Math.PI * (unit * Math.sqrt(item.count) + 4) ** 2, 0)
    const height = Math.max(450, Math.ceil(area / (Math.PI / 4 * width * .57)))
    const bodies = createBodies(items, width, height, 'tags')
    settleBodies(bodies, width, height, true)
    checkSpacing(bodies, width, height)
    assert.ok(bodies[0].radius > bodies[3].radius && bodies[3].radius > bodies[15].radius)
    for (let i = 0; i < 600; i++) stepBodies(bodies, width, height, i / 60, 1 / 60, 'tags')
    checkSpacing(bodies, width, height)
  })
}

test('hover grows progressively, holds the pointer target and pushes its neighbour away', () => {
  const a = { id: 'a', x: 150, y: 150, homeX: 150, homeY: 150, radius: 50, baseRadius: 50, vx: 0, vy: 0, phase: 0, hovered: true }
  const b = { ...a, id: 'b', x: 245, homeX: 245, hovered: false }
  stepBodies([a, b], 500, 300, 0, 1 / 60, 'topics')
  assert.ok(a.radius > 50 && a.radius < 60)
  for (let i = 1; i < 60; i++) stepBodies([a, b], 500, 300, i / 60, 1 / 60, 'topics')
  assert.ok(a.radius > 59.9)
  assert.equal(a.x, 150)
  assert.ok(b.x > 250)
  checkSpacing([a, b], 500, 300)
  a.hovered = false
  for (let i = 0; i < 90; i++) stepBodies([a, b], 500, 300, 1 + i / 60, 1 / 60, 'topics')
  assert.ok(a.radius < 50.01)
})
