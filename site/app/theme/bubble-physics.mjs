// Spring motion and weighted circle constraints shared by the topic and tag fields.
export function createBodies(items, width, height, kind) {
  const maximum = Math.max(1, ...items.map(item => item.count))
  const topicScale = width <= 650 ? .62 : width <= 1000 ? .78 : 1
  const anchors = width <= 650
    ? [[.24, .16], [.77, .21], [.48, .47], [.23, .76], [.78, .80]]
    : [[.22, .30], [.49, .26], [.76, .33], [.34, .69], [.68, .73]]
  return items.map((item, index) => {
    const baseRadius = kind === 'topics'
      ? 120 * Math.sqrt(item.count / maximum) * topicScale
      : (width < 500 ? 24 : 26) * Math.sqrt(item.count)
    const angle = index * 2.3999632297 + .73
    const distance = Math.sqrt((index + .5) / items.length) * .94
    const [ax, ay] = anchors[index % anchors.length]
    const x = kind === 'topics' ? width * ax : width / 2 + Math.cos(angle) * (width / 2 - baseRadius - 8) * distance
    const y = kind === 'topics' ? height * ay : height / 2 + Math.sin(angle) * (height / 2 - baseRadius - 8) * distance
    return { id: item.id, x, y, homeX: x, homeY: y, vx: 0, vy: 0, radius: baseRadius, baseRadius, phase: index * 1.97, hovered: false }
  })
}

function contain(body, width, height, ellipse) {
  const pad = body.radius + 4
  if (ellipse) {
    const rx = Math.max(1, width / 2 - pad), ry = Math.max(1, height / 2 - pad)
    const dx = body.x - width / 2, dy = body.y - height / 2
    const distance = Math.hypot(dx / rx, dy / ry)
    if (distance > 1) {
      body.x = width / 2 + dx / distance
      body.y = height / 2 + dy / distance
      body.vx *= .65; body.vy *= .65
    }
  } else {
    const x = Math.min(Math.max(body.x, pad), width - pad)
    const y = Math.min(Math.max(body.y, pad), height - pad)
    if (x !== body.x) body.vx *= -.3
    if (y !== body.y) body.vy *= -.3
    body.x = x; body.y = y
  }
}

export function resolveCollisions(bodies, width, height, ellipse = false, strength = .6) {
  for (let i = 0; i < bodies.length; i++) for (let j = i + 1; j < bodies.length; j++) {
    const a = bodies[i], b = bodies[j]
    const dx = b.x - a.x, dy = b.y - a.y
    const distance = Math.hypot(dx, dy) || .001
    const overlap = a.radius + b.radius + 8 - distance
    if (overlap <= 0) continue
    const nx = distance === .001 ? 1 : dx / distance, ny = dy / distance
    // Hovered bubbles stay under the pointer while their neighbours make room.
    const ma = a.hovered ? 0 : 1 / (a.radius * a.radius)
    const mb = b.hovered ? 0 : 1 / (b.radius * b.radius)
    const mass = ma + mb
    if (!mass) continue
    const correction = overlap * strength
    a.x -= nx * correction * ma / mass; a.y -= ny * correction * ma / mass
    b.x += nx * correction * mb / mass; b.y += ny * correction * mb / mass
    const approaching = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
    if (approaching < 0) {
      const impulse = -1.3 * approaching / mass
      a.vx -= impulse * nx * ma; a.vy -= impulse * ny * ma
      b.vx += impulse * nx * mb; b.vy += impulse * ny * mb
    }
  }
  for (const body of bodies) contain(body, width, height, ellipse)
}

export function settleBodies(bodies, width, height, ellipse = false) {
  for (let i = 0; i < 160; i++) resolveCollisions(bodies, width, height, ellipse, .85)
  for (const body of bodies) { body.homeX = body.x; body.homeY = body.y; body.vx = 0; body.vy = 0 }
}

export function stepBodies(bodies, width, height, elapsed, delta, kind) {
  const dt = Math.min(delta, .04)
  for (const body of bodies) {
    const scale = body.hovered ? (kind === 'topics' ? 1.2 : 1.07) : 1
    body.radius += (body.baseRadius * scale - body.radius) * (1 - Math.exp(-9 * dt))
    if (body.hovered) { body.vx = 0; body.vy = 0; continue }
    const amplitude = kind === 'topics' ? Math.min(85, width * .09) : 13
    const targetX = body.homeX + Math.sin(elapsed * .4 + body.phase) * amplitude
    const targetY = body.homeY + Math.cos(elapsed * .47 + body.phase * 1.3) * amplitude * .68
    body.vx += (targetX - body.x) * .65 * dt
    body.vy += (targetY - body.y) * .65 * dt
    const damping = Math.exp(-1.35 * dt)
    body.vx *= damping; body.vy *= damping
    body.x += body.vx * dt; body.y += body.vy * dt
  }
  for (let i = 0; i < 4; i++) resolveCollisions(bodies, width, height, kind === 'tags', .45)
}
