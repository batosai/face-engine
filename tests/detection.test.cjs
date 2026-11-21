const assert = require('node:assert/strict')
const { resolve } = require('node:path')
const { test } = require('node:test')
const { FaceEngine } = require('..')

const manifest = resolve(__dirname, '../profiles/buffalo-sc.json')
const blankPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAMAAAAoyzS7AAAAA1BMVEUAAACnej3aAAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==',
  'base64',
)
const blankRgb = { format: 'rgb24', data: new Uint8Array(3), width: 1, height: 1 }

test('a rejected manifest leaves the process able to open engines', async (t) => {
  await assert.rejects(FaceEngine.create({ manifest: `${manifest}.missing` }), { code: 'MODEL_INCOMPATIBLE' })
  const [first, second] = await Promise.all([FaceEngine.create({ manifest }), FaceEngine.create({ manifest })])
  t.after(() => Promise.all([first.close(), second.close()]))
  assert.deepEqual(await Promise.all([first.detect(blankPng), second.detect(blankRgb)]), [[], []])
})

test('detection copies input bytes and reports invalid inputs', async (t) => {
  const engine = await FaceEngine.create({ manifest })
  t.after(() => engine.close())
  const source = Buffer.from(blankPng)
  const detection = engine.detect(source)
  source.fill(0)
  assert.deepEqual(await detection, [])
  await assert.rejects(engine.detect(source), { code: 'INVALID_IMAGE' })
  await assert.rejects(engine.detect({ ...blankRgb, width: 2 }), { code: 'INVALID_IMAGE' })
  await assert.rejects(engine.detect({ ...blankRgb, width: 17_000 }), { code: 'LIMIT_EXCEEDED' })
})

test('close drains accepted work, rejects new work and resolves once', async (t) => {
  const engine = await FaceEngine.create({ manifest, maxPending: 1 })
  t.after(() => engine.close())
  const first = engine.detect(blankRgb)
  const second = engine.detect(blankPng)
  const rejected = engine.detect(blankRgb)
  const close = engine.close()
  assert.strictEqual(engine.close(), close)
  await assert.rejects(rejected, { code: 'QUEUE_FULL' })
  await assert.rejects(engine.detect(blankRgb), { code: 'ENGINE_CLOSED' })
  assert.deepEqual(await Promise.all([first, second]), [[], []])
  await close
})
