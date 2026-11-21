import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { test } from 'node:test'
import { FaceEngine, type ImageInput } from '../src/index.js'

const manifest = resolve('profiles/buffalo-sc.json')
const blankPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAMAAAAoyzS7AAAAA1BMVEUAAACnej3aAAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==',
  'base64',
)
const blankRgb = { format: 'rgb24', data: new Uint8Array(3), width: 1, height: 1 } satisfies ImageInput

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
  const second = engine.analyze(blankPng)
  const rejected = engine.detect(blankRgb)
  const close = engine.close()
  assert.strictEqual(engine.close(), close)
  await assert.rejects(rejected, { code: 'QUEUE_FULL' })
  await assert.rejects(engine.detect(blankRgb), { code: 'ENGINE_CLOSED' })
  await assert.rejects(engine.analyze(blankRgb), { code: 'ENGINE_CLOSED' })
  assert.deepEqual(await Promise.all([first, second]), [[], []])
  await close
})

test('a detection-only profile keeps working and rejects analyze()', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'face-engine-test-'))
  t.after(() => rm(directory, { recursive: true }))
  const { version, detector } = JSON.parse(await readFile(manifest, 'utf8'))
  const path = join(directory, 'detector.json')
  await writeFile(path, JSON.stringify({
    version,
    detector: { ...detector, path: resolve(dirname(manifest), detector.path) },
  }))
  const engine = await FaceEngine.create({ manifest: path })
  t.after(() => engine.close())
  assert.deepEqual(await engine.detect(blankRgb), [])
  await assert.rejects(engine.analyze(blankRgb), { code: 'MODEL_INCOMPATIBLE' })
})
