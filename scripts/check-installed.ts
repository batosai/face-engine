import assert from 'node:assert/strict'
import type { ImageInput } from '../src/index.js'

const manifest = process.argv[2]
assert.ok(manifest, 'Expected the synthetic model manifest path')

const packageName = '@jrmc/face-engine'
const { FaceEngine } = await import(packageName) as typeof import('../src/index.js')
const engine = await FaceEngine.create({ manifest })

try {
  const dark: ImageInput = { format: 'rgb24', data: new Uint8Array(3), width: 1, height: 1 }
  const bright: ImageInput = { ...dark, data: new Uint8Array([255, 255, 255]) }
  assert.deepEqual(await engine.detect(dark), [])

  const detections = await engine.detect(bright)
  assert.equal(detections.length, 1)
  assert.equal(detections[0].landmarks.length, 5)
  assert.ok(detections[0].box.width > 0)

  const first = await engine.analyze(bright)
  const second = await engine.analyze(bright)
  assert.equal(first.length, 1)
  assert.equal(second.length, 1)
  const a = first[0].embedding
  const b = second[0].embedding
  assert.ok(a.values instanceof Float32Array)
  assert.equal(a.values.length, 512)
  assert.ok(Math.abs(Math.hypot(...a.values) - 1) < 1e-6)
  assert.ok(Math.abs(engine.compare(a, b) - 1) < 1e-12)
  console.log(`Installed package passed detection, embedding, and comparison on ${process.arch}.`)
} finally {
  await engine.close()
}
