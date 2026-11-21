import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import { test } from 'node:test'
import { FaceEngine, type Embedding } from '../src/index.js'

function embedding(x: number, y = 0, spaceId = 'test-space'): Embedding {
  const values = new Float32Array(512)
  values.set([x, y])
  return { spaceId, values }
}

test('compare returns the cosine of finite vectors without requiring unit norms', async (t) => {
  const engine = await FaceEngine.create({ manifest: resolve('profiles/buffalo-sc.json') })
  t.after(() => engine.close())

  const x = embedding(4)
  assert.equal(engine.compare(x, embedding(7)), 1)
  assert.equal(engine.compare(x, embedding(-2)), -1)
  assert.equal(engine.compare(x, embedding(0, 3)), 0)
  assert.ok(Math.abs(engine.compare(x, embedding(1, 1)) - Math.SQRT1_2) < 1e-12)
  assert.equal(engine.compare(embedding(3e38), embedding(1e-45)), 1)

  const shared = new Float32Array(new SharedArrayBuffer(512 * Float32Array.BYTES_PER_ELEMENT))
  shared[0] = 1
  assert.equal(engine.compare(x, { spaceId: x.spaceId, values: shared }), 1)
})

test('compare rejects incompatible spaces, dimensions, values and norms', async (t) => {
  const engine = await FaceEngine.create({ manifest: resolve('profiles/buffalo-sc.json') })
  t.after(() => engine.close())

  const valid = embedding(1)
  const invalid = [
    embedding(1, 0, 'other-space'),
    embedding(1, 0, ''),
    { ...valid, values: new Float32Array(511) },
    { ...valid, values: [1, 0] } as unknown as Embedding,
    embedding(0),
    embedding(Number.NaN),
    embedding(Number.POSITIVE_INFINITY),
  ]
  for (const candidate of invalid) {
    assert.throws(() => engine.compare(valid, candidate), { code: 'INCOMPATIBLE_EMBEDDING' })
  }
  const emptySpace = embedding(1, 0, '')
  assert.throws(() => engine.compare(emptySpace, emptySpace), { code: 'INCOMPATIBLE_EMBEDDING' })
})
