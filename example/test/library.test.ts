import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { test } from 'node:test'
import { setTimeout as delay } from 'node:timers/promises'
import type { Library } from '../src/types.ts'

const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAMAAAAoyzS7AAAAA1BMVEX///+nxBvIAAAACklEQVQI12NgAAAAAgAB4iG8MwAAAABJRU5ErkJggg==',
  'base64',
)

function launch(data: string) {
  const child = spawn(process.execPath, ['dist/server.js'], {
    env: { ...process.env, PORT: '0', FACE_ENGINE_DATA_DIR: data, FACE_ENGINE_MANIFEST: resolve('../tests/fixtures/manifest.json') },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  const exited = once(child, 'exit')
  let stderr = ''
  child.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString() })
  const ready = new Promise<string>((resolve, reject) => {
    child.on('error', reject)
    child.on('exit', () => reject(new Error(stderr || 'The example server exited before startup.')))
    child.stdout.on('data', (chunk: Buffer) => {
      const match = /http:\/\/127\.0\.0\.1:\d+/.exec(chunk.toString())
      if (match) resolve(match[0])
    })
  })
  return {
    ready,
    async stop() {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM')
      await exited
    },
  }
}

async function fetchLibrary(base: string): Promise<Library> {
  const response = await fetch(`${base}/api/library`)
  assert.equal(response.status, 200)
  return response.json() as Promise<Library>
}

async function analyzed(base: string, id: string) {
  for (let attempt = 0; attempt < 100; attempt++) {
    const library = await fetchLibrary(base)
    const photo = library.photos.find((photo) => photo.id === id)
    if (photo?.status === 'ready') return library
    assert.notEqual(photo?.status, 'failed', photo?.error ?? undefined)
    await delay(40)
  }
  throw new Error('The queued analysis did not complete.')
}

async function upload(base: string, name: string) {
  const response = await fetch(`${base}/api/photos`, { method: 'POST', headers: { 'X-Filename': name }, body: png })
  assert.equal(response.status, 202)
  return (await response.json() as { id: string }).id
}

test('uploads, suggests, preserves assignments through replay and restart, and deletes the original', { timeout: 15_000 }, async (t) => {
  const data = await mkdtemp(join(tmpdir(), 'face-engine-example-'))
  let server = launch(data)
  t.after(async () => {
    await server.stop()
    await rm(data, { recursive: true, force: true })
  })
  let base = await server.ready
  const first = await upload(base, 'first.png')
  const firstLibrary = await analyzed(base, first)
  const [face] = firstLibrary.faces
  assert.ok(face)
  assert.equal(face.personId, null)
  assert.equal('embedding' in face, false)
  assert.deepEqual(await readFile(join(data, 'photos', first)), png)

  const assignment = await fetch(`${base}/api/faces/${face.id}`, { method: 'PATCH', body: JSON.stringify({ name: 'Alex' }) })
  assert.equal(assignment.status, 204)
  const second = await upload(base, 'second.png')
  const current = await analyzed(base, second)
  const unassigned = current.faces.find((face) => face.photoId === second)!
  assert.equal(unassigned.personId, null)
  assert.equal(unassigned.suggestions.length, 1)
  assert.equal(unassigned.suggestions[0].personId, current.people[0].id)
  assert.ok(Math.abs(unassigned.suggestions[0].score - 1) < 1e-12)

  const replay = await fetch(`${base}/api/photos/${first}/analyze`, { method: 'POST' })
  assert.equal(replay.status, 202)
  assert.equal((await analyzed(base, first)).faces.filter((face) => face.photoId === first).length, 1)

  await server.stop()
  server = launch(data)
  base = await server.ready
  const saved = await fetchLibrary(base)
  assert.equal(saved.faces.find((stored) => stored.id === face.id)?.personId, saved.people[0].id)

  const deleted = await fetch(`${base}/api/photos/${first}`, { method: 'DELETE' })
  assert.equal(deleted.status, 204)
  const afterDelete = await fetchLibrary(base)
  assert.equal(afterDelete.photos.some((photo) => photo.id === first), false)
  assert.equal(afterDelete.faces.some((face) => face.photoId === first), false)
  await assert.rejects(readFile(join(data, 'photos', first)), { code: 'ENOENT' })
})
