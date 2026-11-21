import { randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { join, resolve } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'
import { FaceEngine } from '@jrmc/face-engine'
import { LibraryDatabase } from './database.js'
import type { Photo } from './types.js'

const port = Number(process.env.PORT ?? 4173)
if (!Number.isInteger(port) || port < 0 || port > 65_535) throw new Error('PORT must be between 0 and 65535')

const dataDir = resolve(process.env.FACE_ENGINE_DATA_DIR ?? fileURLToPath(new URL('../data/', import.meta.url)))
const originals = join(dataDir, 'photos')
const manifest = process.env.FACE_ENGINE_MANIFEST ?? fileURLToPath(new URL('../../profiles/buffalo-sc.json', import.meta.url))
const engine = await FaceEngine.create({ manifest })
await mkdir(originals, { recursive: true, mode: 0o700 })
const database = new LibraryDatabase(join(dataDir, 'library.sqlite'))
const scheduled = new Set<string>()
let work = Promise.resolve()

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}

const pngSignature = Buffer.from('89504e470d0a1a0a', 'hex')
const jpegSignature = Buffer.from('ffd8ff', 'hex')
const photoRoute = /^\/api\/photos\/([0-9a-f-]{36})(?:\/(image|analyze))?$/
const faceRoute = /^\/api\/faces\/([1-9][0-9]*)$/

const assets = new Map([
  ['/', { type: 'text/html; charset=utf-8', file: new URL('../public/index.html', import.meta.url) }],
  ['/styles.css', { type: 'text/css; charset=utf-8', file: new URL('../public/styles.css', import.meta.url) }],
  ['/app.js', { type: 'text/javascript; charset=utf-8', file: new URL('./client.js', import.meta.url) }],
])

function enqueue(id: string) {
  if (scheduled.has(id)) return
  database.setStatus(id, 'queued')
  scheduled.add(id)
  work = work.then(async () => {
    if (!database.photo(id)) return
    database.setStatus(id, 'analyzing')
    const faces = await engine.analyze(join(originals, id))
    database.saveAnalysis(id, faces)
  }).catch((error: unknown) => {
    database.setStatus(id, 'failed', error instanceof Error ? error.message : 'Analysis failed')
  }).finally(() => { scheduled.delete(id) })
}

async function recoverUploads() {
  for (const file of await readdir(originals)) {
    if (!/^[0-9a-f-]{36}\.deleted$/.test(file)) continue
    const id = file.slice(0, -'.deleted'.length)
    const tombstone = join(originals, file)
    if (database.photo(id)) await rename(tombstone, join(originals, id))
    else await rm(tombstone)
  }
  for (const id of database.pending()) enqueue(id)
}

async function readBody(request: IncomingMessage, limit: number) {
  if (Number(request.headers['content-length']) > limit) {
    request.resume()
    throw new HttpError(413, 'The request exceeds the size limit.')
  }
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of request.iterator({ destroyOnReturn: false }) as AsyncIterable<Buffer>) {
    size += chunk.length
    if (size > limit) {
      request.resume()
      throw new HttpError(413, 'The request exceeds the size limit.')
    }
    chunks.push(chunk)
  }
  return Buffer.concat(chunks, size)
}

async function importPhoto(request: IncomingMessage) {
  const filename = request.headers['x-filename']
  if (typeof filename !== 'string') throw new HttpError(400, 'A filename is required.')
  let name: string
  try {
    name = decodeURIComponent(filename).split(/[\\/]/).pop()!.trim()
  } catch {
    throw new HttpError(400, 'The filename is invalid.')
  }
  if (!name) throw new HttpError(400, 'A filename is required.')

  const bytes = await readBody(request, 32 * 1024 * 1024)
  let mime: Photo['mime']
  if (bytes.subarray(0, pngSignature.length).equals(pngSignature)) mime = 'image/png'
  else if (bytes.subarray(0, jpegSignature.length).equals(jpegSignature)) mime = 'image/jpeg'
  else throw new HttpError(415, 'Only JPEG and PNG images are supported.')

  const id = randomUUID()
  const file = join(originals, id)
  await writeFile(file, bytes, { flag: 'wx', mode: 0o600 })
  try {
    database.addPhoto(id, name, mime)
  } catch (error) {
    await rm(file, { force: true })
    throw error
  }
  enqueue(id)
  return { id }
}

async function deletePhoto(id: string) {
  const file = join(originals, id)
  const tombstone = `${file}.deleted`
  let renamed = false
  try {
    await rename(file, tombstone)
    renamed = true
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
  }
  try {
    database.removePhoto(id)
  } catch (error) {
    if (renamed) await rename(tombstone, file)
    throw error
  }
  if (renamed) await rm(tombstone, { force: true })
}

async function assignFace(request: IncomingMessage, id: number) {
  let input: unknown
  try {
    input = JSON.parse((await readBody(request, 4096)).toString())
  } catch (error) {
    if (error instanceof HttpError) throw error
    throw new HttpError(400, 'Expected a valid assignment.')
  }
  if (typeof input !== 'object' || !input) throw new HttpError(400, 'Expected a valid assignment.')
  let assigned: boolean
  if ('name' in input && typeof input.name === 'string' && input.name.trim()) {
    assigned = database.assign(id, { name: input.name.trim() })
  } else if ('personId' in input && (input.personId === null || (typeof input.personId === 'number' && Number.isSafeInteger(input.personId) && input.personId > 0))) {
    assigned = database.assign(id, { personId: input.personId })
  } else {
    throw new HttpError(400, 'Choose a person, enter a name, or clear the assignment.')
  }
  if (!assigned) throw new HttpError(404, 'The face or person no longer exists.')
}

function json(response: ServerResponse, status: number, value: unknown) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' })
  response.end(JSON.stringify(value))
}

async function handle(request: IncomingMessage, response: ServerResponse) {
  const { pathname } = new URL(request.url ?? '/', 'http://localhost')
  const method = request.method
  response.setHeader('X-Content-Type-Options', 'nosniff')
  if (method === 'GET') {
    const asset = assets.get(pathname)
    if (asset) {
      const bytes = await readFile(asset.file)
      response.writeHead(200, { 'Content-Type': asset.type })
      response.end(bytes)
      return
    }
    if (pathname === '/api/library') return json(response, 200, database.library(engine))
  }

  if (method === 'POST' && pathname === '/api/photos') return json(response, 202, await importPhoto(request))

  const photoMatch = photoRoute.exec(pathname)
  if (photoMatch) {
    const [, id, action] = photoMatch
    const photo = database.photo(id)
    if (!photo) throw new HttpError(404, 'The photo no longer exists.')
    if (method === 'GET' && action === 'image') {
      const file = join(originals, id)
      const info = await stat(file)
      response.writeHead(200, { 'Content-Type': photo.mime, 'Content-Length': info.size, 'Cache-Control': 'private, max-age=3600' })
      await pipeline(createReadStream(file), response)
      return
    }
    if (method === 'POST' && action === 'analyze') {
      enqueue(id)
      return json(response, 202, { id })
    }
    if (method === 'DELETE' && action === undefined) {
      await deletePhoto(id)
      response.writeHead(204).end()
      return
    }
  }
  const faceMatch = faceRoute.exec(pathname)
  if (method === 'PATCH' && faceMatch) {
    await assignFace(request, Number(faceMatch[1]))
    response.writeHead(204).end()
    return
  }
  throw new HttpError(404, 'Not found.')
}

await recoverUploads()
const server = createServer((request, response) => {
  void handle(request, response).catch((error: unknown) => {
    if (response.headersSent || response.destroyed) return response.destroy()
    if (error instanceof HttpError) return json(response, error.status, { error: error.message })
    console.error(error)
    return json(response, 500, { error: 'The request could not be completed.' })
  })
})

server.listen(port, '127.0.0.1', () => {
  const address = server.address()
  if (typeof address === 'object' && address) console.log(`Photo library on http://127.0.0.1:${address.port}`)
})

function shutdown() {
  server.close(() => {
    void work.finally(async () => {
      await engine.close()
      database.close()
    }).catch((error: unknown) => { console.error(error); process.exitCode = 1 })
  })
}

process.once('SIGINT', shutdown)
process.once('SIGTERM', shutdown)
