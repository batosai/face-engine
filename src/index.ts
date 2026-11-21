import { resolve } from 'node:path'
import { types } from 'node:util'
import { Worker } from 'node:worker_threads'
import type { WorkerInput, WorkerReply, WorkerRequest } from './detection-worker.js'

export type ImageInput = string | Uint8Array | {
  format: 'rgb24'
  data: Uint8Array
  width: number
  height: number
}

export type Point = { x: number; y: number }

export type Detection = {
  box: { x: number; y: number; width: number; height: number }
  detectionScore: number
  landmarks: [Point, Point, Point, Point, Point]
}

type EngineOptions = {
  manifest: string
  threads?: number
  maxPending?: number
}

const MAX_ENCODED_BYTES = 32 * 1024 * 1024
const MAX_PIXELS = 32_000_000
const MAX_SIDE = 16_384
const createEngine = Symbol('createEngine')

function error(code: string, message: string) {
  return Object.assign(new Error(message), { code })
}

function prepareInput(image: ImageInput): { input: WorkerInput; transfer: ArrayBuffer[] } {
  if (typeof image === 'string') {
    if (!image || image.includes('\0')) throw error('INVALID_IMAGE', 'Expected a valid image path')
    return { input: { path: resolve(image) }, transfer: [] }
  }
  if (types.isUint8Array(image)) {
    if (image.byteLength > MAX_ENCODED_BYTES) throw error('LIMIT_EXCEEDED', 'Encoded image exceeds 32 MiB')
    const data = new Uint8Array(image)
    return { input: { data }, transfer: [data.buffer] }
  }
  if (image?.format !== 'rgb24') throw error('INVALID_IMAGE', 'Expected an image path, Uint8Array or rgb24 frame')
  const { data, width, height } = image
  if (!types.isUint8Array(data) || !Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw error('INVALID_IMAGE', 'Expected rgb24 bytes and positive integer dimensions')
  }
  if (width > MAX_SIDE || height > MAX_SIDE || width * height > MAX_PIXELS) {
    throw error('LIMIT_EXCEEDED', 'Image exceeds the dimension or pixel limit')
  }
  if (data.byteLength !== width * height * 3) throw error('INVALID_IMAGE', 'rgb24 length must equal width × height × 3')
  const copy = new Uint8Array(data)
  return { input: { data: copy, width, height }, transfer: [copy.buffer] }
}

export class FaceEngine {
  #worker
  #ready = Promise.withResolvers<void>()
  #exit = Promise.withResolvers<void>()
  #closing?: Promise<void>
  #failure?: Error
  #requests = new Map<number, PromiseWithResolvers<Detection[]>>()
  #capacity: number
  #nextRequest = 0

  private constructor(token: symbol, manifest: string, threads: number, maxPending: number) {
    if (token !== createEngine) throw new TypeError('Use FaceEngine.create()')
    this.#capacity = 1 + maxPending
    this.#worker = new Worker(new URL('./detection-worker.js', import.meta.url), {
      workerData: { manifest, threads },
    })
    this.#worker.on('message', (reply: WorkerReply) => {
      if (reply.id === undefined) {
        if ('failure' in reply) this.#ready.reject(error(reply.failure.code, reply.failure.message))
        else this.#ready.resolve()
        return
      }
      const request = this.#requests.get(reply.id)
      this.#requests.delete(reply.id)
      if ('failure' in reply) request?.reject(error(reply.failure.code, reply.failure.message))
      else request?.resolve(reply.result)
    })
    this.#worker.on('error', (failure) => this.#abort(failure))
    this.#worker.on('exit', (code) => {
      if (code !== 0 || !this.#closing) {
        this.#abort(error('ENGINE_CLOSED', `Detection worker exited with code ${code}`))
      }
      this.#exit.resolve()
    })
  }

  static create(options: EngineOptions): Promise<FaceEngine>
  static async create({ manifest, threads = 1, maxPending = 1 }: Partial<EngineOptions> = {}) {
    if (typeof manifest !== 'string' || !manifest || manifest.includes('\0')) {
      throw error('MODEL_INCOMPATIBLE', 'Expected a manifest path')
    }
    if (!Number.isInteger(threads) || threads < 1 || threads > 64 || !Number.isInteger(maxPending) || maxPending < 0 || maxPending > 32) {
      throw error('LIMIT_EXCEEDED', 'Expected 1–64 threads and 0–32 pending requests')
    }
    const engine = new FaceEngine(createEngine, resolve(manifest), threads, maxPending)
    try {
      await engine.#ready.promise
      return engine
    } catch (failure) {
      await engine.#exit.promise
      throw failure
    }
  }

  async detect(image: ImageInput): Promise<Detection[]> {
    if (this.#closing || this.#failure) throw error('ENGINE_CLOSED', 'The engine is closed')
    if (this.#requests.size >= this.#capacity) throw error('QUEUE_FULL', 'The image queue is full')
    const id = this.#nextRequest++
    const request = Promise.withResolvers<Detection[]>()
    this.#requests.set(id, request)
    try {
      const { input, transfer } = prepareInput(image)
      this.#worker.postMessage({ id, input } satisfies WorkerRequest, transfer)
    } catch (failure) {
      this.#requests.delete(id)
      throw failure
    }
    return request.promise
  }

  close(): Promise<void> {
    if (!this.#closing) {
      this.#closing = this.#exit.promise.then(() => {
        if (this.#failure) throw this.#failure
      })
      if (!this.#failure) this.#worker.postMessage({ close: true } satisfies WorkerRequest)
    }
    return this.#closing
  }

  #abort(failure: Error) {
    this.#failure ??= error('ENGINE_CLOSED', failure.message)
    this.#ready.reject(this.#failure)
    for (const request of this.#requests.values()) request.reject(this.#failure)
    this.#requests.clear()
  }
}
