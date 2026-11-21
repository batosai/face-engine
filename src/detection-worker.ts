import { parentPort, workerData } from 'node:worker_threads'
import type { NativeDetector } from '#native'
import type { Detection, Face } from './index.js'

export type WorkerInput =
  | { path: string }
  | { data: Uint8Array }
  | { data: Uint8Array; width: number; height: number }

export type Operation = 'detect' | 'analyze'
export type WorkerRequest = { close: true } | { id: number; input: WorkerInput; operation: Operation }

export type WorkerResult =
  | { operation: 'detect'; faces: Detection[] }
  | { operation: 'analyze'; faces: Face[] }

export type WorkerReply =
  | { id?: undefined }
  | { id?: number; failure: { code: string; message: string } }
  | { id: number; result: WorkerResult }

if (!parentPort) throw new Error('The detector must run inside a worker')
const port = parentPort
const { manifest, threads } = workerData as { manifest: string; threads: number }

function sendFailure(failure: unknown, fallback: string, id?: number) {
  const detail = failure instanceof Error ? failure.message : String(failure)
  const native = /^([A-Z_]+): (.*)$/s.exec(detail)
  const [, code, message] = native ?? [null, fallback, detail]
  port.postMessage({ id, failure: { code, message } } satisfies WorkerReply)
}

function processImage(detector: NativeDetector, operation: Operation, input: WorkerInput): WorkerResult {
  if (operation === 'detect') {
    if ('path' in input) return { operation, faces: detector.detectPath(input.path) }
    if ('width' in input) return { operation, faces: detector.detectRgb(input.data, input.width, input.height) }
    return { operation, faces: detector.detectEncoded(input.data) }
  }
  if ('path' in input) return { operation, faces: detector.analyzePath(input.path) }
  if ('width' in input) return { operation, faces: detector.analyzeRgb(input.data, input.width, input.height) }
  return { operation, faces: detector.analyzeEncoded(input.data) }
}

async function start() {
  let detector: NativeDetector
  try {
    const native = await import('#native')
    detector = native.NativeDetector.open(manifest, threads)
  } catch (failure) {
    sendFailure(failure, 'MODEL_INCOMPATIBLE')
    port.close()
    return
  }

  port.on('message', (request: WorkerRequest) => {
    if ('close' in request) {
      detector.close()
      port.close()
      return
    }
    const { id, input, operation } = request
    try {
      const result = processImage(detector, operation, input)
      port.postMessage({ id, result } satisfies WorkerReply)
    } catch (failure) {
      sendFailure(failure, 'INVALID_IMAGE', id)
    }
  })
  port.postMessage({} satisfies WorkerReply)
}

await start()
