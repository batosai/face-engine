import { parentPort, workerData } from 'node:worker_threads'
import type { NativeDetector } from '#native'
import type { Detection } from './index.js'

export type WorkerInput =
  | { path: string }
  | { data: Uint8Array }
  | { data: Uint8Array; width: number; height: number }

export type WorkerRequest = { close: true } | { id: number; input: WorkerInput }

export type WorkerReply =
  | { id?: undefined }
  | { id?: number; failure: { code: string; message: string } }
  | { id: number; result: Detection[] }

if (!parentPort) throw new Error('The detector must run inside a worker')
const port = parentPort
const { manifest, threads } = workerData as { manifest: string; threads: number }

function sendFailure(failure: unknown, fallback: string, id?: number) {
  const detail = failure instanceof Error ? failure.message : String(failure)
  const native = /^([A-Z_]+): (.*)$/s.exec(detail)
  const [, code, message] = native ?? [null, fallback, detail]
  port.postMessage({ id, failure: { code, message } } satisfies WorkerReply)
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
    const { id, input } = request
    try {
      let result: Detection[]
      if ('path' in input) result = detector.detectPath(input.path)
      else if ('width' in input) result = detector.detectRgb(input.data, input.width, input.height)
      else result = detector.detectEncoded(input.data)
      port.postMessage({ id, result } satisfies WorkerReply)
    } catch (failure) {
      sendFailure(failure, 'INVALID_IMAGE', id)
    }
  })
  port.postMessage({} satisfies WorkerReply)
}

await start()
