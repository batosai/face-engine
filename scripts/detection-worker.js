import { parentPort, workerData } from 'node:worker_threads'

function sendFailure(failure, fallback, id) {
  const native = /^([A-Z_]+): (.*)$/s.exec(failure.message)
  const [, code, message] = native ?? [null, fallback, failure.message]
  parentPort.postMessage({ id, failure: { code, message } })
}

async function start() {
  let detector
  try {
    const { NativeDetector } = await import('../build/index.js')
    detector = NativeDetector.open(workerData.manifest, workerData.threads)
  } catch (failure) {
    sendFailure(failure, 'MODEL_INCOMPATIBLE')
    parentPort.close()
    return
  }

  parentPort.on('message', ({ id, input, close }) => {
    if (close) {
      detector.close()
      parentPort.close()
      return
    }
    try {
      let result
      if (input.path) result = detector.detectPath(input.path)
      else if (input.width !== undefined) result = detector.detectRgb(input.data, input.width, input.height)
      else result = detector.detectEncoded(input.data)
      parentPort.postMessage({ id, result })
    } catch (failure) {
      sendFailure(failure, 'INVALID_IMAGE', id)
    }
  })
  parentPort.postMessage({})
}

await start()
