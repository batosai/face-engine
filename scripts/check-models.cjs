const { createHash } = require('node:crypto')
const { createReadStream } = require('node:fs')
const { resolve } = require('node:path')
const { performance } = require('node:perf_hooks')

const profile = {
  detector: {
    file: 'det_500m.onnx',
    sha256: '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a',
  },
  recognizer: {
    file: 'w600k_mbf.onnx',
    sha256: '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
  },
}

async function checkHash(path, expected) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  const actual = hash.digest('hex')
  if (actual !== expected) throw new Error(`SHA-256 inattendu pour ${path} : ${actual}`)
}

async function main() {
  const directory = resolve(process.argv[2] ?? 'models/buffalo_sc')
  const detectorPath = resolve(directory, profile.detector.file)
  const recognizerPath = resolve(directory, profile.recognizer.file)
  await checkHash(detectorPath, profile.detector.sha256)
  await checkHash(recognizerPath, profile.recognizer.sha256)

  const baselineRssKiB = process.resourceUsage().maxRSS
  const start = performance.now()
  const { probeModels } = require('../build/index.js')
  const models = probeModels(detectorPath, recognizerPath)

  console.log(JSON.stringify({
    node: process.version,
    arch: process.arch,
    loadMs: Math.round(performance.now() - start),
    baselineRssKiB,
    peakRssKiB: process.resourceUsage().maxRSS,
    ...models,
  }, null, 2))
}

main().catch((error) => {
  console.error(error.message)
  process.exitCode = 1
})
