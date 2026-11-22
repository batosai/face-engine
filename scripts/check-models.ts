import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { performance } from 'node:perf_hooks'

type Model = { path: string; sha256: string }
type Profile = { detector: Model; recognizer: Model }

async function checkHash(path: string, expected: string) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  const actual = hash.digest('hex')
  if (actual !== expected) throw new Error(`Unexpected SHA-256 for ${path}: ${actual}`)
}

async function main() {
  const manifest = resolve(process.argv[2] ?? 'profiles/yunet-sface.json')
  const profile = JSON.parse(await readFile(manifest, 'utf8')) as Profile
  if (!profile.recognizer) throw new Error('The manifest must include a recognizer')
  const detectorPath = resolve(dirname(manifest), profile.detector.path)
  const recognizerPath = resolve(dirname(manifest), profile.recognizer.path)
  await checkHash(detectorPath, profile.detector.sha256)
  await checkHash(recognizerPath, profile.recognizer.sha256)

  const baselineRssKiB = process.resourceUsage().maxRSS
  const start = performance.now()
  const { probeModels } = await import('#native')
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
