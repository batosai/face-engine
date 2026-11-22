import { execFile } from 'node:child_process'
import { readdir, readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { monitorEventLoopDelay, performance } from 'node:perf_hooks'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { FaceEngine } from '../src/index.js'

const execFileAsync = promisify(execFile)
const manifest = fileURLToPath(new URL('../../profiles/yunet-sface.json', import.meta.url))

function percentile(samples: number[], fraction: number) {
  const sorted = samples.toSorted((a, b) => a - b)
  return sorted[Math.ceil(sorted.length * fraction) - 1]
}

function measureCopy(input: Uint8Array) {
  const samples = []
  for (let i = 0; i < 30; i++) {
    const start = performance.now()
    const snapshot = new Uint8Array(input)
    samples.push(performance.now() - start)
    if (snapshot.byteLength !== input.byteLength) throw new Error('Incomplete copy')
  }
  return { bytes: input.byteLength, p95Ms: percentile(samples, 0.95) }
}

async function bench(photos: string[], iterations: number, threads: number) {
  const started = performance.now()
  const engine = await FaceEngine.create({ manifest, threads })
  const afterCreate = performance.now()
  const latency = []
  const rss = []
  const lag = monitorEventLoopDelay({ resolution: 1 })
  let faces = 0
  try {
    await engine.analyze(photos[0])
    const afterFirst = performance.now()
    for (const photo of photos.slice(0, 5)) await engine.analyze(photo)

    lag.enable()
    const cpu = process.cpuUsage()
    const start = performance.now()
    for (let i = 0; i < iterations; i++) {
      const before = performance.now()
      faces += (await engine.analyze(photos[i % photos.length])).length
      latency.push(performance.now() - before)
      rss.push(process.memoryUsage.rss() / 1024 ** 2)
    }
    const elapsed = performance.now() - start
    const usage = process.cpuUsage(cpu)
    lag.disable()
    const window = Math.floor(iterations / 4)
    const firstWindow = rss.slice(0, window)
    const lastWindow = rss.slice(-window)
    const peakRssMiB = process.resourceUsage().maxRSS / 1024

    return {
      threads,
      iterations,
      faces,
      createMs: afterCreate - started,
      firstImageMs: afterFirst - afterCreate,
      p50Ms: percentile(latency, 0.50),
      p95Ms: percentile(latency, 0.95),
      imagesPerSecond: iterations * 1000 / elapsed,
      cpuPercent: (usage.user + usage.system) / (elapsed * 10),
      peakRssMiB,
      firstRssP50MiB: percentile(firstWindow, 0.50),
      lastRssP50MiB: percentile(lastWindow, 0.50),
      lastRssMinMiB: Math.min(...lastWindow),
      lastRssMaxMiB: Math.max(...lastWindow),
      eventLoopDelayP95Ms: lag.percentile(95) / 1e6,
      encodedCopy: measureCopy(await readFile(photos[0])),
    }
  } finally {
    lag.disable()
    await engine.close()
  }
}

async function main([directory = 'photos', count = '400', workerThreads]: string[]) {
  const iterations = Number(count)
  if (!Number.isSafeInteger(iterations) || iterations < 4 || iterations > 10_000) {
    throw new Error('Expected 4–10000 iterations')
  }
  directory = resolve(directory)
  const photos = (await readdir(directory, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && /\.(jpe?g|png)$/i.test(entry.name))
    .map((entry) => join(directory, entry.name))
    .sort()
  if (!photos.length) throw new Error('Expected a directory containing JPEG or PNG photos')
  if (workerThreads !== undefined) {
    console.log(JSON.stringify(await bench(photos, iterations, Number(workerThreads))))
    return
  }
  console.log(JSON.stringify({ node: process.version, platform: process.platform, arch: process.arch, photos: photos.length }))
  for (const threads of [1, 2, 4]) {
    const { stdout } = await execFileAsync(process.execPath, [fileURLToPath(import.meta.url), directory, count, String(threads)])
    process.stdout.write(stdout)
  }
}

main(process.argv.slice(2)).catch((failure: unknown) => {
  console.error(failure instanceof Error ? failure.message : String(failure))
  process.exitCode = 1
})
