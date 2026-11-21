import { fileURLToPath } from 'node:url'
import { FaceEngine } from '../src/index.js'

async function embeddingFrom(engine: FaceEngine, path: string) {
  const faces = await engine.analyze(path)
  if (faces.length !== 1) {
    throw new Error(`Expected exactly one analyzable face in ${path} (found ${faces.length}).`)
  }
  return faces[0].embedding
}

async function main(args: string[]) {
  if (args.length < 2 || args.length > 3 || args.some((arg) => !arg)) {
    throw new Error('Usage: npm run compare -- <photo-a> <photo-b> [manifest]')
  }
  const [first, second, manifest = fileURLToPath(new URL('../../profiles/buffalo-sc.json', import.meta.url))] = args
  const engine = await FaceEngine.create({ manifest })
  try {
    const a = await embeddingFrom(engine, first)
    const b = await embeddingFrom(engine, second)
    console.log(engine.compare(a, b))
  } finally {
    await engine.close()
  }
}

main(process.argv.slice(2)).catch((failure: unknown) => {
  console.error(failure instanceof Error ? failure.message : String(failure))
  process.exitCode = 1
})
