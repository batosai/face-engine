import { execFile } from 'node:child_process'
import { copyFile, cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const architectures = ['x64', 'arm64'] as const
const onnxNotices = ['ONNX-Runtime-LICENSE', 'ONNX-Runtime-ThirdPartyNotices.txt']
const localPathFields = new Set(['source_path', 'src_path', 'manifest_path'])

type ProjectManifest = {
  name: string
  version: string
  license: string
  repository: { type: string; url: string }
  private?: boolean
  type: string
  main: string
  types: string
  imports: Record<string, string>
}

async function writeManifest(directory: string, manifest: object) {
  await writeFile(join(directory, 'package.json'), JSON.stringify(manifest, null, 2) + '\n')
}

async function pack(directory: string, destination: string) {
  const { stdout } = await execFileAsync('npm', ['pack', '--offline', '--ignore-scripts', '--json', '--pack-destination', destination], { cwd: directory })
  const [{ filename, size, unpackedSize, integrity }] = JSON.parse(stdout) as [{
    filename: string; size: number; unpackedSize: number; integrity: string
  }]
  return { path: join(destination, filename), size, unpackedSize, integrity }
}

async function main([licenseDir, outputDirectory = 'artifacts', nativeDirectory = 'build']: string[]) {
  if (!licenseDir) throw new Error('Usage: npm run pack:local -- <notices-dir> [output-dir] [native-dir]')
  const arch = process.arch
  if (process.platform !== 'linux' || (arch !== 'x64' && arch !== 'arm64')) {
    throw new Error('Expected a Linux x64 or ARM64 build host')
  }
  const { name, version, license, repository, private: privatePackage, type, main, types, imports } =
    JSON.parse(await readFile('package.json', 'utf8')) as ProjectManifest
  const platformName = `${name}-linux-${arch}-gnu`
  const binary = `face-engine.linux-${arch}-gnu.node`
  const optionalDependencies = Object.fromEntries(architectures.map((target) => [`${name}-linux-${target}-gnu`, version]))
  const destination = resolve(outputDirectory)
  const staging = await mkdtemp(join(tmpdir(), 'face-engine-pack-'))
  try {
    const core = join(staging, 'core')
    const platform = join(staging, 'platform')
    await mkdir(join(core, 'build'), { recursive: true })
    await mkdir(join(platform, 'licenses'), { recursive: true })
    await cp('dist/src', join(core, 'dist', 'src'), { recursive: true })
    await copyFile(join(nativeDirectory, 'index.js'), join(core, 'build', 'index.js'))
    await copyFile(join(nativeDirectory, binary), join(platform, binary))
    await copyFile('README.md', join(core, 'README.md'))
    await copyFile('LICENSE', join(core, 'LICENSE'))
    await copyFile('LICENSE', join(platform, 'LICENSE'))
    await cp('licenses/models', join(core, 'licenses', 'models'), { recursive: true })
    await copyFile('node_modules/@napi-rs/cli/LICENSE', join(core, 'THIRD-PARTY-NAPI-RS'))
    for (const notice of onnxNotices) {
      await copyFile(join(licenseDir, notice), join(platform, 'licenses', notice))
    }
    const rustLicenses = 'Rust-dependencies.json'
    const report: unknown = JSON.parse(await readFile(join(licenseDir, rustLicenses), 'utf8'))
    const licenseText = JSON.stringify(report, (key, value: unknown) => localPathFields.has(key) ? undefined : value, 2)
    await writeFile(join(platform, 'licenses', rustLicenses), licenseText + '\n')

    await writeManifest(core, {
      name, version, license, repository, private: privatePackage, type, main, types, imports,
      optionalDependencies,
    })
    await writeManifest(platform, {
      name: platformName, version, license, repository, private: privatePackage, main: binary,
      os: ['linux'], cpu: [arch], libc: ['glibc'],
    })

    await mkdir(destination, { recursive: true })
    const coreArchive = await pack(core, destination)
    const platformArchive = await pack(platform, destination)
    console.log(JSON.stringify([coreArchive, platformArchive], null, 2))
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}

main(process.argv.slice(2)).catch((failure: unknown) => {
  console.error(failure instanceof Error ? failure.message : String(failure))
  process.exitCode = 1
})
