import { DatabaseSync } from 'node:sqlite'
import type { Embedding, Face as DetectedFace, FaceEngine } from '@jrmc/face-engine'
import type { Box, Face, Library, Person, Photo } from './types.js'

type StoredFace = Omit<Face, 'box' | 'suggestions'> & Box & { spaceId: string; embedding: Uint8Array }

const selectPhotos = `SELECT p.id, p.name, p.mime, p.status, p.error,
  COUNT(f.id) AS faceCount,
  COUNT(f.id) FILTER (WHERE f.person_id IS NULL) AS unassignedCount
  FROM photos p LEFT JOIN faces f ON f.photo_id = p.id`

const selectFaces = `SELECT id, photo_id AS photoId, x, y, width, height, score,
  space_id AS spaceId, embedding, person_id AS personId FROM faces`

function encode(values: Float32Array) {
  const bytes = Buffer.alloc(values.length * 4)
  values.forEach((value, index) => bytes.writeFloatLE(value, index * 4))
  return bytes
}

function decode(face: Pick<StoredFace, 'spaceId' | 'embedding'>): Embedding {
  const bytes = Buffer.from(face.embedding.buffer, face.embedding.byteOffset, face.embedding.byteLength)
  const values = Float32Array.from({ length: bytes.length / 4 }, (_, index) => bytes.readFloatLE(index * 4))
  return { spaceId: face.spaceId, values }
}

function overlap(a: Box, b: Box) {
  const width = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
  const height = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y))
  const area = width * height
  return area / (a.width * a.height + b.width * b.height - area)
}

export class LibraryDatabase {
  #db: DatabaseSync

  constructor(path: string) {
    this.#db = new DatabaseSync(path)
    this.#db.exec(`PRAGMA foreign_keys = ON;
      CREATE TABLE IF NOT EXISTS photos (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, mime TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('queued', 'analyzing', 'ready', 'failed')),
        error TEXT
      );
      CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE
      );
      CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        x REAL NOT NULL, y REAL NOT NULL, width REAL NOT NULL, height REAL NOT NULL,
        score REAL NOT NULL, space_id TEXT NOT NULL, embedding BLOB NOT NULL,
        person_id INTEGER REFERENCES people(id) ON DELETE SET NULL
      );
      CREATE INDEX IF NOT EXISTS faces_by_photo ON faces(photo_id);
    `)
  }

  photo(id: string) {
    return this.#db.prepare(`${selectPhotos} WHERE p.id = ? GROUP BY p.id`).get(id) as Photo | undefined
  }

  addPhoto(id: string, name: string, mime: Photo['mime']) {
    this.#db.prepare("INSERT INTO photos (id, name, mime, status) VALUES (?, ?, ?, 'queued')").run(id, name, mime)
  }

  pending() {
    return this.#db.prepare("SELECT id FROM photos WHERE status IN ('queued', 'analyzing') ORDER BY rowid")
      .all().map((row) => row.id as string)
  }

  setStatus(id: string, status: Photo['status'], error: string | null = null) {
    this.#db.prepare('UPDATE photos SET status = ?, error = ? WHERE id = ?').run(status, error, id)
  }

  removePhoto(id: string) {
    this.#db.prepare('DELETE FROM photos WHERE id = ?').run(id)
  }

  saveAnalysis(id: string, detections: DetectedFace[]) {
    this.#transaction(() => {
      if (!this.photo(id)) return
      const old = this.#db.prepare(`${selectFaces} WHERE photo_id = ? ORDER BY id`).all(id) as unknown as StoredFace[]
      const unmatched = new Set(old)
      const matches = detections.map((face) => {
        let best: StoredFace | undefined
        let score = 0.7
        for (const candidate of unmatched) {
          const fit = overlap(candidate, face.box)
          if (fit > score) {
            best = candidate
            score = fit
          }
        }
        if (best) unmatched.delete(best)
        return { face, best }
      })
      if ([...unmatched].some((face) => face.personId !== null)) {
        throw new Error('The new analysis could not preserve all assigned faces. The previous results were kept.')
      }

      const insert = this.#db.prepare('INSERT INTO faces (x, y, width, height, score, space_id, embedding, photo_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
      const update = this.#db.prepare('UPDATE faces SET x = ?, y = ?, width = ?, height = ?, score = ?, space_id = ?, embedding = ? WHERE id = ?')
      const remove = this.#db.prepare('DELETE FROM faces WHERE id = ?')
      for (const previous of unmatched) remove.run(previous.id)
      for (const { face, best } of matches) {
        const { x, y, width, height } = face.box
        const fields = [x, y, width, height, face.detectionScore, face.embedding.spaceId, encode(face.embedding.values)] as const
        if (best) update.run(...fields, best.id)
        else insert.run(...fields, id)
      }
      this.setStatus(id, 'ready')
    })
  }

  assign(faceId: number, selection: { personId: number | null } | { name: string }) {
    return this.#transaction(() => {
      if (!this.#db.prepare('SELECT id FROM faces WHERE id = ?').get(faceId)) return false
      let personId = 'personId' in selection ? selection.personId : null
      if ('name' in selection) {
        this.#db.prepare('INSERT INTO people (name) VALUES (?) ON CONFLICT(name) DO NOTHING').run(selection.name)
        personId = this.#db.prepare('SELECT id FROM people WHERE name = ?').get(selection.name)!.id as number
      } else if (personId !== null && !this.#db.prepare('SELECT id FROM people WHERE id = ?').get(personId)) {
        return false
      }
      this.#db.prepare('UPDATE faces SET person_id = ? WHERE id = ?').run(personId, faceId)
      return true
    })
  }

  library(engine: FaceEngine): Library {
    const photos = this.#db.prepare(`${selectPhotos} GROUP BY p.id ORDER BY p.rowid DESC`).all() as unknown as Photo[]
    const people = this.#db.prepare(`SELECT p.id, p.name, COUNT(f.id) AS faceCount
      FROM people p LEFT JOIN faces f ON f.person_id = p.id GROUP BY p.id ORDER BY p.name`).all() as unknown as Person[]
    const stored = this.#db.prepare(`${selectFaces} ORDER BY id`).all() as unknown as StoredFace[]
    const references = stored.flatMap((face) => face.personId === null ? [] : [{ personId: face.personId, embedding: decode(face) }])
    const faces = stored.map(({ id, photoId, x, y, width, height, score, personId, ...data }) => {
      const scores = new Map<number, number>()
      if (personId === null && references.length > 0) {
        const embedding = decode(data)
        for (const reference of references) {
          if (reference.embedding.spaceId !== embedding.spaceId) continue
          const similarity = engine.compare(embedding, reference.embedding)
          scores.set(reference.personId, Math.max(scores.get(reference.personId) ?? -1, similarity))
        }
      }
      const suggestions = [...scores].sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([personId, score]) => ({ personId, score }))
      return { id, photoId, box: { x, y, width, height }, score, personId, suggestions }
    })
    return { photos, people, faces }
  }

  close() {
    this.#db.close()
  }

  #transaction<T>(work: () => T): T {
    this.#db.exec('BEGIN IMMEDIATE')
    try {
      const result = work()
      this.#db.exec('COMMIT')
      return result
    } catch (error) {
      this.#db.exec('ROLLBACK')
      throw error
    }
  }
}
