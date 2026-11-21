export type Box = { x: number; y: number; width: number; height: number }

export type Photo = {
  id: string
  name: string
  mime: 'image/jpeg' | 'image/png'
  status: 'queued' | 'analyzing' | 'ready' | 'failed'
  error: string | null
  faceCount: number
  unassignedCount: number
}

export type Person = { id: number; name: string; faceCount: number }

export type Face = {
  id: number
  photoId: string
  box: Box
  score: number
  personId: number | null
  suggestions: { personId: number; score: number }[]
}

export type Library = { photos: Photo[]; people: Person[]; faces: Face[] }
