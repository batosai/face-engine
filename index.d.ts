export type ImageInput = string | Uint8Array | {
  format: 'rgb24'
  data: Uint8Array
  width: number
  height: number
}

export type Point = { x: number; y: number }

export type Detection = {
  box: { x: number; y: number; width: number; height: number }
  detectionScore: number
  landmarks: [Point, Point, Point, Point, Point]
}

export declare class FaceEngine {
  private constructor()

  static create(options: {
    manifest: string
    threads?: number
    maxPending?: number
  }): Promise<FaceEngine>

  detect(image: ImageInput): Promise<Detection[]>
  close(): Promise<void>
}
