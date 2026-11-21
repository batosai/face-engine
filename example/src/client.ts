import type { Face, Library, Person, Photo } from './types.js'

type View = 'photos' | 'review' | 'people'
const content = document.querySelector<HTMLDivElement>('#content')!
const title = document.querySelector<HTMLHeadingElement>('#title')!
const subtitle = document.querySelector<HTMLParagraphElement>('#subtitle')!
const input = document.querySelector<HTMLInputElement>('#files')!
const importButton = document.querySelector<HTMLButtonElement>('#import')!
const notice = document.querySelector<HTMLParagraphElement>('#notice')!
const dialogNotice = document.querySelector<HTMLParagraphElement>('#dialog-notice')!
const reviewCount = document.querySelector<HTMLSpanElement>('#review-count')!
const dialog = document.querySelector<HTMLDialogElement>('#photo-dialog')!
const photoTitle = document.querySelector<HTMLHeadingElement>('#photo-title')!
const photoDetail = document.querySelector<HTMLDivElement>('#photo-detail')!
const tabs = document.querySelectorAll<HTMLButtonElement>('[data-view]')

let library: Library = { photos: [], people: [], faces: [] }
let view: View = 'photos'
let selectedPerson: number | null = null
let selectedPhoto: string | null = null
let importing = false
let refreshing: Promise<void> | undefined

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = '', text?: string) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function button(text: string, className: string, onClick: () => void) {
  const node = element('button', className, text)
  node.type = 'button'
  node.addEventListener('click', onClick)
  return node
}

function message(text = '', failure = false) {
  const region = dialog.open ? dialogNotice : notice
  region.textContent = text
  region.hidden = !text
  region.classList.toggle('failure', failure)
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'The request could not be completed.'
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { error?: string } | null
    throw new Error(payload?.error ?? `Request failed (${response.status}).`)
  }
  return (response.status === 204 ? undefined : await response.json()) as T
}

function changeView(next: View) {
  view = next
  selectedPerson = null
  render()
}

function photoUrl(id: string) {
  return `/api/photos/${id}/image`
}

function photoStatus(photo: Photo) {
  if (photo.status === 'queued') return 'Queued'
  if (photo.status === 'analyzing') return 'Analyzing'
  if (photo.status === 'failed') return 'Analysis failed'
  if (photo.faceCount === 0) return 'No faces'
  if (photo.unassignedCount > 0) return `${photo.unassignedCount} to review`
  return 'Sorted'
}

function plural(count: number, item: string) {
  return `${count} ${item}${count === 1 ? '' : 's'}`
}

function picture(photo: Photo, className = '') {
  const img = element('img', className)
  img.src = photoUrl(photo.id)
  img.alt = photo.name
  img.loading = 'lazy'
  return img
}

function crop(face: Face, label: string) {
  const canvas = element('canvas', 'face-crop')
  canvas.width = 192
  canvas.height = 192
  canvas.setAttribute('role', 'img')
  canvas.setAttribute('aria-label', label)
  const image = new Image()
  image.addEventListener('load', () => {
    const ctx = canvas.getContext('2d')
    if (!ctx || !canvas.isConnected) return
    const { x, y, width, height } = face.box
    const side = Math.min(Math.max(width, height) * 1.45, image.naturalWidth, image.naturalHeight)
    const left = Math.max(0, Math.min(image.naturalWidth - side, x + (width - side) / 2))
    const top = Math.max(0, Math.min(image.naturalHeight - side, y + (height - side) / 2))
    ctx.drawImage(image, left, top, side, side, 0, 0, canvas.width, canvas.height)
  }, { once: true })
  image.src = photoUrl(face.photoId)
  return canvas
}

function empty(heading: string, text: string) {
  const section = element('div', 'empty')
  section.append(element('h2', '', heading), element('p', '', text))
  return section
}

function photoGrid(photos: Photo[]) {
  const grid = element('div', 'photo-grid')
  for (const photo of photos) {
    const card = button('', 'photo-card', () => openPhoto(photo.id))
    const image = element('div', 'photo-frame')
    const badge = element('span', `badge ${photo.status === 'failed' ? 'badge-error' : ''}`, photoStatus(photo))
    image.append(picture(photo), badge)
    const label = element('div', 'photo-label')
    const name = element('strong', '', photo.name)
    name.title = photo.name
    label.append(name, element('span', '', plural(photo.faceCount, 'face')))
    card.append(image, label)
    grid.append(card)
  }
  return grid
}

function personName(id: number | null) {
  return library.people.find((person) => person.id === id)?.name
}

function action(control: HTMLButtonElement | HTMLSelectElement, work: () => Promise<unknown>) {
  control.disabled = true
  void (async () => {
    try {
      await work()
      await refreshAfterMutation()
      message()
    } catch (error) {
      message(errorMessage(error), true)
    } finally {
      control.disabled = false
    }
  })()
}

function assign(face: Face, selection: { personId: number | null } | { name: string }) {
  return request(`/api/faces/${face.id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(selection),
  })
}

function faceCard(face: Face, showSource: boolean) {
  const card = element('article', 'face-card')
  const source = library.photos.find((photo) => photo.id === face.photoId)
  const currentName = personName(face.personId)
  const body = element('div', 'face-body')
  const heading = element('div', 'face-heading')
  heading.append(element('strong', '', currentName ?? 'Unassigned face'))
  if (showSource && source) heading.append(button(source.name, 'text-button source-link', () => openPhoto(source.id)))
  body.append(heading)

  if (face.suggestions.length) {
    const suggestions = element('div', 'suggestions')
    suggestions.append(element('span', 'small-label', 'Suggested matches'))
    const list = element('div', 'suggestion-list')
    for (const { personId, score } of face.suggestions) {
      const name = personName(personId)
      if (!name) continue
      const choice = button(`${name} · ${score.toFixed(2)}`, 'button suggestion', () => action(choice, () => assign(face, { personId })))
      choice.setAttribute('aria-label', `Assign to ${name}, cosine score ${score.toFixed(2)}`)
      list.append(choice)
    }
    suggestions.append(list)
    body.append(suggestions)
  }

  if (library.people.length) {
    const select = element('select', 'control')
    select.setAttribute('aria-label', 'Assign an existing person')
    const unassigned = element('option', '', 'Unassigned')
    unassigned.value = ''
    select.append(unassigned)
    for (const person of library.people) {
      const option = element('option', '', person.name)
      option.value = `${person.id}`
      select.append(option)
    }
    select.value = face.personId?.toString() ?? ''
    select.addEventListener('change', () => action(select, () => assign(face, { personId: select.value ? Number(select.value) : null })))
    body.append(select)
  }

  const form = element('form', 'new-person')
  const name = element('input', 'control')
  name.placeholder = 'New person’s name'
  name.setAttribute('aria-label', 'New person’s name')
  name.required = true
  name.autocomplete = 'off'
  const submit = element('button', 'button', 'Create & assign')
  submit.type = 'submit'
  form.append(name, submit)
  form.addEventListener('submit', (event) => {
    event.preventDefault()
    if (name.value.trim()) action(submit, () => assign(face, { name: name.value.trim() }))
  })
  body.append(form)
  card.append(crop(face, currentName ?? 'Detected face'), body)
  return card
}

function openPhoto(id: string) {
  selectedPhoto = id
  dialogNotice.hidden = true
  renderPhoto()
  if (!dialog.open && selectedPhoto) dialog.showModal()
}

function renderPhoto() {
  const photo = library.photos.find(({ id }) => id === selectedPhoto)
  if (!photo) {
    selectedPhoto = null
    if (dialog.open) dialog.close()
    return
  }

  photoTitle.textContent = photo.name
  const layout = element('div', 'photo-detail')
  const preview = element('div', 'detail-preview')
  const image = picture(photo)
  image.loading = 'eager'
  preview.append(image)
  const side = element('div', 'detail-side')
  side.append(element('p', 'detail-status', photoStatus(photo)))
  if (photo.error) side.append(element('p', 'inline-error', photo.error))
  const faces = library.faces.filter((face) => face.photoId === photo.id)
  for (const face of faces) side.append(faceCard(face, false))

  const controls = element('div', 'detail-actions')
  const analyze = button('Analyze again', 'button', () => action(analyze, () => request(`/api/photos/${photo.id}/analyze`, { method: 'POST' })))
  analyze.disabled = photo.status === 'queued' || photo.status === 'analyzing'
  const remove = button('Delete photo', 'button danger', () => {
    if (!confirm(`Delete ${photo.name} and its detected faces?`)) return
    action(remove, async () => {
      await request(`/api/photos/${photo.id}`, { method: 'DELETE' })
      dialog.close()
    })
  })
  controls.append(analyze, remove)
  side.append(controls)
  layout.append(preview, side)
  photoDetail.replaceChildren(layout)
}

function personCard(person: Person) {
  const card = button('', 'person-card', () => {
    selectedPerson = person.id
    render()
  })
  const first = library.faces.find((face) => face.personId === person.id)
  const preview = first ? crop(first, person.name) : element('span', 'person-placeholder', person.name.slice(0, 1).toUpperCase())
  card.append(preview, element('strong', '', person.name), element('span', '', plural(person.faceCount, 'face')))
  return card
}

function render() {
  const unassigned = library.faces.filter((face) => face.personId === null)
  reviewCount.hidden = unassigned.length === 0
  reviewCount.textContent = String(unassigned.length)
  for (const tab of tabs) {
    if (tab.dataset.view === view) tab.setAttribute('aria-current', 'page')
    else tab.removeAttribute('aria-current')
  }

  if (view === 'photos') {
    title.textContent = 'Photos'
    subtitle.textContent = plural(library.photos.length, 'photo')
    const upload = element('div', 'drop-zone')
    const prompt = element('div')
    prompt.append(element('strong', '', 'Drop photos here'), element('p', '', 'JPEG or PNG'))
    upload.append(prompt, button('Choose files', 'button', () => input.click()))
    upload.addEventListener('dragover', (event) => {
      event.preventDefault()
      upload.classList.add('dragging')
    })
    upload.addEventListener('dragleave', (event) => {
      if (!upload.contains(event.relatedTarget as Node | null)) upload.classList.remove('dragging')
    })
    upload.addEventListener('drop', (event) => {
      event.preventDefault()
      upload.classList.remove('dragging')
      if (event.dataTransfer) void uploadFiles([...event.dataTransfer.files])
    })
    content.replaceChildren(upload, photoGrid(library.photos))
  } else if (view === 'review') {
    title.textContent = 'To review'
    subtitle.textContent = `${plural(unassigned.length, 'face')} awaiting assignment`
    const grid = element('div', 'review-grid')
    for (const face of unassigned) grid.append(faceCard(face, true))
    content.replaceChildren(unassigned.length ? grid : empty('You’re all caught up', 'Unassigned faces appear here after an import.'))
  } else {
    const person = library.people.find(({ id }) => id === selectedPerson)
    title.textContent = person?.name ?? 'People'
    subtitle.textContent = person ? plural(person.faceCount, 'face') : `${library.people.length} ${library.people.length === 1 ? 'person' : 'people'}`
    if (person) {
      const ids = new Set(library.faces.filter((face) => face.personId === person.id).map((face) => face.photoId))
      const photos = library.photos.filter(({ id }) => ids.has(id))
      const back = button('All people', 'button back', () => changeView('people'))
      content.replaceChildren(back, photos.length ? photoGrid(photos) : empty('No assigned photos', 'Photos appear here once you assign a face.'))
    } else {
      const grid = element('div', 'people-grid')
      for (const person of library.people) grid.append(personCard(person))
      content.replaceChildren(library.people.length ? grid : empty('No people yet', 'Create a person while reviewing a detected face.'))
    }
  }
  if (dialog.open) renderPhoto()
}

function refresh() {
  if (!refreshing) {
    refreshing = request<Library>('/api/library').then((next) => {
      library = next
      render()
    }).finally(() => { refreshing = undefined })
  }
  return refreshing
}

async function refreshAfterMutation() {
  if (refreshing) await refreshing.catch(() => {})
  await refresh()
}

async function uploadFiles(files: File[]) {
  if (files.length === 0 || importing) return
  importing = true
  importButton.disabled = true
  changeView('photos')
  const failures: string[] = []
  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    message(`Importing ${i + 1} of ${files.length}…`)
    try {
      await request('/api/photos', { method: 'POST', headers: { 'X-Filename': encodeURIComponent(file.name) }, body: file })
    } catch (error) {
      failures.push(`${file.name} — ${errorMessage(error)}`)
    }
  }
  try {
    await refreshAfterMutation()
    message(failures.join(' '), failures.length > 0)
  } catch (error) {
    message(errorMessage(error), true)
  } finally {
    importing = false
    importButton.disabled = false
    input.value = ''
  }
}

for (const tab of tabs) tab.addEventListener('click', () => changeView(tab.dataset.view as View))
importButton.addEventListener('click', () => input.click())
input.addEventListener('change', () => void uploadFiles([...(input.files ?? [])]))
dialog.addEventListener('close', () => { selectedPhoto = null })

void refresh().catch((error: unknown) => message(errorMessage(error), true))
setInterval(() => {
  if (document.activeElement instanceof HTMLInputElement || document.activeElement instanceof HTMLSelectElement) return
  if (library.photos.some((photo) => photo.status === 'queued' || photo.status === 'analyzing')) {
    void refresh().catch((error: unknown) => message(errorMessage(error), true))
  }
}, 700)
