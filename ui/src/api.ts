import type {
  AuthResponse,
  CatalogEntry,
  Deadline,
  Decision,
  ExampleRow,
  HistoryEntry,
  OptimizeFor,
  PreferenceOutcome,
  QualityResponse,
  RuleEntry,
  RulesResponse,
  ScreenText,
  Skill,
  Status,
  TiersResponse,
  TrainResult,
  UncoveredExample,
  UsersListResponse,
  VideoProfile,
} from './types'

// Production builds default to same-origin, where vercel.json routes /v1 to the API.
const BASE: string = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `Request failed with status ${status}`)
    this.status = status
    this.detail = detail
  }
}

/**
 * All requests send `credentials: 'include'` so the signed session cookie from
 * SessionMiddleware flows cross-origin to the UI. The backend's
 * `allow_credentials=True` on CORS matches this; specific origins (not `*`) are required
 * at the backend for credentialed CORS to work.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { credentials: 'include', ...init })
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body ? (body as { detail: unknown }).detail : body
    throw new ApiError(response.status, detail)
  }
  return body as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export interface Health {
  status: string
}

export interface ExampleInput {
  metadata: Record<string, string>
  text: string
  image: File | null
}

function toFormData(input: ExampleInput): FormData {
  const form = new FormData()
  form.set('metadata', JSON.stringify(input.metadata))
  form.set('text', input.text)
  if (input.image) form.set('image', input.image)
  return form
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>('/healthz', { signal }),
  // P0 — auth
  login: (username: string, password: string) =>
    request<AuthResponse>('/v1/auth/login', json({ username, password })),
  logout: () => request<{ status: string }>('/v1/auth/logout', { method: 'POST' }),
  me: (signal?: AbortSignal) => request<AuthResponse>('/v1/auth/me', { signal }),
  listUsers: (signal?: AbortSignal) => request<UsersListResponse>('/v1/users', { signal }),
  createUser: (input: { username: string; password: string; role: 'admin' | 'member'; notes?: string }) =>
    request<AuthResponse>('/v1/users', json({ ...input, notes: input.notes ?? '' })),
  updateUser: (userId: string, patch: { disabled: boolean }) =>
    // method after ...json(...) — json() defaults to POST and would overwrite PUT.
    request<AuthResponse>(`/v1/users/${encodeURIComponent(userId)}`, { ...json(patch), method: 'PUT' }),
  tiers: (signal?: AbortSignal) => request<TiersResponse>('/v1/tiers', { signal }),
  rules: (signal?: AbortSignal) => request<RulesResponse>('/v1/rules', { signal }),
  saveRules: (rules: RuleEntry[]) =>
    request<RulesResponse>('/v1/rules', { ...json({ rules }), method: 'PUT' }),
  addExample: (input: ExampleInput) =>
    request<ExampleRow>('/v1/examples', { method: 'POST', body: toFormData(input) }),
  examples: (signal?: AbortSignal) => request<{ examples: ExampleRow[] }>('/v1/examples', { signal }),
  uncovered: (signal?: AbortSignal) =>
    request<{ examples: UncoveredExample[] }>('/v1/examples/uncovered', { signal }),
  addJudgment: (exampleId: string, tier: string, judgedBy: string, notes: string) =>
    request<{ status: string }>(
      '/v1/judgments',
      json({ example_id: exampleId, tier, judged_by: judgedBy, notes }),
    ),
  train: () => request<TrainResult>('/v1/train', { method: 'POST' }),
  status: (signal?: AbortSignal) => request<Status>('/v1/status', { signal }),
  catalog: (signal?: AbortSignal) => request<{ catalog: CatalogEntry[] }>('/v1/catalog', { signal }),
  // P1 — admin-only tier assignment / demotion. Returns the full catalog after update.
  updateCatalogTier: (modelId: string, tier: string) =>
    request<{ catalog: CatalogEntry[] }>(`/v1/catalog/${encodeURIComponent(modelId)}`, {
      ...json({ tier }),
      method: 'PUT',
    }),
  // Called by Catalog.tsx (admin "Label by cost") → POST /v1/catalog/label-tiers.
  labelCatalogTiers: (force = true) =>
    request<{ catalog: CatalogEntry[] }>(
      `/v1/catalog/label-tiers${force ? '?force=true' : ''}`,
      { method: 'POST' },
    ),
  // Called by Catalog.tsx + Resolve.tsx for task→model pools (user: map selective models to a task).
  taskPools: (signal?: AbortSignal) =>
    request<{ pools: Record<string, string[]> }>('/v1/task-pools', { signal }),
  channels: () => request<{ channels: Record<string, Record<string, string>> }>('/v1/channels'),
  saveTaskPool: (task: string, models: string[]) =>
    request<{ pools: Record<string, string[]> }>(`/v1/task-pools/${encodeURIComponent(task)}`, {
      ...json({ models }),
      method: 'PUT',
    }),
  deleteTaskPool: (task: string) =>
    request<{ pools: Record<string, string[]> }>(`/v1/task-pools/${encodeURIComponent(task)}`, {
      method: 'DELETE',
    }),
  /** Members always get only their own requests; `requestedBy` narrows an admin's view. */
  history: (limit = 50, signal?: AbortSignal, requestedBy?: string) =>
    request<{ history: HistoryEntry[] }>(
      `/v1/history?limit=${limit}${requestedBy ? `&requested_by=${encodeURIComponent(requestedBy)}` : ''}`,
      { signal },
    ),
  quality: (signal?: AbortSignal) => request<QualityResponse>('/v1/quality', { signal }),
  // Returns the configured quality-bucket field name, or null.
  qualityField: (signal?: AbortSignal) =>
    request<{ field: string | null }>('/v1/quality/field', { signal }),
  // judged_by is intentionally omitted — the server records the session user (P0 anti-
  // impersonation). Passing it would be ignored anyway.
  // With requestId the server takes the model, metadata and skill needs from that request.
  rateQuality: (input: {
    modelId: string
    metadata: Record<string, string>
    rating: number
    notes?: string
    requestId?: string | null
    wrongSkills?: Skill[]
  }) =>
    request<{ status: string }>(
      '/v1/quality',
      json({
        model_id: input.modelId,
        metadata: input.metadata,
        rating: input.rating,
        notes: input.notes ?? '',
        request_id: input.requestId ?? '',
        wrong_skills: input.wrongSkills ?? [],
      }),
    ),
  comparePick: (input: { requestId: string; otherModel: string; outcome: PreferenceOutcome; skills?: Skill[]; notes?: string }) =>
    request<{ status: string }>(
      '/v1/preferences',
      json({
        request_id: input.requestId,
        other_model: input.otherModel,
        outcome: input.outcome,
        skills: input.skills ?? [],
        notes: input.notes ?? '',
      }),
    ),
  correctNeeds: (requestId: string, needs: Partial<Record<Skill, number>>) =>
    request<{ status: string }>('/v1/need-labels', json({ request_id: requestId, needs })),
  ocr: (frames: File[]) => {
    const form = new FormData()
    for (const frame of frames) form.append('frames', frame)
    return request<ScreenText>('/v1/ocr', { method: 'POST', body: form })
  },
  /** Reads the clip once on the server; pass the returned profile_id with every resolve about it. */
  profileVideo: (video: File) => {
    const form = new FormData()
    form.set('video', video)
    return request<{ profile_id: string; profile: VideoProfile }>('/v1/video/profile', { method: 'POST', body: form })
  },
  resolve: (
    input: ExampleInput & {
      providers?: string[]
      minConfidence?: number
      selectedModels?: string[] | null
      task?: string
      minQuality?: number
      optimizeFor?: OptimizeFor
      ocrFrames?: File[]
      videoProfileId?: string
      deadline?: Deadline
    },
  ) => {
    const form = toFormData(input)
    for (const frame of input.ocrFrames ?? []) form.append('ocr_frames', frame)
    if (input.videoProfileId) form.set('video_profile_id', input.videoProfileId)
    if (input.deadline) form.set('deadline', input.deadline)
    if (input.providers?.length) form.set('provider', input.providers.join(','))
    if (input.minConfidence !== undefined) form.set('min_confidence', String(input.minConfidence))
    if (input.selectedModels) form.set('selected_models', JSON.stringify(input.selectedModels))
    if (input.task) form.set('task', input.task)
    if (input.minQuality !== undefined) form.set('min_quality', String(input.minQuality))
    if (input.optimizeFor) form.set('optimize_for', input.optimizeFor)
    return request<Decision>('/v1/resolve', { method: 'POST', body: form })
  },
}

export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return typeof error.detail === 'string' ? error.detail : error.message
  }
  if (error instanceof DOMException && error.name === 'AbortError') return 'Cancelled'
  return error instanceof Error ? error.message : 'Unknown error'
}
