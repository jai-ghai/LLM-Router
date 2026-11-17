import type {
  CatalogEntry,
  Decision,
  ExampleRow,
  HistoryEntry,
  QualityResponse,
  RuleEntry,
  RulesResponse,
  Status,
  TiersResponse,
  TrainResult,
  UncoveredExample,
} from './types'

const BASE: string = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `Request failed with status ${status}`)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
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
  tiers: (signal?: AbortSignal) => request<TiersResponse>('/v1/tiers', { signal }),
  rules: (signal?: AbortSignal) => request<RulesResponse>('/v1/rules', { signal }),
  saveRules: (rules: RuleEntry[]) => request<RulesResponse>('/v1/rules', { method: 'PUT', ...json({ rules }) }),
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
  history: (limit = 50, signal?: AbortSignal) =>
    request<{ history: HistoryEntry[] }>(`/v1/history?limit=${limit}`, { signal }),
  quality: (signal?: AbortSignal) => request<QualityResponse>('/v1/quality', { signal }),
  rateQuality: (input: { modelId: string; metadata: Record<string, string>; rating: number; judgedBy: string; notes?: string }) =>
    request<{ status: string }>(
      '/v1/quality',
      json({ model_id: input.modelId, metadata: input.metadata, rating: input.rating, judged_by: input.judgedBy, notes: input.notes ?? '' }),
    ),
  resolve: (
    input: ExampleInput & {
      provider?: string
      minConfidence?: number
      selectedModels?: string[] | null
      minQuality?: number
    },
  ) => {
    const form = toFormData(input)
    if (input.provider) form.set('provider', input.provider)
    if (input.minConfidence !== undefined) form.set('min_confidence', String(input.minConfidence))
    if (input.selectedModels) form.set('selected_models', JSON.stringify(input.selectedModels))
    if (input.minQuality !== undefined) form.set('min_quality', String(input.minQuality))
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
