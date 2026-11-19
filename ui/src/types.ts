export interface TierSpec {
  name: string
  description: string
  models: Record<string, string>
}

export interface TiersResponse {
  tiers: TierSpec[]
  default_provider: string
  low_confidence_fallback: string
}

// P2 — accuracy-first selection mode (PLAN-v2.md §5). `null` on a rule means "no per-rule
// preference"; a string value forces that mode when the rule matches, overriding whatever the
// request asked for.
export type OptimizeFor = 'cost' | 'accuracy' | 'balanced'

export interface RuleEntry {
  when: Record<string, string | string[]>
  tier: string
  priority?: number
  optimize_for?: OptimizeFor | null
}

export interface CatalogEntry {
  id: string
  provider: string
  tier: string
  cost: number
  supports_image: boolean
  // P3 — PLAN-v2.md §6. Absent on older servers → treat as false in the UI.
  supports_video?: boolean
  supports_audio?: boolean
  label: string
  // P1 — PLAN-v2.md §7. Empty until an admin assigns a tier via PUT /v1/catalog/{id}.
  assigned_by?: string
}

/** Catalog-only sentinel — never routable until an admin assigns a real tier. */
export const UNREVIEWED_TIER = 'UNREVIEWED'

export interface HistoryEntry {
  timestamp: string
  metadata: Record<string, string>
  text: string
  has_image: boolean
  provider: string | null
  selected_models: string[] | null
  min_quality: number | null
  optimize_for_requested?: OptimizeFor
  optimize_for?: OptimizeFor
  tier: string
  model_name: string
  confidence: number | null
  reason: string
  rule_outcome: string
  classifier_outcome: string | null
  fallback_outcome: string | null
}

export interface QualityResponse {
  field: string | null
  scores?: { model_id: string; value: string; average: number }[]
  ratings?: { model_id: string; metadata: Record<string, string>; rating: number; judged_by: string }[]
}

// P0 — Auth (PLAN-v2.md §4). Mirrors the backend's `User.public()` shape; never carries
// the password hash (the server never puts it on the wire).
export type UserRole = 'admin' | 'member'

export interface User {
  id: string
  username: string
  role: UserRole
  created_at: string
  disabled: boolean
  notes: string
}

export interface AuthResponse {
  user: User
}

export interface UsersListResponse {
  users: User[]
}

export interface RulesResponse {
  rules: RuleEntry[]
}

export interface ExampleRow {
  example_id: string
  metadata: Record<string, string>
  text: string
  has_image: boolean
  covered_by_rule: boolean
  rule_reason: string | null
}

export interface UncoveredExample {
  example_id: string
  metadata: Record<string, string>
  text: string
  has_image: boolean
}

export interface TrainResult {
  n_labels: number
  n_rule_labels: number
  n_human_labels: number
  n_uncovered: number
  uses_images: boolean
  report: {
    accuracy?: number
    baseline_accuracy?: number
    accuracy_by_bucket?: Record<string, number>
    bucket_sizes?: Record<string, number>
    field?: string
  }
}

export interface Status {
  trained: boolean
  n_labels_last_trained_on: number | null
  uses_images: boolean
  /** Present on Gradist ≥0.5 — Settings / Bell use these. Older servers omit them. */
  quality_field?: string | null
  catalog_configured?: boolean
  allow_read_without_auth?: boolean
  unreviewed_count?: number
  quality_head?: boolean
  qc_risk?: boolean
}

export interface Decision {
  tier: string
  model_name: string
  confidence: number | null
  reason: string
  rule_outcome: string
  classifier_outcome: string | null
  fallback_outcome: string | null
  optimize_for: OptimizeFor
}
