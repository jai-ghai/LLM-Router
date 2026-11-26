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

// Accuracy-first selection mode. `null` on a rule means "no per-rule
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
  // Absent on older servers → treat as false in the UI.
  supports_video?: boolean
  supports_audio?: boolean
  label: string
  // Empty until an admin assigns a tier via PUT /v1/catalog/{id}.
  assigned_by?: string
  capability?: number | null
  capability_source?: string
}

/** Catalog-only sentinel — never routable until an admin assigns a real tier. */
export const UNREVIEWED_TIER = 'UNREVIEWED'

export interface HistoryEntry {
  /** Absent on entries recorded before the feedback loop existed; those rate by model only. */
  request_id?: string
  skill_needs?: Partial<Record<Skill, number>>
  difficulty?: number | null
  /** Ratings given to this request's pick (by request_id). */
  ratings?: { rating: number; judged_by: string }[]
  /** The tier's top candidates at resolve time, best first; Compare offers these. */
  shortlist?: string[]
  preferences?: { other: string; outcome: PreferenceOutcome; skills: Skill[]; judged_by: string }[]
  /** The latest reviewer correction of this request's needs. */
  need_label?: { needs: Partial<Record<Skill, number>>; judged_by: string }
  timestamp: string
  /** Absent on entries recorded before auth existed. */
  requested_by?: string
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

// Auth. Mirrors the backend's `User.public()` shape; never carries
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
  /** Rate the pick with POST /v1/quality {request_id}; the rating teaches the model's skills. */
  request_id?: string | null
  tier: string
  model_name: string
  confidence: number | null
  reason: string
  rule_outcome: string
  classifier_outcome: string | null
  fallback_outcome: string | null
  optimize_for: OptimizeFor
  tier_scores?: Record<string, number> | null
  tier_scores_kind?: string | null
  candidates?: {
    id: string
    provider: string
    tier: string
    cost: number
    capability?: number | null
    capability_source?: string
    fit?: number
    fit_parts?: Partial<Record<Skill | 'capability' | 'price', number>>
    skill_sources?: Partial<Record<Skill, string>>
    learned?: Partial<Record<Skill, { prior: number; value: number; ratings: number }>>
  }[]
  pool_size?: number | null
  signals?: {
    text_chars?: number
    text_words?: number
    metadata?: Record<string, string>
    image_bytes?: number
    modality?: string
    providers?: string[]
    selection_size?: number | null
    path?: string
    metadata_used?: string[]
    metadata_ignored?: string[]
    classifier_inputs?: ClassifierInputs
    screen_text?: ScreenText
    skill_needs?: SkillNeeds
    /** Models in the tier whose skills ratings have moved, with their rank now (may be outside the top 8). */
    moved_by_ratings?: {
      id: string
      rank: number
      fit: number
      learned: Partial<Record<Skill, { prior: number; value: number; ratings: number }>>
    }[]
  }
}

export type Skill = 'video' | 'audio' | 'hindi' | 'reasoning' | 'long_input' | 'writing' | 'fast'

/** Which output was better: the router's pick, the other model's, or neither. */
export type PreferenceOutcome = 'picked' | 'other' | 'tie'

export interface SkillNeeds {
  needs: Record<Skill, number>
  reasons: Partial<Record<Skill, string[]>>
  task_profile: string | null
  input_tokens?: number
  /** How hard this request looks, 0–1; scales the capability weight by 0.5 + difficulty. Null = neutral. */
  difficulty?: number | null
  difficulty_reasons?: string[]
}

export interface ScreenText {
  engine: string | null
  lines: { text: string; confidence: number }[]
  frames_read?: number
  lines_per_frame?: number[]
  error?: string
}

/** What the domain classifier read from a request, and its tier odds with one input blanked. */
export interface ClassifierInputs {
  metadata_buckets: Record<string, string>
  metadata_columns: number
  text_embedder: string
  text_dims: number
  image_embedder: string | null
  image_dims: number
  image_used: boolean
  screen_text_used: boolean
  without_image: Record<string, number>
  without_text: Record<string, number>
  without_screen_text: Record<string, number>
  without_metadata: Record<string, number>
}
