export interface RiskScoreBreakdown {
  weights: {
    annualized_volatility: number
    max_drawdown: number
    var_95: number
    downside_volatility: number
    avg_daily_volume_inverse: number
  }
  components: {
    annualized_volatility: number
    max_drawdown_abs: number
    var_95_abs: number
    downside_volatility: number
    avg_daily_volume: number
    avg_daily_volume_inverse: number
  }
  weighted_components: {
    annualized_volatility: number
    max_drawdown: number
    var_95: number
    downside_volatility: number
    avg_daily_volume_inverse: number
  }
  raw_score: number
  raw_score_from_formula: number
  min_raw_score: number
  max_raw_score: number
  normalized_score: number
  normalized_score_from_formula: number
}

export interface PortfolioRiskBreakdown {
  weights: {
    annualized_volatility: number
    max_drawdown: number
    var_95: number
    downside_volatility: number
    avg_daily_volume_inverse: number
  }
  components: {
    annualized_volatility: number
    max_drawdown_abs: number
    var_95_abs: number
    downside_volatility: number
    avg_daily_volume: number
    avg_daily_volume_inverse: number
  }
  weighted_components: {
    annualized_volatility: number
    max_drawdown: number
    var_95: number
    downside_volatility: number
    avg_daily_volume_inverse: number
  }
  raw_score: number
  min_raw_score: number
  max_raw_score: number
  normalized_score: number
  final_score: number
  matched_tickers: string[]
}

export interface SimilarityDriver {
  dimension?: number
  label?: string
  query_value?: number
  stock_value?: number
  term?: string
  contribution: number
  relationship?: string
  top_positive_terms?: string[]
  top_negative_terms?: string[]
}

export interface SimilarityExplanation {
  method: 'svd_cosine' | 'tfidf_cosine'
  similarity_score: number
  dot_product: number
  query_norm: number
  stock_norm: number
  denominator: number
  portfolio_weight: number
  text_weight: number
  text_weight_level?: 'low' | 'medium' | 'high'
  top_drivers: SimilarityDriver[]
}

export interface DescriptionDetail {
  bullet: string
  headlines: Array<{
    title: string
    url?: string | null
  }>
}

export interface QueryInterpretation {
  original: string
  interpreted: string
  corrections: Record<string, string>
}

export interface RecommendationSentiment {
  label: string
  average_compound: number
  article_count: number
}

export interface Recommendation {
  ticker: string
  similarity: number
  similarityExplanation?: SimilarityExplanation
  riskScore?: number
  riskBreakdown?: RiskScoreBreakdown
  companyName?: string
  logoUrl?: string
  description?: string[]
  descriptionDetails?: DescriptionDetail[]
  sentiment?: RecommendationSentiment
}

export interface ScanResponse {
  baseRiskScore: number
  portfolioRiskBreakdown?: PortfolioRiskBreakdown
  riskTypes: string[]
  summary: string
  recommendations: Recommendation[]
  queryInterpretation?: QueryInterpretation
}
