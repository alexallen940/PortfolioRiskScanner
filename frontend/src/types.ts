export interface Recommendation {
  ticker: string
  similarity: number
  description?: string[]
}

export interface ScanResponse {
  baseRiskScore: number
  riskTypes: string[]
  summary: string
  recommendations: Recommendation[]
}
