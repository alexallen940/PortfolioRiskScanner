export interface Recommendation {
  ticker: string
  similarity: number
  riskScore?: number
  companyName?: string
  logoUrl?: string
  description?: string[]
}

export interface ScanResponse {
  baseRiskScore: number
  riskTypes: string[]
  summary: string
  recommendations: Recommendation[]
}
