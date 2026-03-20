export type TimePeriod = '1M' | '3M' | '1Y'

export interface RiskSignal {
  category: string
  sharedWith: string
  detail: string
  intensity: 'Low' | 'Moderate' | 'Elevated'
}

export interface Recommendation {
  ticker: string
  company: string
  sector: string
  sp500: boolean
  riskScore: number
  volatility: Record<TimePeriod, number>
  matchReason: string
  signals: RiskSignal[]
}

export interface PortfolioInsight {
  ticker: string
  riskScore: number
  signalSummary: string
}

export interface ScanResponse {
  baseRiskScore: number
  summary: string
  recommendations: Recommendation[]
  portfolioInsights: PortfolioInsight[]
}
