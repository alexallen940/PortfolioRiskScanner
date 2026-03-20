import { FormEvent, useState } from 'react'
import './App.css'
import { Recommendation, ScanResponse, TimePeriod } from './types'

const recommendationCatalog: Recommendation[] = [
  {
    ticker: 'MSFT',
    company: 'Microsoft',
    sector: 'Technology',
    sp500: true,
    riskScore: 4.2,
    volatility: { '1M': 2.4, '3M': 5.8, '1Y': 14.1 },
    matchReason: 'Large-cap technology name with comparatively stable price action and recurring discussion around AI regulation and enterprise demand.',
    signals: [
      { category: 'Regulation', sharedWith: 'NVDA', detail: 'AI-policy scrutiny overlaps with semiconductor and cloud infrastructure exposure.', intensity: 'Moderate' },
      { category: 'Sentiment', sharedWith: 'AAPL', detail: 'News tone stays sensitive to product-cycle expectations and antitrust headlines.', intensity: 'Low' },
    ],
  },
  {
    ticker: 'XOM',
    company: 'Exxon Mobil',
    sector: 'Energy',
    sp500: true,
    riskScore: 6.5,
    volatility: { '1M': 4.8, '3M': 8.7, '1Y': 19.6 },
    matchReason: 'Energy exposure with macro and commodity-price sensitivity that fits portfolios looking for cyclical names and inflation-linked risk.',
    signals: [
      { category: 'Revenue Concerns', sharedWith: 'CVX', detail: 'Revenue sentiment tracks commodity demand and refining margin compression.', intensity: 'Elevated' },
      { category: 'Regulation', sharedWith: 'TSLA', detail: 'Climate policy and emissions reporting create cross-sector regulatory pressure.', intensity: 'Moderate' },
    ],
  },
  {
    ticker: 'LLY',
    company: 'Eli Lilly',
    sector: 'Healthcare',
    sp500: true,
    riskScore: 5.1,
    volatility: { '1M': 3.1, '3M': 6.4, '1Y': 17.3 },
    matchReason: 'Healthcare name with strong momentum but meaningful headline risk tied to pricing policy and trial updates.',
    signals: [
      { category: 'Regulation', sharedWith: 'PFE', detail: 'Drug-pricing news and FDA decisions create similar policy-driven swings.', intensity: 'Elevated' },
      { category: 'Sentiment', sharedWith: 'JNJ', detail: 'Investor sentiment shifts quickly on pipeline and reimbursement commentary.', intensity: 'Moderate' },
    ],
  },
  {
    ticker: 'JPM',
    company: 'JPMorgan Chase',
    sector: 'Financials',
    sp500: true,
    riskScore: 4.9,
    volatility: { '1M': 2.8, '3M': 5.9, '1Y': 15.2 },
    matchReason: 'Financials exposure with relatively controlled volatility and risk signals tied to rates, credit quality, and regulatory sentiment.',
    signals: [
      { category: 'Revenue Concerns', sharedWith: 'BAC', detail: 'Net interest margin commentary overlaps with broader banking pressure.', intensity: 'Moderate' },
      { category: 'Controversies', sharedWith: 'GS', detail: 'Sector-wide scrutiny around deal activity and capital rules keeps risk elevated.', intensity: 'Low' },
    ],
  },
  {
    ticker: 'NEE',
    company: 'NextEra Energy',
    sector: 'Utilities',
    sp500: true,
    riskScore: 3.8,
    volatility: { '1M': 2.2, '3M': 4.7, '1Y': 11.9 },
    matchReason: 'Lower-volatility utility profile that still captures regulatory and renewable-infrastructure themes.',
    signals: [
      { category: 'Regulation', sharedWith: 'DUK', detail: 'Rate-case decisions and clean-energy policy drive similar risk narratives.', intensity: 'Moderate' },
      { category: 'Sentiment', sharedWith: 'SO', detail: 'Investor tone shifts with rate expectations and dividend sustainability.', intensity: 'Low' },
    ],
  },
  {
    ticker: 'SHOP',
    company: 'Shopify',
    sector: 'Technology',
    sp500: false,
    riskScore: 7.2,
    volatility: { '1M': 6.2, '3M': 11.4, '1Y': 28.5 },
    matchReason: 'Higher-volatility growth stock that fits aggressive screens focused on digital commerce and sentiment-driven momentum.',
    signals: [
      { category: 'Sentiment', sharedWith: 'AMZN', detail: 'E-commerce demand expectations produce sharp narrative shifts after earnings.', intensity: 'Elevated' },
      { category: 'Revenue Concerns', sharedWith: 'META', detail: 'Growth durability is questioned when ad and consumer spending cools.', intensity: 'Moderate' },
    ],
  },
  {
    ticker: 'CAT',
    company: 'Caterpillar',
    sector: 'Industrials',
    sp500: true,
    riskScore: 5.8,
    volatility: { '1M': 3.6, '3M': 7.1, '1Y': 18.2 },
    matchReason: 'Industrial cyclicality with exposure to infrastructure demand, commodity activity, and global growth risk.',
    signals: [
      { category: 'Revenue Concerns', sharedWith: 'DE', detail: 'Capital spending slowdowns and order-book revisions affect both names.', intensity: 'Moderate' },
      { category: 'Sentiment', sharedWith: 'BA', detail: 'Macro confidence and manufacturing headlines move investor tone quickly.', intensity: 'Moderate' },
    ],
  },
]

const sectorKeywords: Record<string, string> = {
  technology: 'Technology',
  tech: 'Technology',
  ai: 'Technology',
  healthcare: 'Healthcare',
  biotech: 'Healthcare',
  finance: 'Financials',
  financial: 'Financials',
  bank: 'Financials',
  energy: 'Energy',
  oil: 'Energy',
  utility: 'Utilities',
  utilities: 'Utilities',
  industrial: 'Industrials',
  infrastructure: 'Industrials',
}

const signalKeywords: Record<string, string[]> = {
  regulation: ['Regulation'],
  regulatory: ['Regulation'],
  controversy: ['Controversies'],
  controversies: ['Controversies'],
  sentiment: ['Sentiment'],
  volatility: ['Sentiment', 'Revenue Concerns'],
  revenue: ['Revenue Concerns'],
  growth: ['Revenue Concerns', 'Sentiment'],
}

function parseTickers(input: string): string[] {
  return Array.from(
    new Set(
      input
        .split(/[\s,\n]+/)
        .map(token => token.trim().toUpperCase())
        .filter(token => /^[A-Z.]{1,5}$/.test(token)),
    ),
  )
}

function scorePortfolioTicker(ticker: string): number {
  const seed = ticker.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0)
  return Number((((seed % 55) / 10) + 3.5).toFixed(1))
}

function buildSignalSummary(ticker: string): string {
  const summaries = [
    'negative news clustering around leadership commentary and earnings guidance',
    'headline sensitivity driven by regulation and sentiment reversals',
    'risk profile shaped by volatility spikes after macro or sector news',
    'investor concern centered on demand durability and revenue visibility',
  ]
  const seed = ticker.charCodeAt(0) + ticker.charCodeAt(ticker.length - 1)
  return summaries[seed % summaries.length]
}

function detectPreferredSector(query: string): string | null {
  const normalizedQuery = query.toLowerCase()

  for (const [keyword, sector] of Object.entries(sectorKeywords)) {
    if (normalizedQuery.includes(keyword)) {
      return sector
    }
  }

  return null
}

function queryMentionsSignal(query: string, category: string): boolean {
  const normalizedQuery = query.toLowerCase()

  return Object.entries(signalKeywords).some(([keyword, categories]) => {
    return normalizedQuery.includes(keyword) && categories.includes(category)
  })
}

function rankRecommendations(
  recommendations: Recommendation[],
  rankingPeriod: TimePeriod,
  query: string,
  preferredSector: string | null,
): Recommendation[] {
  const normalizedQuery = query.toLowerCase()
  const wantsLowerRisk = /(low risk|conservative|stable|lower volatility|defensive)/.test(normalizedQuery)
  const wantsHigherRisk = /(high risk|aggressive|higher volatility|speculative|growth)/.test(normalizedQuery)

  return [...recommendations].sort((left, right) => {
    const leftSignalBoost = left.signals.filter(signal => queryMentionsSignal(query, signal.category)).length
    const rightSignalBoost = right.signals.filter(signal => queryMentionsSignal(query, signal.category)).length
    const leftSectorBoost = preferredSector !== null && left.sector === preferredSector ? 2 : 0
    const rightSectorBoost = preferredSector !== null && right.sector === preferredSector ? 2 : 0
    const leftRiskBoost = wantsLowerRisk ? -left.riskScore : wantsHigherRisk ? left.riskScore : 0
    const rightRiskBoost = wantsLowerRisk ? -right.riskScore : wantsHigherRisk ? right.riskScore : 0
    const leftScore = leftSignalBoost * 4 + leftSectorBoost * 3 + leftRiskBoost + left.volatility[rankingPeriod]
    const rightScore = rightSignalBoost * 4 + rightSectorBoost * 3 + rightRiskBoost + right.volatility[rankingPeriod]

    return rightScore - leftScore
  })
}

function generateScanResponse(
  portfolio: string[],
  query: string,
  rankingPeriod: TimePeriod,
  excludedSector: string,
  sp500Only: boolean,
): ScanResponse {
  const preferredSector = detectPreferredSector(query)
  const portfolioInsights = portfolio.map(ticker => ({
    ticker,
    riskScore: scorePortfolioTicker(ticker),
    signalSummary: buildSignalSummary(ticker),
  }))
  const baseRiskScore = Number(
    (
      portfolioInsights.reduce((sum, insight) => sum + insight.riskScore, 0) / portfolioInsights.length
    ).toFixed(1),
  )

  const filtered = recommendationCatalog.filter(stock => {
    if (portfolio.includes(stock.ticker)) return false
    if (excludedSector !== 'All' && stock.sector === excludedSector) return false
    if (sp500Only && !stock.sp500) return false
    return true
  })

  const ranked = rankRecommendations(filtered, rankingPeriod, query, preferredSector).slice(0, 4)
  const summaryParts = [
    `Base portfolio risk score ${baseRiskScore}/10 from ${portfolio.length} holdings.`,
    preferredSector ? `Query emphasis detected for ${preferredSector.toLowerCase()} names.` : 'Query did not force a single sector preference.',
    `Recommendations ranked by ${rankingPeriod} volatility with shared risk-theme weighting.`,
  ]

  return {
    baseRiskScore,
    summary: summaryParts.join(' '),
    recommendations: ranked,
    portfolioInsights,
  }
}

function App(): JSX.Element {
  const [portfolioInput, setPortfolioInput] = useState('')
  const [queryInput, setQueryInput] = useState('')
  const [validationMessage, setValidationMessage] = useState('')
  const [results, setResults] = useState<ScanResponse | null>(() => {
    return generateScanResponse(
      ['AAPL', 'NVDA', 'JPM'],
      'Looking for relatively stable technology or healthcare stocks with moderate risk and similar regulatory or sentiment concerns.',
      '3M',
      'All',
      true,
    )
  })

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    const parsedPortfolio = parseTickers(portfolioInput)

    if (parsedPortfolio.length === 0) {
      setValidationMessage('Enter at least one stock ticker using commas, spaces, or new lines.')
      setResults(null)
      return
    }

    if (queryInput.trim().length < 12) {
      setValidationMessage('Add a slightly more specific query so the recommendation cards have useful context.')
      setResults(null)
      return
    }

    setValidationMessage('')
    setResults(generateScanResponse(parsedPortfolio, queryInput, '3M', 'All', true))
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <h1 className="site-title">Portfolio Risk Scanner</h1>
        <p className="site-description">
          Enter your current holdings and describe the kind of stock you want to add. The tool returns a portfolio risk score and recommended stocks with desired characteristics and risk signal context.
        </p>
      </header>

      <section className="workspace-grid">
        <form className="control-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <h2>Search Inputs</h2>
            <p>Use tickers only for the portfolio. Describe the risk profile, industry, or concerns in plain language.</p>
          </div>

          <label className="field-block" htmlFor="portfolio-input">
            <span>Portfolio tickers</span>
            <textarea
              id="portfolio-input"
              value={portfolioInput}
              onChange={event => setPortfolioInput(event.target.value)}
              placeholder="AAPL, NVDA, JPM"
              rows={4}
            />
          </label>

          <label className="field-block" htmlFor="query-input">
            <span>Desired stock characteristics</span>
            <textarea
              id="query-input"
              value={queryInput}
              onChange={event => setQueryInput(event.target.value)}
              placeholder="Example: I want lower-volatility healthcare or tech stocks with moderate risk and sensitivity to regulation."
              rows={6}
            />
          </label>

          <button className="submit-button" type="submit">Generate matches</button>

          {validationMessage && <p className="validation-message">{validationMessage}</p>}
        </form>

        <section className="results-panel">
          <div className="panel-header">
            <h2>Risk Report</h2>
          </div>

          {results && (
            <>
              <div className="risk-score-section">
                <p className="risk-score-line">
                  <strong className="report-label">Portfolio Risk Score:</strong>{' '}
                  <span className="report-value">{results.baseRiskScore}/10</span>
                </p>
                <p className="risk-types-line">
                  <strong className="report-label">Risk Types:</strong>{' '}
                  <span className="report-value">{Array.from(new Set(results.recommendations.flatMap(r => r.signals.map(s => s.category)))).join(', ')}</span>
                </p>
              </div>

              <div className="recommendations-section">
                <h3 className="section-divider">Stock Suggestions</h3>
                <div className="recommendation-list">
                  {results.recommendations.map((stock, index) => (
                    <article key={stock.ticker} className="recommendation-card">
                      <p className="ticker-line"><span className="rec-number">#{index + 1}</span><span className="ticker-symbol">{stock.ticker}</span><span className="company-name">{stock.company}</span></p>
                      <ul className="signal-bullets">
                        {stock.signals.map(signal => (
                          <li key={`${stock.ticker}-${signal.category}`}>{signal.detail}</li>
                        ))}
                      </ul>
                    </article>
                  ))}
                </div>
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  )
}

export default App
