import { FormEvent, useEffect, useState } from 'react'
import './App.css'
import { Recommendation, ScanResponse } from './types'

function StockLogo({ ticker, companyName, logoUrl }: { ticker: string; companyName?: string; logoUrl?: string }): JSX.Element {
  const [hasImageError, setHasImageError] = useState(false)

  if (logoUrl && !hasImageError) {
    return (
      <span className="company-logo has-image" aria-hidden="true">
        <img src={logoUrl} alt="" loading="lazy" onError={() => setHasImageError(true)} />
      </span>
    )
  }

  return (
    <span className="company-logo" aria-label={companyName ?? ticker}>
      {ticker.slice(0, 2)}
    </span>
  )
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

async function postJson<T>(url: string, payload: object): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`)
  }

  return data as T
}

async function fetchRecommendationDescriptions(recommendations: Recommendation[]): Promise<Recommendation[]> {
  const withDescriptions = await Promise.all(
    recommendations.map(async recommendation => {
      try {
        const data = await postJson<{ description: string[] }>('/api/portfolio/recommendation-description', {
          ticker: recommendation.ticker,
        })
        return { ...recommendation, description: data.description }
      } catch {
        return recommendation
      }
    }),
  )

  return withDescriptions
}

function App(): JSX.Element {
  const [portfolioInput, setPortfolioInput] = useState('')
  const [queryInput, setQueryInput] = useState('')
  const [validationMessage, setValidationMessage] = useState('')
  const [results, setResults] = useState<ScanResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedRecommendation, setSelectedRecommendation] = useState<Recommendation | null>(null)
  const [isFormulaOpen, setIsFormulaOpen] = useState(false)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        setSelectedRecommendation(null)
        setIsFormulaOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()
    const parsedPortfolio = parseTickers(portfolioInput)

    if (parsedPortfolio.length === 0) {
      setValidationMessage('Enter at least one stock ticker using commas, spaces, or new lines.')
      setResults(null)
      return
    }

    if (queryInput.trim().length === 0) {
      setValidationMessage('Enter at least one desired stock characteristic.')
      setResults(null)
      return
    }

    setValidationMessage('')
    setIsLoading(true)

    try {
      const [riskScoreData, riskTypesData, recsData] = await Promise.all([
        postJson<{ risk_score: number }>('/api/portfolio/risk-score', { portfolio: parsedPortfolio }),
        postJson<{ risk_types: string[] }>('/api/portfolio/risk-types', { portfolio: parsedPortfolio }),
        postJson<{ recommendations: Array<{ ticker: string; similarity: number }> }>('/api/portfolio/recommendations', {
          desired_characteristics: queryInput.trim(),
          portfolio: parsedPortfolio,
        }),
      ])

      const baseRecommendations: Recommendation[] = recsData.recommendations.slice(0, 4).map(rec => ({
        ticker: rec.ticker,
        similarity: rec.similarity,
        riskScore: rec.risk_score,
        companyName: rec.company_name,
        logoUrl: rec.logo_url,
      }))

      const recommendations = await fetchRecommendationDescriptions(baseRecommendations)

      const response: ScanResponse = {
        baseRiskScore: riskScoreData.risk_score,
        riskTypes: Array.from(new Set(riskTypesData.risk_types)).slice(0, 5),
        summary: `Generated ${recommendations.length} recommendations for ${parsedPortfolio.length} holdings. Query context: ${queryInput.trim()}`,
        recommendations,
      }

      setResults(response)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to connect to backend services.'
      setResults(null)
      setValidationMessage(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <div className="page-orb page-orb-left" aria-hidden="true" />
      <div className="page-orb page-orb-right" aria-hidden="true" />
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

          <button className="submit-button" type="submit" disabled={isLoading}>
            {isLoading ? 'Running scan...' : 'Generate matches'}
          </button>

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
                  <strong className="report-label">Portfolio Risk Score:</strong>
                  <span className="report-value">{results.baseRiskScore}/10</span>
                  <span
                    className="info-popover-wrap"
                    onMouseEnter={() => setIsFormulaOpen(true)}
                    onMouseLeave={() => setIsFormulaOpen(false)}
                  >
                    <button
                      type="button"
                      className="formula-trigger"
                      aria-label="Show risk score formula"
                      aria-expanded={isFormulaOpen}
                      onClick={() => setIsFormulaOpen(open => !open)}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M7 3.75A2.25 2.25 0 0 0 4.75 6v12A2.25 2.25 0 0 0 7 20.25h10A2.25 2.25 0 0 0 19.25 18V6A2.25 2.25 0 0 0 17 3.75H7ZM6.25 6c0-.41.34-.75.75-.75h10c.41 0 .75.34.75.75v12c0 .41-.34.75-.75.75H7a.75.75 0 0 1-.75-.75V6Zm2 1.75a.75.75 0 0 1 .75-.75h6a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm0 4a.75.75 0 0 1 .75-.75h1.25a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm4.25 0a.75.75 0 0 1 .75-.75h1.75a.75.75 0 0 1 0 1.5H13.25a.75.75 0 0 1-.75-.75Zm-4.25 4a.75.75 0 0 1 .75-.75h1.25a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm4.25 0a.75.75 0 0 1 .75-.75h1.75a.75.75 0 0 1 0 1.5H13.25a.75.75 0 0 1-.75-.75Z" />
                      </svg>
                    </button>
                    {isFormulaOpen && (
                      <div className="info-popover formula-popover" role="dialog" aria-label="Risk score formula details">
                        <p className="popover-kicker">Formula preview</p>
                        <p>Add your risk-score formula here.</p>
                        <p className="popover-note">This pop-up supports hover and click so you can swap in the final methodology later.</p>
                      </div>
                    )}
                  </span>
                </p>
                <p className="risk-types-line">
                  <strong className="report-label">Risk Types:</strong>{' '}
                  <span className="report-value">{results.riskTypes.length > 0 ? results.riskTypes.join(', ') : 'No strong risk signals detected'}</span>
                </p>
              </div>

              <div className="recommendations-section">
                <h3 className="section-divider">Stock Suggestions</h3>
                <div className="recommendation-list">
                  {results.recommendations.map((stock, index) => (
                    <button
                      key={stock.ticker}
                      type="button"
                      className="recommendation-card"
                      onClick={() => setSelectedRecommendation(stock)}
                    >
                      <span className="recommendation-card-copy">
                        <span className="suggestion-header">
                          <span className="suggestion-left">
                            <span className="rec-number">#{index + 1}</span>
                            <StockLogo ticker={stock.ticker} companyName={stock.companyName} logoUrl={stock.logoUrl} />
                            <span className="company-copy">
                              <span className="ticker-symbol">{stock.ticker}</span>
                              <span className="company-name">{stock.companyName ?? stock.ticker}</span>
                            </span>
                          </span>
                          <span className="suggestion-risk-block">
                            <span className="suggestion-risk-label">Similarity</span>
                            <strong className="suggestion-risk-value">{(stock.similarity * 100).toFixed(1)}%</strong>
                          </span>
                        </span>
                        <ul className="signal-bullets compact-bullets">
                          {(stock.description ?? ['No risk summary available yet.']).slice(0, 2).map((bullet, bulletIndex) => (
                            <li key={`${stock.ticker}-${bulletIndex}`}>{bullet}</li>
                          ))}
                        </ul>
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </section>
      </section>

      {selectedRecommendation && (
        <div className="modal-backdrop" role="presentation" onClick={() => setSelectedRecommendation(null)}>
          <section
            className="detail-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="recommendation-title"
            onClick={event => event.stopPropagation()}
          >
            <div className="detail-modal-header">
              <div>
                <p className="modal-kicker">Suggestion detail</p>
                <h3 id="recommendation-title">{selectedRecommendation.companyName ?? selectedRecommendation.ticker}</h3>
                <p className="modal-subtitle">{selectedRecommendation.ticker}</p>
              </div>
              <button
                type="button"
                className="modal-close"
                aria-label="Close suggestion details"
                onClick={() => setSelectedRecommendation(null)}
              >
                ×
              </button>
            </div>

            <div className="detail-metrics">
              <div className="detail-metric-card">
                <span>Risk score</span>
                <strong>{selectedRecommendation.riskScore !== undefined ? `${selectedRecommendation.riskScore.toFixed(1)}/10` : 'N/A'}</strong>
              </div>
              <div className="detail-metric-card">
                <span>Similarity</span>
                <strong>{(selectedRecommendation.similarity * 100).toFixed(1)}%</strong>
              </div>
            </div>

            <div className="detail-section">
              <h4>Risk signal notes</h4>
              <ul className="signal-bullets detail-bullets">
                {(selectedRecommendation.description ?? ['No risk summary available yet.']).map((bullet, bulletIndex) => (
                  <li key={`${selectedRecommendation.ticker}-${bulletIndex}`}>{bullet}</li>
                ))}
              </ul>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}

export default App
