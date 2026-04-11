import { FormEvent, useState } from 'react'
import './App.css'
import { Recommendation, ScanResponse } from './types'

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
                  <strong className="report-label">Portfolio Risk Score:</strong>{' '}
                  <span className="report-value">{results.baseRiskScore}/10</span>
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
                    <article key={stock.ticker} className="recommendation-card">
                      <p className="ticker-line"><span className="rec-number">#{index + 1}</span><span className="ticker-symbol">{stock.ticker}</span><span className="company-name">Similarity {(stock.similarity * 100).toFixed(1)}%</span></p>
                      <ul className="signal-bullets">
                        {(stock.description ?? ['No risk summary available yet.']).map((bullet, bulletIndex) => (
                          <li key={`${stock.ticker}-${bulletIndex}`}>{bullet}</li>
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
