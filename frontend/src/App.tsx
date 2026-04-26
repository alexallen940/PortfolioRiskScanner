import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import "./App.css";
import {
  DescriptionDetail,
  PortfolioRiskBreakdown,
  Recommendation,
  RiskScoreBreakdown,
  ScanResponse,
  SimilarityExplanation,
} from "./types";

function StockLogo({
  ticker,
  companyName,
  logoUrl,
}: {
  ticker: string;
  companyName?: string;
  logoUrl?: string;
}): JSX.Element {
  
  const [hasImageError, setHasImageError] = useState(false);

  if (logoUrl && !hasImageError) {
    return (
      <span className="company-logo has-image" aria-hidden="true">
        <img
          src={logoUrl}
          alt=""
          loading="lazy"
          onError={() => setHasImageError(true)}
        />
      </span>
    );
  }

  return (
    <span className="company-logo" aria-label={companyName ?? ticker}>
      {ticker.slice(0, 2)}
    </span>
  );
}

function parseTickers(input: string): string[] {
  return Array.from(
    new Set(
      input
        .split(/[\s,\n]+/)
        .map((token) => token.trim().toUpperCase())
        .filter((token) => /^[A-Z.]{1,5}$/.test(token)),
    ),
  );
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      const nextChar = line[i + 1];
      if (inQuotes && nextChar === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  cells.push(current.trim());
  return cells;
}

function parseTickersFromCsv(content: string): string[] {
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headerCells = parseCsvLine(lines[0]).map((cell) => cell.toLowerCase());
  const tickerColumnIndex = headerCells.findIndex((cell) =>
    ["ticker", "symbol", "stock", "stocks"].includes(cell),
  );

  const startRow = tickerColumnIndex >= 0 ? 1 : 0;
  const tickers = new Set<string>();

  for (let i = startRow; i < lines.length; i += 1) {
    const cells = parseCsvLine(lines[i]);
    if (tickerColumnIndex >= 0) {
      const token = cells[tickerColumnIndex] ?? "";
      parseTickers(token).forEach((ticker) => tickers.add(ticker));
    } else {
      cells.forEach((token) => {
        parseTickers(token).forEach((ticker) => tickers.add(ticker));
      });
    }
  }

  return Array.from(tickers);
}

async function postJson<T>(url: string, payload: object): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }

  return data as T;
}

async function fetchRecommendationDescriptions(
  recommendations: Recommendation[],
  use_llm: boolean,
): Promise<Recommendation[]> {
  const withDescriptions = await Promise.all(
    recommendations.map(async (recommendation) => {
      try {
        const data = await postJson<{
          description: string[];
          description_details?: DescriptionDetail[];
        }>("/api/portfolio/recommendation-description", {
          ticker: recommendation.ticker,
          use_llm: use_llm,
        });
        return {
          ...recommendation,
          description: data.description,
          descriptionDetails: data.description_details,
        };
      } catch {
        return recommendation;
      }
    }),
  );

  return withDescriptions;
}

async function fetchRecommendationSummaries(
  tickers: string[],
): Promise<Record<string, string>> {
  if (tickers.length === 0) {
    return {};
  }

  try {
    const data = await postJson<Record<string, string>>(
      "/api/portfolio/recommendations-summary",
      {
        tickers,
        positive_bias: false,
      },
    );
    return data;
  } catch {
    return {};
  }
}

function formatNumber(value: number | undefined, digits = 4): string {
  if (value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return value.toFixed(digits);
}

function formatWeight(weight: number): string {
  return weight.toFixed(2);
}

function queryWeightLabel(level: "low" | "medium" | "high"): string {
  if (level === "low") return "Low (110)";
  if (level === "high") return "High (200)";
  return "Medium (150)";
}

function sentimentBadgeClass(label?: string): string {
  switch (label) {
    case "very positive":
      return "sentiment-badge very-positive";
    case "positive":
      return "sentiment-badge positive";
    case "slightly positive":
      return "sentiment-badge slightly-positive";
    case "very negative":
      return "sentiment-badge very-negative";
    case "negative":
      return "sentiment-badge negative";
    case "slightly negative":
      return "sentiment-badge slightly-negative";
    default:
      return "sentiment-badge neutral";
  }
}

function formatSentimentLabel(label?: string): string {
  if (!label) return "Neutral";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function App(): JSX.Element {
  const [portfolioInput, setPortfolioInput] = useState("");
  const [queryInput, setQueryInput] = useState("");
  const [queryWeightLevel, setQueryWeightLevel] = useState<
    "low" | "medium" | "high"
  >("medium");
  const [validationMessage, setValidationMessage] = useState("");
  const [csvTickers, setCsvTickers] = useState<string[]>([]);
  const [csvFileName, setCsvFileName] = useState("");
  const [csvLoadMessage, setCsvLoadMessage] = useState("");
  const [results, setResults] = useState<ScanResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedRecommendationKey, setExpandedRecommendationKey] = useState<
    string | null
  >(null);
  const [isFormulaOpen, setIsFormulaOpen] = useState(false);
  const [suggestionsTab, setSuggestionsTab] = useState<"ir" | "llm">("llm");
  const [theme, setTheme] = useState<"light" | "dark">(() => {
  const stored = localStorage.getItem("theme");

  if (stored === "light" || stored === "dark") return stored;
  
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);  

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        setExpandedRecommendationKey(null);
        setIsFormulaOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleCsvUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
    const file = event.target.files?.[0];

    if (!file) {
      setCsvTickers([]);
      setCsvFileName("");
      setCsvLoadMessage("");
      return;
    }

    try {
      const content = await file.text();
      const parsedFromCsv = parseTickersFromCsv(content);
      setCsvTickers(parsedFromCsv);
      setCsvFileName(file.name);

      if (parsedFromCsv.length === 0) {
        setCsvLoadMessage("CSV loaded, but no valid tickers were found.");
      } else {
        setCsvLoadMessage(
          `Loaded ${parsedFromCsv.length} tickers from ${file.name}.`,
        );
      }
    } catch {
      setCsvTickers([]);
      setCsvFileName(file.name);
      setCsvLoadMessage("Unable to read this CSV file.");
    }
  };

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>,
  ): Promise<void> => {
    event.preventDefault();
    const parsedPortfolio = Array.from(
      new Set([...parseTickers(portfolioInput), ...csvTickers]),
    );

    if (parsedPortfolio.length === 0) {
      setValidationMessage(
        "Enter at least one stock ticker in the textbox or upload a CSV file.",
      );
      setResults(null);
      return;
    }

    // if (queryInput.trim().length === 0) {
    //   setValidationMessage("Enter at least one desired stock characteristic.");
    //   setResults(null);
    //   return;
    // }

    setValidationMessage("");
    setIsLoading(true);

    try {
      const [riskScoreData, riskTypesData, recsData] = await Promise.all([
        postJson<{
          risk_score: number;
          risk_breakdown?: PortfolioRiskBreakdown;
        }>("/api/portfolio/risk-score", {
          portfolio: parsedPortfolio,
        }),
        postJson<{ risk_types: string[] }>("/api/portfolio/risk-types", {
          portfolio: parsedPortfolio,
        }),
        postJson<{
          recommendations: Array<{
            ticker: string;
            similarity: number;
            similarity_explanation?: SimilarityExplanation;
            risk_score?: number;
            risk_breakdown?: RiskScoreBreakdown;
            company_name?: string;
            logo_url?: string;
            sentiment?: {
              label: string;
              average_compound: number;
              article_count: number;
            };
          }>;
          ir_recommendations: Array<{
            ticker: string;
            similarity: number;
            similarity_explanation?: SimilarityExplanation;
            risk_score?: number;
            risk_breakdown?: RiskScoreBreakdown;
            company_name?: string;
            logo_url?: string;
            sentiment?: {
              label: string;
              average_compound: number;
              article_count: number;
            };
          }>;
          query_interpretation?: {
            original: string;
            interpreted: string;
            corrections: Record<string, string>;
          };
        }>("/api/portfolio/recommendations", {
          desired_characteristics: queryInput.trim(),
          portfolio: parsedPortfolio,
          query_weight_level: queryWeightLevel,
        }),
      ]);

      const mapRec = (
        rec: (typeof recsData.recommendations)[0],
      ): Recommendation => ({
        ticker: rec.ticker,
        similarity: rec.similarity,
        similarityExplanation: rec.similarity_explanation,
        riskScore: rec.risk_score,
        riskBreakdown: rec.risk_breakdown,
        companyName: rec.company_name,
        logoUrl: rec.logo_url,
        sentiment: rec.sentiment,
      });

      const baseRecommendations = recsData.recommendations
        .slice(0, 4)
        .map(mapRec);
      const baseIrRecommendations = (
        recsData.ir_recommendations ?? recsData.recommendations
      )
        .slice(0, 4)
        .map(mapRec);

      const summaryByTicker = await fetchRecommendationSummaries(
        baseIrRecommendations.map((rec) => rec.ticker),
      );

      const withSummary = (recommendation: Recommendation): Recommendation => ({
        ...recommendation,
        llmSummary: summaryByTicker[recommendation.ticker],
      });

      const [recommendations, irRecommendations] = await Promise.all([
        fetchRecommendationDescriptions(baseRecommendations, true),
        fetchRecommendationDescriptions(baseIrRecommendations, false),
      ]);

      const recommendationsWithSummary = recommendations.map(withSummary);
      const irRecommendationsWithSummary = irRecommendations.map(withSummary);

      const response: ScanResponse = {
        baseRiskScore: riskScoreData.risk_score,
        portfolioRiskBreakdown: riskScoreData.risk_breakdown,
        riskTypes: Array.from(new Set(riskTypesData.risk_types)).slice(0, 5),
        summary: `Generated ${recommendationsWithSummary.length} recommendations for ${parsedPortfolio.length} holdings. Query context: ${queryInput.trim()}`,
        recommendations: recommendationsWithSummary,
        irRecommendations: irRecommendationsWithSummary,
        queryInterpretation: recsData.query_interpretation,
      };

      setResults(response);
      setExpandedRecommendationKey(null);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to connect to backend services.";
      setResults(null);
      setValidationMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <div className="page-orb page-orb-left" aria-hidden="true" />
      <div className="page-orb page-orb-right" aria-hidden="true" />
      <header className="site-header">
        <div className="site-header-top">
          <h1 className="site-title">Portfolio Risk Scanner</h1>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>
        </div>
        <p className="site-description">
          Enter your current holdings and describe the kind of stock you want to
          add. The tool returns a portfolio risk score and recommended stocks
          with desired characteristics and risk signal context.
        </p>
      </header>

      <section className="workspace-grid">
      <form className="control-panel-form" onSubmit={handleSubmit}>
        <section className="control-panel">
          <div className="panel-header">
            <h2>Portfolio</h2>
            <p>
              Provide a list of your portfolio tickers using the portfolio text
              box (separated by commas) and/or a CSV file (separated by
              whitespace, commas, or cells, though a ticker column would be
              optimal). All inputted tickers will be included across the text
              box and CSV.
            </p>
          </div>

          <label className="field-block" htmlFor="portfolio-input">
            <span>Portfolio tickers</span>
            <textarea
              id="portfolio-input"
              value={portfolioInput}
              onChange={(event) => setPortfolioInput(event.target.value)}
              placeholder="AAPL, NVDA, JPM"
              rows={4}
            />
          </label>

          <label className="field-block" htmlFor="portfolio-csv-input">
            <span>Portfolio CSV</span>
            <input
              id="portfolio-csv-input"
              type="file"
              accept=".csv,text/csv"
              onChange={handleCsvUpload}
            />
            {csvLoadMessage && (
              <small className="csv-upload-note">{csvLoadMessage}</small>
            )}
            {csvFileName && (
              <small className="csv-upload-note">
                Using file: {csvFileName}
              </small>
            )}
          </label>
        </section>

        <section className="control-panel">
          <div className="panel-header">
            <h2>Query</h2>
            <p>
              Describe the risk profile, industry, or other characteristics in
              plain language.
            </p>
            <p className="field-note">
              Your free-text query influences which stocks are suggested, but
              the risk bullet points are based on each stock&apos;s risk-signal
              analysis and may not directly reflect your query wording.
            </p>
          </div>

          <label className="field-block" htmlFor="query-input">
            <span>Desired stock characteristics</span>
            <textarea
              id="query-input"
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              placeholder="Example: I want lower-volatility healthcare or tech stocks with moderate risk and sensitivity to regulation."
              rows={6}
            />
          </label>

          <label className="field-block" htmlFor="query-weight-level">
            <span>Free-text query weighting</span>
            <select
              id="query-weight-level"
              value={queryWeightLevel}
              disabled={queryInput.trim().length === 0}
              onChange={(event) =>
                setQueryWeightLevel(
                  event.target.value as "low" | "medium" | "high",
                )
              }
            >
              <option value="low">Low (110)</option>
              <option value="medium">Medium (150)</option>
              <option value="high">High (200)</option>
            </select>
            <small className="csv-upload-note">
              Controls how strongly your stock characteristics description
              influences matching.
            </small>
          </label>
        </section>

        <div className="form-actions">
          <button className="submit-button" type="submit" disabled={isLoading}>
            {isLoading ? "Running scan..." : "Generate matches"}
          </button>

          {validationMessage && (
            <p className="validation-message">{validationMessage}</p>
          )}
        </div>
      </form>

        <section className="results-panel">
          <div className="panel-header">
            <h2>Risk Report</h2>
          </div>

          {results && (
            <>
              {results.queryInterpretation?.interpreted && (
                <div className="query-interpretation-card">
                  <p className="query-interp-kicker">AI query expansion</p>
                  <div className="query-interp-row">
                    <span className="query-interp-label">Original</span>
                    <span className="query-interp-value">
                      {results.queryInterpretation.original}
                    </span>
                  </div>
                  <div className="query-interp-row">
                    <span className="query-interp-label">Expanded</span>
                    <span className="query-interp-value query-interp-expanded">
                      {results.queryInterpretation.interpreted}
                    </span>
                  </div>
                </div>
              )}
              <div className="risk-score-section">
                <p className="risk-score-line">
                  <strong className="report-label">
                    Portfolio Risk Score:
                  </strong>
                  <span className="report-value">
                    {results.baseRiskScore}/10
                  </span>
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
                      onClick={() => setIsFormulaOpen((open) => !open)}
                    >
                      <svg viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M7 3.75A2.25 2.25 0 0 0 4.75 6v12A2.25 2.25 0 0 0 7 20.25h10A2.25 2.25 0 0 0 19.25 18V6A2.25 2.25 0 0 0 17 3.75H7ZM6.25 6c0-.41.34-.75.75-.75h10c.41 0 .75.34.75.75v12c0 .41-.34.75-.75.75H7a.75.75 0 0 1-.75-.75V6Zm2 1.75a.75.75 0 0 1 .75-.75h6a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm0 4a.75.75 0 0 1 .75-.75h1.25a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm4.25 0a.75.75 0 0 1 .75-.75h1.75a.75.75 0 0 1 0 1.5H13.25a.75.75 0 0 1-.75-.75Zm-4.25 4a.75.75 0 0 1 .75-.75h1.25a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Zm4.25 0a.75.75 0 0 1 .75-.75h1.75a.75.75 0 0 1 0 1.5H13.25a.75.75 0 0 1-.75-.75Z" />
                      </svg>
                    </button>
                    {isFormulaOpen && (
                      <div
                        className="info-popover formula-popover"
                        role="dialog"
                        aria-label="Risk score formula details"
                      >
                        <p className="popover-kicker">
                          Portfolio risk score calculation
                        </p>
                        <p className="popover-subheading">General formula</p>
                        <p>
                          Raw Risk Score = 0.30(AV) + 0.25(MDD) + 0.20(VaR) +
                          0.15(DV) + 0.10*(1/(ADV+1))
                        </p>
                        <p>
                          Final risk score is the raw risk score normalized to a
                          10 scale and rounded to 2 decimal places.
                        </p>

                        {results.portfolioRiskBreakdown ? (
                          <>
                            <p className="popover-subheading">
                              Your portfolio values
                            </p>
                            <p className="popover-note">
                              Matched tickers:{" "}
                              {results.portfolioRiskBreakdown.matched_tickers.join(
                                ", ",
                              ) || "None"}
                            </p>
                            <p>
                              Raw Risk Score ={" "}
                              {formatWeight(
                                results.portfolioRiskBreakdown.weights
                                  .annualized_volatility,
                              )}
                              *
                              {formatNumber(
                                results.portfolioRiskBreakdown.components
                                  .annualized_volatility,
                              )}{" "}
                              +{" "}
                              {formatWeight(
                                results.portfolioRiskBreakdown.weights
                                  .max_drawdown,
                              )}
                              *
                              {formatNumber(
                                results.portfolioRiskBreakdown.components
                                  .max_drawdown_abs,
                              )}{" "}
                              +{" "}
                              {formatWeight(
                                results.portfolioRiskBreakdown.weights.var_95,
                              )}
                              *
                              {formatNumber(
                                results.portfolioRiskBreakdown.components
                                  .var_95_abs,
                              )}{" "}
                              +{" "}
                              {formatWeight(
                                results.portfolioRiskBreakdown.weights
                                  .downside_volatility,
                              )}
                              *
                              {formatNumber(
                                results.portfolioRiskBreakdown.components
                                  .downside_volatility,
                              )}{" "}
                              +{" "}
                              {formatWeight(
                                results.portfolioRiskBreakdown.weights
                                  .avg_daily_volume_inverse,
                              )}
                              *
                              {formatNumber(
                                results.portfolioRiskBreakdown.components
                                  .avg_daily_volume_inverse,
                                8,
                              )}{" "}
                              ={" "}
                              <strong>
                                {formatNumber(
                                  results.portfolioRiskBreakdown.raw_score,
                                  6,
                                )}
                              </strong>
                            </p>
                            <p>
                              Final Risk Score ={" "}
                              <strong>
                                {results.baseRiskScore.toFixed(2)}
                              </strong>
                            </p>
                          </>
                        ) : (
                          <p>No portfolio risk breakdown available yet.</p>
                        )}
                        <p className="popover-subheading">Formula terms</p>
                        <ul className="popover-list">
                          <li>
                            AV: annualized volatility, a measure of risk based
                            on historical price fluctuations.
                          </li>
                          <li>
                            MDD: maximum drawdown, largest peak-to-trough
                            decline before a new peak.
                          </li>
                          <li>
                            VaR (95%): expected worst loss with 95% confidence.
                          </li>
                          <li>
                            DV: downside volatility, volatility of returns below
                            target.
                          </li>
                          <li>
                            ADV: average daily trading volume, used as liquidity
                            proxy.
                          </li>
                        </ul>
                      </div>
                    )}
                  </span>
                </p>
                <p className="risk-types-line">
                  <strong className="report-label">
                    Portfolio's Risk Types:
                  </strong>{" "}
                  <span className="report-value">
                    {results.riskTypes.length > 0
                      ? results.riskTypes.join(", ")
                      : "No strong risk signals detected"}
                  </span>
                </p>
              </div>

              <div className="recommendations-section">
                <h3 className="section-divider">Stock Suggestions</h3>
                <div className="suggestions-tabs">
                  <button
                    type="button"
                    className={`suggestions-tab${suggestionsTab === "ir" ? " active" : ""}`}
                    onClick={() => setSuggestionsTab("ir")}
                  >
                    IR ranking
                  </button>
                  <button
                    type="button"
                    className={`suggestions-tab${suggestionsTab === "llm" ? " active" : ""}`}
                    onClick={() => setSuggestionsTab("llm")}
                  >
                    AI ranking
                  </button>
                </div>
                <p className="suggestions-tab-note">
                  {suggestionsTab === "ir"
                    ? "Ranked by cosine similarity between your query and each stock's article profile."
                    : "Re-rankings, risk signal refinement, and ticker summary by AI using the expanded query and article evidence."}
                </p>
                <div className="recommendation-list">
                  {(suggestionsTab === "llm"
                    ? results.recommendations
                    : results.irRecommendations
                  ).map((stock, index) => {
                    const recommendationKey = `${suggestionsTab}:${stock.ticker}`;
                    const isExpanded =
                      expandedRecommendationKey === recommendationKey;

                    return (
                      <article key={recommendationKey} className="recommendation-card">
                        <div className="recommendation-card-copy">
                          <span className="suggestion-header">
                            <span className="suggestion-left">
                              <span className="rec-number">#{index + 1}</span>
                              <StockLogo
                                ticker={stock.ticker}
                                companyName={stock.companyName}
                                logoUrl={stock.logoUrl}
                              />
                              <span className="company-copy">
                                <span className="ticker-symbol">
                                  {stock.ticker}
                                </span>
                                <span className="company-name">
                                  {stock.companyName ?? stock.ticker}
                                </span>
                              </span>
                            </span>
                            <span className="suggestion-risk-block">
                              {suggestionsTab === "ir" && (
                                <>
                                  <span className="suggestion-risk-label">
                                    Similarity
                                  </span>
                                  <strong className="suggestion-risk-value">
                                    {(stock.similarity * 100).toFixed(1)}%
                                  </strong>
                                </>
                              )}
                              {stock.sentiment && (
                                <span
                                  className={sentimentBadgeClass(
                                    stock.sentiment.label,
                                  )}
                                >
                                  {formatSentimentLabel(stock.sentiment.label)}
                                </span>
                              )}
                            </span>
                          </span>
                          <ul className="signal-bullets compact-bullets">
                            {(
                              stock.description ?? [
                                "No risk summary available yet.",
                              ]
                            )
                              .slice(0, 2)
                              .map((bullet, bulletIndex) => (
                                <li key={`${stock.ticker}-${bulletIndex}`}>
                                  {bullet}
                                </li>
                              ))}
                          </ul>
                          {suggestionsTab === "llm" && (
                            <p className="recommendation-summary">
                              {stock.llmSummary ??
                                "Summary unavailable for this recommendation."}
                            </p>
                          )}
                          <button
                            type="button"
                            className="recommendation-expand-toggle"
                            aria-expanded={isExpanded}
                            onClick={() =>
                              setExpandedRecommendationKey((current) =>
                                current === recommendationKey
                                  ? null
                                  : recommendationKey,
                              )
                            }
                          >
                            {isExpanded ? "Hide details" : "Show details"}
                          </button>

                          {isExpanded && (
                            <div className="recommendation-dropdown">
                              <div className="detail-metrics">
                                <div className="detail-metric-card">
                                  <span className="detail-metric-label">
                                    Risk score
                                  </span>
                                  <span className="detail-metric-value">
                                    {stock.riskScore !== undefined
                                      ? `${stock.riskScore.toFixed(1)}/10`
                                      : "N/A"}
                                  </span>
                                </div>
                                <div className="detail-metric-card">
                                  <span className="detail-metric-label">
                                    Similarity
                                  </span>
                                  <span className="detail-metric-value">
                                    {(stock.similarity * 100).toFixed(1)}%
                                  </span>
                                </div>
                                {stock.sentiment && (
                                  <div className="detail-metric-card">
                                    <span className="detail-metric-label">
                                      Sentiment
                                    </span>
                                    <span className="detail-metric-value">
                                      {formatSentimentLabel(stock.sentiment.label)}
                                    </span>
                                  </div>
                                )}
                              </div>
                              {stock.sentiment && (
                                <p className="sentiment-detail-line">
                                  Average compound score:{" "}
                                  <strong>
                                    {stock.sentiment.average_compound.toFixed(3)}
                                  </strong>{" "}
                                  based on{" "}
                                  <strong>{stock.sentiment.article_count}</strong>{" "}
                                  recent articles.
                                </p>
                              )}

                              <div className="detail-section">
                                <h4>Risk signal notes</h4>
                                <ul className="signal-bullets detail-bullets">
                                  {stock.descriptionDetails?.length
                                    ? stock.descriptionDetails.map(
                                      (detail, bulletIndex) => (
                                        <li
                                          key={`${stock.ticker}-detail-${bulletIndex}`}
                                        >
                                          <span className="risk-bullet-text">
                                            {detail.bullet}
                                          </span>
                                          {detail.headlines.length > 0 && (
                                            <>
                                              <div className="headlines-label">
                                                Relevant headlines:
                                              </div>
                                              <ul className="headline-samples">
                                                {detail.headlines.map((hl, hlIndex) => (
                                                  <li
                                                    key={`${stock.ticker}-hl-${bulletIndex}-${hlIndex}`}
                                                  >
                                                    {hl.url ? (
                                                      <a
                                                        href={hl.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                      >
                                                        {hl.title}
                                                      </a>
                                                    ) : (
                                                      hl.title
                                                    )}
                                                  </li>
                                                ))}
                                              </ul>
                                            </>
                                          )}
                                        </li>
                                      ),
                                    )
                                    : (
                                      stock.description ?? [
                                        "No risk summary available yet.",
                                      ]
                                    ).map((bullet, bulletIndex) => (
                                      <li key={`${stock.ticker}-${bulletIndex}`}>
                                        {bullet}
                                      </li>
                                    ))}
                                </ul>
                              </div>

                              <div className="detail-section">
                                <h4>Similarity score calculation</h4>
                                <p className="formula-line">
                                  Similarity is cosine similarity between your
                                  weighted query and this stock profile. Weights
                                  used in this run: portfolio ={" "}
                                  <strong>
                                    {stock.similarityExplanation
                                      ?.portfolio_weight ?? 1}
                                  </strong>
                                  , free-text query ={" "}
                                  <strong>
                                    {stock.similarityExplanation?.text_weight ??
                                      150}
                                  </strong>{" "}
                                  (
                                  {queryWeightLabel(
                                    (stock.similarityExplanation
                                      ?.text_weight_level as
                                      | "low"
                                      | "medium"
                                      | "high") || "medium",
                                  )}
                                  ).
                                </p>
                                {stock.similarityExplanation && (
                                  <>
                                    <p className="formula-line">
                                      Formula: similarity = dot(query, stock) /
                                      (||query|| x ||stock||)
                                    </p>
                                    <p className="formula-line">
                                      Filled values:{" "}
                                      {formatNumber(
                                        stock.similarityExplanation.dot_product,
                                        6,
                                      )}{" "}
                                      / (
                                      {formatNumber(
                                        stock.similarityExplanation.query_norm,
                                        6,
                                      )}{" "}
                                      x{" "}
                                      {formatNumber(
                                        stock.similarityExplanation.stock_norm,
                                        6,
                                      )}
                                      ) ={" "}
                                      <strong>
                                        {formatNumber(
                                          stock.similarityExplanation
                                            .similarity_score,
                                          6,
                                        )}
                                      </strong>
                                    </p>
                                  </>
                                )}
                                {stock.similarityExplanation?.top_drivers
                                  ?.length ? (
                                  <ul className="formula-parts similarity-parts">
                                    {stock.similarityExplanation.top_drivers
                                      .slice(0, 3)
                                      .map((driver, driverIndex) => (
                                        <li key={`${stock.ticker}-driver-${driverIndex}`}>
                                          {driver.term ? (
                                            <>
                                              Shared term <strong>{driver.term}</strong>
                                            </>
                                          ) : (
                                            <>
                                              <strong>
                                                {driver.label ??
                                                  `Latent dimension ${driver.dimension}`}
                                              </strong>{" "}
                                              ({`dimension ${driver.dimension}`})
                                              {driver.relationship
                                                ? ` | Relationship: ${driver.relationship}`
                                                : ""}
                                              {driver.top_positive_terms?.length
                                                ? ` | Positive terms: ${driver.top_positive_terms.join(", ")}`
                                                : ""}
                                              {driver.top_negative_terms?.length
                                                ? ` | Negative terms: ${driver.top_negative_terms.join(", ")}`
                                                : ""}
                                            </>
                                          )}
                                        </li>
                                      ))}
                                  </ul>
                                ) : (
                                  <p className="formula-line">
                                    No term-level similarity drivers were
                                    available for this recommendation.
                                  </p>
                                )}
                              </div>

                              {stock.riskBreakdown && (
                                <div className="detail-section formula-breakdown-section">
                                  <h4>Risk score calculation</h4>
                                  <div className="formula-grid">
                                    <div className="formula-column">
                                      <p className="formula-subheading">
                                        General formula
                                      </p>
                                      <p className="formula-line">
                                        Raw Risk Score = 0.30(AV) + 0.25(MDD) +
                                        0.20(VaR) + 0.15(DV) + 0.10*(1/(ADV+1))
                                      </p>
                                      <p className="formula-line">
                                        Final risk score is the raw risk score
                                        normalized to a 10 scale and rounded to 2
                                        decimal places.
                                      </p>
                                    </div>
                                    <div className="formula-column">
                                      <p className="formula-subheading">
                                        This suggestion&apos;s values
                                      </p>
                                      <p className="formula-line">
                                        Raw Risk Score ={" "}
                                        {formatWeight(
                                          stock.riskBreakdown.weights
                                            .annualized_volatility,
                                        )}
                                        *
                                        {formatNumber(
                                          stock.riskBreakdown.components
                                            .annualized_volatility,
                                        )}{" "}
                                        +{" "}
                                        {formatWeight(
                                          stock.riskBreakdown.weights.max_drawdown,
                                        )}
                                        *
                                        {formatNumber(
                                          stock.riskBreakdown.components
                                            .max_drawdown_abs,
                                        )}{" "}
                                        +{" "}
                                        {formatWeight(
                                          stock.riskBreakdown.weights.var_95,
                                        )}
                                        *
                                        {formatNumber(
                                          stock.riskBreakdown.components.var_95_abs,
                                        )}{" "}
                                        +{" "}
                                        {formatWeight(
                                          stock.riskBreakdown.weights
                                            .downside_volatility,
                                        )}
                                        *
                                        {formatNumber(
                                          stock.riskBreakdown.components
                                            .downside_volatility,
                                        )}{" "}
                                        +{" "}
                                        {formatWeight(
                                          stock.riskBreakdown.weights
                                            .avg_daily_volume_inverse,
                                        )}
                                        *
                                        {formatNumber(
                                          stock.riskBreakdown.components
                                            .avg_daily_volume_inverse,
                                          8,
                                        )}{" "}
                                        ={" "}
                                        <strong>
                                          {formatNumber(
                                            stock.riskBreakdown.raw_score_from_formula,
                                            6,
                                          )}
                                        </strong>
                                      </p>
                                      <p className="formula-line">
                                        Final Risk Score ={" "}
                                        <strong>
                                          {stock.riskBreakdown.normalized_score.toFixed(
                                            2,
                                          )}
                                        </strong>
                                      </p>
                                    </div>
                                  </div>

                                  <p className="formula-subheading">
                                    Formula terms
                                  </p>
                                  <ul className="formula-parts">
                                    <li>
                                      AV: annualized volatility, a measure of risk
                                      based on historical price fluctuations.
                                    </li>
                                    <li>
                                      MDD: maximum drawdown, largest peak-to-trough
                                      decline before a new peak.
                                    </li>
                                    <li>
                                      VaR (95%): expected worst loss with 95%
                                      confidence.
                                    </li>
                                    <li>
                                      DV: downside volatility, volatility of returns
                                      below target.
                                    </li>
                                    <li>
                                      ADV: average daily trading volume, used as
                                      liquidity proxy.
                                    </li>
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  );
}

export default App;
