from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Article(db.Model):
    __tablename__ = "articles"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.Text, nullable=False, index=True)
    headline = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"Article {self.id}: {self.ticker}, {self.headline}"


class RiskData(db.Model):
    __tablename__ = "sp500_risk_with_scores"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.Text, nullable=False, index=True)
    n_trading_days = db.Column(db.Integer, nullable=False)
    annualized_volatility = db.Column(db.Float, nullable=False)
    avg_daily_volume = db.Column(db.Float, nullable=False)
    max_drawdown = db.Column(db.Float, nullable=False)
    var_95 = db.Column(db.Float, nullable=False)
    downside_volatility = db.Column(db.Float, nullable=False)
    raw_risk_score = db.Column(db.Float, nullable=False)
    risk_score_1_10 = db.Column(db.Float, nullable=False, index=True)

    def __repr__(self):
        return f"<RiskData {self.ticker}: risk_score={self.risk_score_1_10}>"
