import csv
import os
import nltk
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from models import db, Article, RiskData
from routes import register_routes
from services.recommender import INDEX


load_dotenv()

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


# src/ directory and project root (one level up)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

# Serve React build files from <project_root>/frontend/dist
app = Flask(
    __name__,
    static_folder=os.path.join(project_root, "frontend", "dist"),
    static_url_path="",
)
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)


def _init_articles():
    if Article.query.count():
        return
    csv_file_path = os.path.join(project_root, "data", "articles.csv")
    with open(csv_file_path, "r") as file:
        for row in csv.DictReader(file):
            db.session.add(
                Article(
                    ticker=row["ticker"],
                    headline=row["headline"],
                    summary=row["summary"],
                )
            )
    db.session.commit()
    print("Database initialized with articles data")


def _init_risk_data():
    if RiskData.query.count():
        return
    csv_file_path = os.path.join(project_root, "data", "sp500_risk_with_scores.csv")
    with open(csv_file_path, "r") as file:
        for row in csv.DictReader(file):
            db.session.add(
                RiskData(
                    ticker=row["ticker"],
                    n_trading_days=row["n_trading_days"],
                    annualized_volatility=row["annualized_volatility"],
                    avg_daily_volume=row["avg_daily_volume"],
                    max_drawdown=row["max_drawdown"],
                    var_95=row["var_95"],
                    downside_volatility=row["downside_volatility"],
                    raw_risk_score=row["raw_risk_score"],
                    risk_score_1_10=row["risk_score_1_10"],
                )
            )
    db.session.commit()
    print("Database initialized with data/sp500_risk_with_scores data")


def init_db():
    with app.app_context():
        db.create_all()
        _init_articles()
        _init_risk_data()


init_db()

with app.app_context():
    INDEX.build(max_features=5000, ngram_range=(1, 2), min_df=10, n_components=20, max_df=0.95)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
