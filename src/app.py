import json
import csv
import os
from dotenv import load_dotenv
from flask import Flask
from datetime import datetime
from services.risk import get_portfolio_risk_score, get_portfolio_risk_types


load_dotenv()
from flask_cors import CORS
from models import db, Article, RiskData
from routes import register_routes

# src/ directory and project root (one level up)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__, static_folder=os.path.join(project_root, "frontend", "dist"), static_url_path="")
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)


# Function to initialize database, change this to your own database initialization logic
def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()

        if Article.query.count() == 0:
            csv_file_path = os.path.join(project_root, "data", "articles.csv")
            with open(csv_file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    article = Article(
                        ticker=row["ticker"],
                        headline=row["headline"],
                        summary=row["summary"],
                    )
                    db.session.add(article)

            db.session.commit()

            print("Database initialized with articles data")

        if RiskData.query.count() == 0:
            csv_file_path = os.path.join(project_root, "data", "sp500_risk_with_scores.csv")
            with open(csv_file_path, "r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    riskData = RiskData(
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
                    db.session.add(riskData)

            db.session.commit()

            print("Database initialized with data/sp500_risk_with_scores data")


init_db()

# with app.app_context():
#     print(Article.query.count())

# with app.app_context():
#     print(RiskData.query.count())

# with app.app_context():
#     print(
#         get_portfolio_risk_types(
#             ["XOM", "CVX", "COP", "EOG", "SLB", "HAL"],
#             top_k=5,
#         )
#     )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
