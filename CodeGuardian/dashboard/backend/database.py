from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, nullable=False)
    issue_count = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<AnalysisResult {self.date}: {self.issue_count} issues>"