from flask import Flask, jsonify
from database import db, AnalysisResult

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/api/code-quality-trends', methods=['GET'])
def get_trends():
    try:
        results = AnalysisResult.query.order_by(AnalysisResult.date.desc()).limit(30).all()
        return jsonify([{
            "date": r.date,
            "issueCount": r.issue_count
        } for r in results])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)