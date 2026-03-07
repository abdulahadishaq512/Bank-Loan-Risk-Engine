# Credit-Risk-Engine
A machine learning pipeline for credit risk assessment that utilizes SQL for data management, Random Forest for default prediction, and a Streamlit-based web interface for real-time risk evaluation.
Features
• SQL Integration: Normalized data management using sqlite3.

• Feature Engineering: Created custom domain-specific metrics like Loan_to_Income_Ratio to improve model predictive power.

• Predictive Modeling: Utilized RandomForestClassifier to identify high-risk loan profiles.

• Interactive Dashboard: Deployed a web-based interface using streamlit for live risk inference.

Tech Stack
• Language: Python 3.12

• Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, streamlit, joblib

• Data Management: sqlite3, Kaggle API

Project Structure
Plaintext
Credit-Risk-Engine/
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
├── data/               # Raw and processed datasets
└── src/                # Modular scripts for training and ETL
How to Run
1. Clone the repository:

Bash
git clone https://github.com/yourusername/Credit-Risk-Engine.git
cd Credit-Risk-Engine
2. Install dependencies:

Bash
pip install -r requirements.txt
3. Run the application:

Bash
streamlit run app.py
