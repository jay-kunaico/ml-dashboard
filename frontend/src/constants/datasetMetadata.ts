export const datasetMetadata = {
	"cc_fraud.csv": {
		bestFeatures: ["Amount", "Location", "MerchantID", "TransactionType"],
		targetField: "IsFraud",
		models: ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost", "K-Nerest Neighbors"],
		description: "This dataset is used for classification whether a transaction is fraudulent or not.",
	},
	"credit_risk.csv": {
		bestFeatures: ["Age", "Checking_account", "credit_amount,", "duration", "houseing", "job", "purpose", "savings_account", "sex"],
		targetField: "Leave Blank",
		models: ["Logistic Regression", "K-Means", "Random Forest", "Decision Tree"],
		description: "This dataset is used for classification whether the credit risk is high or not.",
	},
	"credit_score.csv": {
		bestFeatures: ["Age", "Bill_amt", "Education", "Limit_bal", "pay", "pay_amt"],
		targetField: "Leave Blank",
		models: ["Random Forest Regressor", "Decision Tree Regressor", "K-Nearest Neighbors Regressor"],
		description: "This dataset is a regression dataset for credit scoring.",
	},
	"customer_churn.csv": {
		bestFeatures: ["Country", "Favorite_Genre", "Last_login", "Subscrption_type", "Watch_time_hours"],
		targetField: "Leave Blank",
		models: ["K-Means", "Random Forest", "Decision Tree"],
		description: "This dataset is used for classification whether a customer will leave or not.",
	},
};