export const datasetMetadata = {
	"credit_card_fraud.csv": {
		bestFeatures: ["distance_from_home", "distance_from_last_transaction", "online_order", "ratio_to_median_purchase_price", "repeat_retailier", "used_chip", "used_pin_number"],
		targetField: "fraud",
		models: ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost", "K-Nerest Neighbors"],
		description: "This is a combination of cc_fraud and card_transdata datasets.",
	},
	"credit_risk.csv": {
		bestFeatures: ["Age", "Checking_account", "credit_amount,", "duration", "houseing", "job", "purpose", "savings_account", "sex"],
		targetField: "Leave Blank",
		models: ["K-Means", "DBSCAN", "Agglomerative Clustering"],
		description: "This dataset is used for classification whether the credit risk is high or not.",
	},
	"credit_score.csv": {
		bestFeatures: ["Age", "Bill_amt", "Education", "Limit_bal", "pay", "pay_amt"],
		targetField: "Leave Blank",
		models: ["K-Means"],
		description: "This dataset is a regression dataset for credit scoring.",
	},
	"customer_churn.csv": {
		bestFeatures: ["Country", "Favorite_Genre", "Last_login", "Subscrption_type", "Watch_time_hours"],
		targetField: "Leave Blank",
		models: ["K-Means"],
		description: "This dataset is used for classification whether a customer will leave or not.",
	},
};