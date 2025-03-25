export const dataSources = [
	"customers.csv",
	"customer_churn.csv",
	"credit_score.csv",
	"credit_risk.csv",
	"cc_fraud.csv",
];

// Algorithms grouped by type
export const algorithms = {
	classifiers: [
		"Logistic Regression",
		"Decision Tree",
		"K-Nearest Neighbors",
		"Random Forest",
		"XGBoost",
	],
	regressors: [
		"Decision Tree Regressor",
		"K-Nearest Neighbors Regressor",
		"Random Forest Regressor",
	],
	clustering: ["K-Means", "DBSCAN", "Agglomerative Clustering", "Birch"],
};