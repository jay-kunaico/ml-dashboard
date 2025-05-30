export const dataSources = [
	// "customers.csv",
	"customer_churn.csv",
	"credit_score.csv",
	"credit_risk.csv",
	"credit_card_fraud.csv",
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
	clustering: ["K-Means", "DBSCAN", "Agglomerative Clustering",],
};

export const DOCS_URL = "https://docs.google.com/document/d/1nAcNANwVH_nIRnHIKsBswwe2V9ozaD-mKQFrSShrj7Q/edit?usp=sharing";