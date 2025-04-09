import React from "react";
import Spinner from "./Spinner";
import type { ResultType } from "../types/types";

interface ResultsProps {
	response: ResultType | null;
	loadingResults: boolean;
	error: string | null;
}

const Results = ({ response, loadingResults, error }: ResultsProps) => {
	return (
		<div className="mb-4">
			<label htmlFor="data-source" className="block mb-2 text-lg font-medium">
				Results:
			</label>
			<div className="w-[325px] p-2 border rounded-lg focus:ring focus:ring-blue-200 overflow-hidden h-auto max-w-[325px]">
				{error && <div className="text-red-500 text-lg mb-4">{error}</div>}
				{response ? (
					<>
						<div>Model: {response.model}</div>
						{response?.results?.message && (
							<div>{response.results?.message}</div>
						)}
						{/* TODO move tooltips to a constant */}
						{response?.results?.mse != null && (
							<div
								className="relative group"
								title="The average squared difference between the estimated values and the true value. The smaller the value, the better the model."
							>
								<a
									href="https://en.wikipedia.org/wiki/Mean_squared_error"
									target="_blank"
									rel="noreferrer"
								>
									Mean Squared Error: {response?.results?.mse?.toFixed(2)}
								</a>
							</div>
						)}
						{response?.results?.r2 != null && (
							<div
								className="relative group"
								title="R2 ranges from 0 to 1.  A value of 1 indicates that the model explains all of the variance in the target variable. A value of 0 indicates that the model does not explain any of the variance in the target variable. R2 can be negative if the model is worse than predicting the mean of the target variable."
							>
								<a
									href="https://en.wikipedia.org/wiki/Coefficient_of_determination"
									target="_blank"
									rel="noreferrer"
								>
									R-squared: {response?.results?.r2?.toFixed(2)}
								</a>
							</div>
						)}
						{response?.results?.f1_score != null && (
							<div
								className="relative group"
								title="The F1 score balances precision and recall, a good measure of a models accuracy. It ranges from 0 (worst) to 1 (best) and considers false positives and false negatives."
							>
								<a
									href="https://en.wikipedia.org/wiki/F-score"
									target="_blank"
									rel="noreferrer"
								>
									F1: {response?.results?.f1_score?.toFixed(2)}
								</a>
							</div>
						)}
						{response?.results?.accuracy != null && (
							<div title="Accuracy score is a classification metric that measures the fraction of correct predictions a model makes, calculated by dividing the number of correct predictions by the total number of predictions. Closer to 1 is better.">
								<a
									href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"
									target="_blank"
									rel="noreferrer"
								>
									Accuracy: {response?.results?.accuracy?.toFixed(2)}
								</a>
							</div>
						)}

						<div>Size: {response.size}</div>
					</>
				) : (
					loadingResults && <Spinner />
				)}
			</div>
		</div>
	);
};
export default React.memo(Results);
