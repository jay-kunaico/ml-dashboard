import Spinner from "./Spinner";

interface Result {
	model: string;
	size: number;
	results: {
		mse: number;
		r2: number;
		f1_score: number;
		accuracy: number;
	};
	predictions?: string[];
	dataFrame?: Record<string, string>[];
}

interface ResultsProps {
	response: Result;
	loadingResults: boolean;
}

const Results = ({ response, loadingResults }: ResultsProps) => {
	return (
		<div className="mb-4">
			<label htmlFor="data-source" className="block mb-2 text-lg font-medium">
				Results:
			</label>
			<div className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200 overflow-hidden h-auto ">
				{response ? (
					<>
						<div>Model: {response.model}</div>
						{response?.results?.mse && (
							<div
								className="relative group"
								title="the average squared difference between the estimated values and the true value. The smaller the value, the better the model."
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
						{response?.results?.r2 && (
							<div
								className="relative group"
								title="the coefficient of determination, denoted R2 or r2 and pronounced R squared, is the proportion of the variation in the dependent variable that is predictable from the independent variable(s)."
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
						{response?.results?.f1_score && (
							<div
								className="relative group"
								title="The F1 score is the harmonic mean of the precision and recall. It thus symmetrically represents both precision and recall in one metric.."
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
						{response?.results?.accuracy && (
							<div title="accuracy score is a classification metric that measures the fraction of correct predictions a model makes, calculated by dividing the number of correct predictions by the total number of predictions.">
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
export default Results;
