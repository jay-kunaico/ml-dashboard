import "./index.css";
import { useEffect, useState } from "react";
import DataSelector from "./components/DataSelector";
import AlgorithmSelector from "./components/AlgorithmSelector";
import TrainingDataSelector from "./components/TrainingData";
import TargetDataSelector from "./components/TargetData";
import Results from "./components/Results";
import { fetchData, runAlgorithm } from "./services/service";
import Spinner from "./components/Spinner";

const dataSources = [
	"cc_fraud.csv",
	"customers.csv",
	"customer_churn.csv",
	"credit_score.csv",
	"credit_risk.csv",
];
const algorithms = [
	"Linear Regression",
	"Decision Tree",
	"K-Nearest Neighbors",
	"Logistic Regression",
	"XGBoost",
	"Random Forest",
];

function App() {
	const [selectedDataSource, setSelectedDataSource] = useState("");
	const [selectedAlgorithm, setSelectedAlgorithm] = useState("");
	const [data, setData] = useState<Record<string, string>[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [loadingResults, setLoadingResults] = useState(false);
	const [response, setResponse] = useState<any>(null);
	const [columns, setColumns] = useState<string[]>([]);
	const [firstRow, setFirstRow] = useState<string[]>([]);
	const [trainingColumns, setTrainingColumns] = useState<string[]>([]);
	const [targetColumn, setTargetColumn] = useState<string>("");

	const handleLoadData = async (mode: "preview" | "full") => {
		if (!selectedDataSource) {
			setError("Please select a data source");
			return;
		}
		setLoading(true);
		setError(null);

		try {
			const response = await fetchData(selectedDataSource, mode);
			setData(response);
			setError(null);
			setLoading(false);
		} catch (error) {
			console.error("Error:", error);
		}
	};

	useEffect(() => {
		if (selectedDataSource) {
			fetchData(selectedDataSource, "preview").then((data) => {
				setColumns(Object.keys(data[0]));
				setFirstRow(Object.values(data[0]));
			});
		}
	}, [selectedDataSource]);

	const handleRunAlgorithm = async () => {
		if (!selectedAlgorithm) {
			setError("Please select an algorithm");
			return;
		}
		setLoadingResults(true);
		setError(null);
		setResponse(null);

		try {
			const response = await runAlgorithm(
				trainingColumns,
				targetColumn,
				selectedAlgorithm,
			);
			setResponse(response);

			if (response.dataframe) {
				setData(response.dataframe);
			}

			setError(null);
			setLoadingResults(false);
		} catch (error) {
			console.error("Error:", error);
		}
	};

	return (
		<div className="min-h-screen flex flex-col items-center bg-gray-100 p-4">
			<div className="mb-6">
				<h1 className="text-3xl font-bold">Machine Learning Playground</h1>
			</div>
			<div className="flex flex-row space-x-4 mb-6">
				<DataSelector
					dataSources={dataSources}
					selectedDataSource={selectedDataSource}
					onSelect={setSelectedDataSource}
				/>
				<AlgorithmSelector
					algorithms={algorithms}
					selectedAlgorithm={selectedAlgorithm}
					onSelect={setSelectedAlgorithm}
				/>
				<TrainingDataSelector
					dataSources={firstRow}
					columns={columns}
					selectedColumns={trainingColumns}
					onSelect={setTrainingColumns}
				/>
				<TargetDataSelector
					columns={columns}
					targetColumn={targetColumn}
					onSelect={setTargetColumn}
				/>
				<Results response={response} loadingResults={loadingResults} />
			</div>
			{/* {error && <div className="text-red-500 text-lg mb-4">{error}</div>} */}
			<div className="h-6 space-x-4"> {loading && <Spinner />}</div>

			<div className="flex flex-row space-x-4">
				<button
					type="button"
					onClick={handleLoadData.bind(null, "preview")}
					className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition"
					disabled={!selectedDataSource || loading}
				>
					Get Preview
				</button>
				<button
					type="button"
					onClick={handleLoadData.bind(null, "full")}
					className="bg-purple-500 text-white px-6 py-2 rounded-lg hover:bg-purple-600 transition"
					disabled={!selectedDataSource}
				>
					Get Data
				</button>
				<button
					type="button"
					onClick={handleRunAlgorithm}
					className="bg-green-500 text-white px-6 py-2 rounded-lg hover:bg-green-600 transition"
					disabled={!selectedDataSource || !selectedAlgorithm}
				>
					Run Algorithm
				</button>
			</div>

			<table className="table-auto w-full mt-4 border-collapse border-2 border-gray-600">
				<thead>
					<tr>
						{data.length > 0 &&
							Object.keys(data[0]).map((col) => (
								<th key={col} className="border border-gray-500 bg-slate-100">
									{col}
								</th>
							))}
					</tr>
				</thead>
				<tbody>
					{data.map((row) => (
						<tr key={row.id || JSON.stringify(row)}>
							{Object.keys(row).map((col) => (
								<td key={col} className="border border-gray-500">
									{row[col]}
								</td>
							))}
						</tr>
					))}
				</tbody>
			</table>
		</div>
	);
}

export default App;
