import React from "react";
import "./index.css";
import { useEffect, useState } from "react";
import DataSelector from "./components/DataSelector";
import AlgorithmSelector from "./components/AlgorithmSelector";
import TrainingDataSelector from "./components/TrainingData";
import TargetDataSelector from "./components/TargetData";
import Results from "./components/Results";
import { dataSources, algorithms } from "./constants/constants";
import { fetchData, runAlgorithm } from "./services/service";
import Spinner from "./components/Spinner";
import type { ResultType } from "./types/types";
import HeadlessModal from "./components/headless_modal";
import Accordion from "./components/accordian";
import { terminology } from "./constants/terminology";
import { DOCS_URL } from "./constants/constants";

const DataTable = React.memo(
	({
		data,
		columnKeys,
	}: { data: Record<string, string>[]; columnKeys: string[] }) => {
		const [sortedData, setSortedData] = useState(data);
		const [sortConfig, setSortConfig] = useState<{
			key: string;
			direction: "asc" | "desc";
		} | null>(null);

		useEffect(() => {
			setSortedData(data);
		}, [data]);

		// Handle sorting when a column header is clicked
		const handleSort = (key: string) => {
			let direction: "asc" | "desc" = "asc";
			if (
				sortConfig &&
				sortConfig.key === key &&
				sortConfig.direction === "asc"
			) {
				direction = "desc";
			}

			const sorted = [...data].sort((a, b) => {
				if (a[key] === null || b[key] === null) return 0; // Handle null values
				if (typeof a[key] === "number" && typeof b[key] === "number") {
					return direction === "asc" ? a[key] - b[key] : b[key] - a[key];
				}
				return direction === "asc"
					? String(a[key]).localeCompare(String(b[key]))
					: String(b[key]).localeCompare(String(a[key]));
			});

			setSortedData(sorted);
			setSortConfig({ key, direction });
		};

		return (
			<div className="overflow-x-auto w-full rounded-xl border border-gray-400 bg-gray-300 mt-6 max-h-[calc(100vh-350px)] overflow-y-auto">
				<table className="w-full border-collapse">
					<thead className="sticky top-0 bg-gray-200 z-10">
						<tr>
							{columnKeys.map((col) => (
								<th key={col} className="border border-gray-500 bg-white p-2">
									<button
										type="button"
										onClick={() => handleSort(col)}
										className="w-full text-left focus:outline-none focus:ring focus:ring-blue-500 focus:ring-offset-2 rounded"
									>
										<div className="flex items-center justify-between">
											<span>{col}</span>
											{sortConfig?.key === col && (
												<span>
													{sortConfig.direction === "asc" ? "🔼" : "🔽"}
												</span>
											)}
										</div>
									</button>
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{sortedData.map((row, rowIndex) => (
							<tr key={row.id || rowIndex}>
								{columnKeys.map((col) => (
									<td key={col} className="border border-gray-500 p-2">
										{row[col]}
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
		);
	},
);

function App() {
	const [selectedDataSource, setSelectedDataSource] = useState("");
	const [selectedAlgorithm, setSelectedAlgorithm] = useState("");
	const [data, setData] = useState<Record<string, string>[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [loadingResults, setLoadingResults] = useState(false);

	const [response, setResponse] = useState<ResultType | null>(null);
	const [columns, setColumns] = useState<string[]>([]);
	const [firstRow, setFirstRow] = useState<string[]>([]);
	const [trainingColumns, setTrainingColumns] = useState<string[]>([]);
	const [targetColumn, setTargetColumn] = useState<string>("");
	const columnKeys = data.length > 0 ? Object.keys(data[0]) : [];
	const [isModalOpen, setIsModalOpen] = useState(false);

	const openModal = () => setIsModalOpen(true);
	const closeModal = () => setIsModalOpen(false);

	const handleLoadData = async () => {
		if (!selectedDataSource) {
			setError("Please select a data source");
			return;
		}
		setLoading(true);
		setError(null);

		try {
			const response = await fetchData(selectedDataSource);

			setData(response);
			setError(null);
			setLoading(false);
		} catch (error) {
			console.error("Error:", error);
		}
	};

	useEffect(() => {
		const timeout = setTimeout(() => {
			if (selectedDataSource) {
				fetchData(selectedDataSource).then((data) => {
					setColumns(Object.keys(data[0]));
					setFirstRow(Object.values(data[0]));
				});
			}
		}, 300); // Debounce by 300ms

		return () => clearTimeout(timeout);
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
				selectedDataSource,
			);

			// Check for error in the response
			if ("error" in response && response.error) {
				setError(response.error);
				setLoadingResults(false);
				return;
			}
			setResponse(response);

			if (response.dataframe) {
				setData(response.dataframe);
			}

			setError(null);
			setLoadingResults(false);
		} catch (error) {
			console.error("Eerror:", error);

			if (error instanceof Error) {
				setError(error.message);
				console.log("eError", error);
			} else {
				setError("An unknown error occurred");
			}
			setLoadingResults(false);
		}
	};

	return (
		<main className="min-h-screen flex flex-col items-center bg-gray-100 p-4">
			<h1 className="mb-6">
				<button
					type="button"
					className="text-3xl font-bold"
					title="Click to open a modal for information and terminology"
					onClick={openModal}
					aria-expanded={isModalOpen}
					aria-controls="ml-terminology-modal"
					tabIndex={0}
					onKeyDown={(e) => {
						if (e.key === "Enter" || e.key === " ") openModal();
					}}
				>
					Machine Learning Playground
				</button>
				<a
					href={DOCS_URL}
					target="_blank"
					rel="noopener noreferrer"
					className="ml-2 text-blue-600 hover:text-blue-800"
					title="Open help documentation"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						className="h-6 w-6 inline-block"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<title>Help icon</title>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
						/>
					</svg>
				</a>
			</h1>
			<HeadlessModal
				isOpen={isModalOpen}
				onClose={closeModal}
				title="About This Application"
			>
				<p>
					This app implements a Flask-based backend for running machine learning
					algorithms on a dataset. It provides endpoints for loading data,
					preprocessing it, and applying machine learning algorithms.
				</p>
				<p className="mt-4">
					Use the interface to select a dataset, choose features and a target
					variable, if applicable, and run a machine learning model to see the
					results.
				</p>
				<Accordion items={terminology} />
			</HeadlessModal>
			<div className="flex flex-row space-x-4 mb-6">
				<DataSelector
					dataSources={dataSources}
					selectedDataSource={selectedDataSource}
					onSelect={setSelectedDataSource}
					label="Choose Dataset"
				/>
				<TrainingDataSelector
					dataSources={firstRow}
					columns={columns}
					selectedColumns={trainingColumns}
					onSelect={setTrainingColumns}
					label="Select Features"
				/>
				<TargetDataSelector
					columns={columns}
					targetColumn={targetColumn}
					onSelect={setTargetColumn}
					label="Select Target"
				/>
				<AlgorithmSelector
					algorithms={[
						...algorithms.classifiers,
						...algorithms.regressors,
						...algorithms.clustering,
					]}
					selectedAlgorithm={selectedAlgorithm}
					onSelect={setSelectedAlgorithm}
					label="Choose Model"
				/>

				<Results
					response={response}
					loadingResults={loadingResults}
					error={error}
				/>
			</div>
			{/* {error && <div className="text-red-500 text-lg mb-4">{error}</div>} */}
			<div className="h-6 space-x-4"> {loading && <Spinner />}</div>

			<div className="flex flex-row space-x-4">
				<button
					type="button"
					onClick={handleLoadData}
					className={`px-6 py-2 rounded-lg transition ${
						selectedDataSource
							? "bg-blue-300 text-gray-800 hover:bg-blue-400"
							: "bg-gray-200 text-gray-600 cursor-not-allowed"
					}`}
					disabled={!selectedDataSource}
				>
					Load Dataset
				</button>

				{/* Primary Action: Run Model */}
				<button
					type="button"
					onClick={handleRunAlgorithm}
					className={`px-6 py-2 rounded-lg transition ${
						selectedDataSource && selectedAlgorithm
							? "bg-blue-600 text-white hover:bg-blue-800"
							: "bg-gray-200 text-gray-600 cursor-not-allowed"
					}`}
					disabled={!selectedDataSource || !selectedAlgorithm}
				>
					Run Model
				</button>
			</div>
			<DataTable data={data} columnKeys={columnKeys} />
		</main>
	);
}

export default App;
