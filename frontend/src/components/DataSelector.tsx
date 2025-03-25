import { datasetMetadata } from "../constants/datasetMetadata";

interface DataSelectorProps {
	dataSources: string[];
	selectedDataSource: string;
	onSelect: (dataSource: string) => void;
	label: string;
}

const DataSelector = ({
	dataSources,
	selectedDataSource,
	onSelect,
	label,
}: DataSelectorProps) => {
	const metadata =
		datasetMetadata[selectedDataSource as keyof typeof datasetMetadata] || {};

	return (
		<div className="mb-4">
			<label htmlFor="data-source" className="block mb-2 text-lg font-medium">
				{label}
			</label>
			<select
				id="data-source"
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200"
				value={selectedDataSource}
				onChange={(e) => onSelect(e.target.value)}
				title={
					selectedDataSource
						? `Description: ${metadata.description || "No description available"}\n\nBest Features: ${metadata.bestFeatures?.join(", ") || "N/A"}\n\nTarget Field: ${metadata.targetField || "N/A"}\n\nModels: ${metadata.models?.join(", ") || "N/A"}\n\n`
						: "Select a dataset to see details"
				}
			>
				<option value="" disabled>
					Select a data source
				</option>
				{dataSources.map((source) => (
					<option key={source} value={source}>
						{source}
					</option>
				))}
			</select>
		</div>
	);
};
export default DataSelector;
