import { modelMetadata } from "../constants/modelMetadata";

interface AlgorithmSelectorProps {
	algorithms: string[];
	selectedAlgorithm: string;
	onSelect: (algorithm: string) => void;
	label: string;
}

const AlgorithmSelector = ({
	algorithms,
	selectedAlgorithm,
	onSelect,
	label,
}: AlgorithmSelectorProps) => {
	const metadata =
		modelMetadata[selectedAlgorithm as keyof typeof modelMetadata] || {};
	return (
		<div className="mb-4">
			<label
				htmlFor="algorithm-source"
				className="block mb-2 text-lg font-medium"
			>
				{label}
			</label>
			<select
				id="algorithm-source"
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200"
				value={selectedAlgorithm}
				onChange={(e) => onSelect(e.target.value)}
				title={
					selectedAlgorithm ? `${metadata}` : "Select a model to see details"
				}
			>
				<option value="" disabled>
					Select an algorithm
				</option>
				{algorithms.map((source) => (
					<option key={source} value={source}>
						{source}
					</option>
				))}
			</select>
		</div>
	);
};
export default AlgorithmSelector;
