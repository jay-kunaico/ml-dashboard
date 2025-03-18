import React from "react";

interface AlgorithmSelectorProps {
	algorithms: string[];
	selectedAlgorithm: string;
	onSelect: (algorithm: string) => void;
}

const AlgorithmSelector = ({
	algorithms,
	selectedAlgorithm,
	onSelect,
}: AlgorithmSelectorProps) => {
	return (
		<div className="mb-4">
			<label
				htmlFor="algorithm-source"
				className="block mb-2 text-lg font-medium"
			>
				Select Algorithm:
			</label>
			<select
				id="algorithm-source"
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200"
				value={selectedAlgorithm}
				onChange={(e) => onSelect(e.target.value)}
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
