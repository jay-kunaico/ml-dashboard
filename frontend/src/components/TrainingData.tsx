import type React from "react";

interface TrainingDataProps {
	columns: string[];
	selectedColumns: string[];
	dataSources: string[];
	onSelect: (Selected: string[]) => void;
}

const TrainingDataSelector = ({
	columns,
	selectedColumns,
	// dataSources,
	onSelect,
}: TrainingDataProps) => {
	const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		const selectedOptions = Array.from(
			event.target.selectedOptions,
			(option) => option.value,
		);
		onSelect(selectedOptions);
	};

	// below was commented out to allow for all columns to be selected
	// originally, it was only allowing numeric columns to be selected
	// const trainingColumns = columns.filter((column, index) => {
	// 	const value = dataSources[index];
	// 	return !Number.isNaN(Number.parseFloat(value)) && value !== "";
	// });

	return (
		<div className="mb-4">
			<label
				htmlFor="training-data-source"
				className="block mb-2 text-lg font-medium"
			>
				Select Training Data:
			</label>
			<select
				id="training-data-source"
				multiple
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200 overflow-y-scroll h-10 focus:h-auto"
				value={selectedColumns}
				onChange={handleSelectChange}
			>
				{/* <option value="" disabled>
					Select training data
				</option> */}
				{/* {trainingColumns.map((column) => (
					<option key={column} value={column}>
						{column}
					</option>
				))} */}
				{columns.map((column) => (
					<option key={column} value={column}>
						{column}
					</option>
				))}
			</select>
		</div>
	);
};
export default TrainingDataSelector;
