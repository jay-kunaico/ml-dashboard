import type React from "react";

interface TrainingDataProps {
	columns: string[];
	selectedColumns: string[];
	dataSources: string[];
	onSelect: (Selected: string[]) => void;
	label: string;
}

const TrainingDataSelector = ({
	columns,
	selectedColumns,
	// dataSources,
	onSelect,
	label,
}: TrainingDataProps) => {
	const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		const selectedOptions = Array.from(
			event.target.selectedOptions,
			(option) => option.value,
		);
		onSelect(selectedOptions);
	};

	return (
		<div className="mb-4">
			<label
				htmlFor="training-data-source"
				className="block mb-2 text-lg font-medium"
			>
				{label}
			</label>
			<select
				id="training-data-source"
				multiple
				className="w-[16rem] border rounded-lg focus:ring focus:ring-blue-200 overflow-y-scroll h-8 focus:h-auto appearance-none"
				value={selectedColumns}
				onChange={handleSelectChange}
			>
				{columns.map((column, index) => (
					<option
						key={column}
						value={column}
						className={`py-[6px] flex items-center justify-center ${
							index === 0 ? "rounded-t-md p-0" : ""
						} ${
							index === columns.length - 1 ? "rounded-b-md" : "" // Add rounded corners to the last option
						}`}
					>
						{column}
					</option>
				))}
			</select>
		</div>
	);
};
export default TrainingDataSelector;
