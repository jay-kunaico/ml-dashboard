import type React from "react";

interface TargetDataProps {
	columns: string[];
	targetColumn: string;
	onSelect: (Selected: string) => void;
}

const TargetDataSelector = ({
	columns,
	targetColumn,
	onSelect,
}: TargetDataProps) => {
	const handleSelectChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
		const selectedOption = event.target.value;
		onSelect(selectedOption);
	};

	return (
		<div className="mb-4">
			<label
				htmlFor="target-data-source"
				className="block mb-2 text-lg font-medium"
			>
				Select target field:
			</label>
			<select
				id="target-data-source"
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200 overflow-hidden h-10 focus:h-auto"
				value={targetColumn}
				onChange={handleSelectChange}
			>
				<option value="">Select target field</option>
				{columns.map((column) => (
					<option key={column} value={column}>
						{column}
					</option>
				))}
			</select>
		</div>
	);
};
export default TargetDataSelector;
