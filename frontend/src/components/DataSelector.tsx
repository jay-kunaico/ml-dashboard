import React from "react";

interface DataSelectorProps {
	dataSources: string[];
	selectedDataSource: string;
	onSelect: (dataSource: string) => void;
}

const DataSelector = ({
	dataSources,
	selectedDataSource,
	onSelect,
}: DataSelectorProps) => {
	return (
		<div className="mb-4">
			<label htmlFor="data-source" className="block mb-2 text-lg font-medium">
				Select Data Source:
			</label>
			<select
				id="data-source"
				className="w-full p-2 border rounded-lg focus:ring focus:ring-blue-200"
				value={selectedDataSource}
				onChange={(e) => onSelect(e.target.value)}
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
