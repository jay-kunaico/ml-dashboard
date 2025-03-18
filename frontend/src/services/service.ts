const BASE_URL = 'http://127.0.0.1:5000';

export interface AlgorithmResponse {
	model: string;
	size: number;
	results: {
		mse: number;
		r2: number;
		f1_score: number;
		accuracy: number;
	};
	predictions: string[];
	dataframe: Record<string, string>[]; // Array of objects representing the updated dataframe
}
export const fetchData = async (url: string, mode: string) => {
	try {
		const response = await fetch(`${BASE_URL}/load_data`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				filename: url,
				mode: mode
			}),
		});

		if (!response.ok) {
			throw new Error('Network response was not ok');
		}

		return await response.json();
	} catch (error) {
		console.error('Error:', error);
		throw error;
	}
};

export const runAlgorithm = async (
	trainColumns: string[],
	targetColumn: string,
	algorithm: string
): Promise<AlgorithmResponse> => {
	console.log('trainColumns:', trainColumns);
	console.log('targetColumn:', targetColumn);
	console.log('algorithm:', algorithm);

	try {
		const response = await fetch(`${BASE_URL}/run_algorithm`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				trainColumns: trainColumns,
				targetColumn: targetColumn,
				algorithm: algorithm,
			}),
		});
		if (!response.ok) {
			throw new Error(`Network response was not ok: ${response.statusText}`);
		}
		const data: AlgorithmResponse = await response.json();
		return data;
	} catch (error) {
		console.error('Error:', error);
		throw error;
	}
};