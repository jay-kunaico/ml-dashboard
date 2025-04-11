import type { ResultType } from "../types/types";

const BASE_URL = import.meta.env.VITE_API_URL;

export const fetchData = async (url: string) => {
	try {
		const response = await fetch(`${BASE_URL}/load_data`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				filename: url,
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
	algorithm: string,
	url: string,
): Promise<ResultType> => {

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
				filename: url,
			}),
		});
		if (!response.ok) {
			let errorMessage = await response.text();
			if (errorMessage.includes('base_score must be in (0,1)')) {
				errorMessage = 'The base score must be in (0,1) for logistic loss.';
			}
			else if (errorMessage.includes(':')) {
				const parts = errorMessage.split(':');
				errorMessage = parts[1].trim();
			}
			errorMessage = errorMessage.replace(/["'{}[\]()]/g, '');
			const truncatedErrorMessage = errorMessage.length > 100 ? `${errorMessage.substring(0, 200)}...` : errorMessage;
			throw new Error(truncatedErrorMessage);
		}
		const data: ResultType = await response.json();
		return data;
	} catch (error) {
		console.error('Error:', error);
		throw error;
	}
};