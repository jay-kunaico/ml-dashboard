import type { ResultType } from "../types/types";

const BASE_URL = import.meta.env.VITE_API_URL;

export const fetchData = async (url: string) => {
	try {
		const response = await fetch(`${BASE_URL}/run-preview`, {
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
		const data = await response.json();

		// Handle Lambda API Gateway response (stringified body)
		if (data.body && typeof data.body === 'string') {
			return JSON.parse(data.body);
		}

		// Handle Flask API response (direct JSON)
		return data;
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
			let errorMessage = '';
			try {
				const errorData = await response.json();
				// Lambda: error is in stringified JSON in body
				if (errorData.body && typeof errorData.body === 'string') {
					const parsedBody = JSON.parse(errorData.body);
					errorMessage = parsedBody.error || JSON.stringify(parsedBody);
				} else {
					// Flask: error is direct JSON
					errorMessage = errorData.error || JSON.stringify(errorData);
				}
				if (!errorMessage || errorMessage === "''") {
					errorMessage = "An unknown error occurred. Please check the backend logs.";
				}
				else if (errorMessage.includes('base_score must be in (0,1)')) {
					errorMessage = 'The base score must be in (0,1) for logistic loss.';
				}
				// Comment out below to see full error message
				else if (errorMessage.includes(':')) {
					const parts = errorMessage.split(':');
					errorMessage = parts[1].trim();
				}
			} catch {
				errorMessage = await response.text();
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