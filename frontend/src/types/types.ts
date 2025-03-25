export interface ResultType {
	dataframe?: Record<string, string>[] | null;
	model: string;
	size: number;
	results: {
		mse: number;
		r2: number;
		f1_score: number;
		accuracy: number;
		message: string;
	};
	predictions?: string[];
}