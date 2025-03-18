import { Component } from "react";

export default class Spinner extends Component {
	render() {
		return (
			<div className="flex justify-center items-center">
				<div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-gray-700" />
			</div>
		);
	}
}
