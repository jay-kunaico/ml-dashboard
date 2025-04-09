import type React from "react";

interface ModalProps {
	isOpen: boolean;
	onClose: () => void;
	children: React.ReactNode;
}

const Modal = ({ isOpen, onClose, children }: ModalProps) => {
	if (!isOpen) return null;

	return (
		<div
			className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
			id="ml-terminology-modal"
		>
			<div className="bg-white rounded-lg shadow-lg p-6 w-[90%] max-w-lg">
				<button
					type="button"
					onClick={onClose}
					className="right-2 text-gray-500 hover:text-gray-800"
				>
					&times;
				</button>
				{children}
			</div>
		</div>
	);
};

export default Modal;
