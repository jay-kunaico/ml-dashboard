import React, { useState } from "react";

interface AccordionItem {
	title: string;
	content: string;
}

interface AccordionProps {
	items: AccordionItem[];
}

const Accordion = ({ items }: AccordionProps) => {
	const [openIndex, setOpenIndex] = useState<number | null>(null);

	const handleToggle = (index: number) => {
		setOpenIndex(openIndex === index ? null : index); // Toggle the accordion section
	};

	return (
		<div className="accordion">
			{items.map((item, index) => (
				<div key={item.title} className="mb-2 border rounded-lg">
					<h3>
						<button
							type="button"
							className="flex items-center w-full h-10 text-left p-4 bg-gray-200 hover:bg-gray-300 focus:outline-none focus:ring focus:ring-blue-500"
							aria-expanded={openIndex === index}
							aria-controls={`accordion-content-${index}`}
							onClick={() => handleToggle(index)}
						>
							{item.title}
						</button>
					</h3>
					<section
						id={`accordion-content-${index}`}
						className={`p-4 bg-white ${
							openIndex === index ? "block" : "hidden"
						}`}
						aria-labelledby={`accordion-header-${index}`}
					>
						{item.content}
					</section>
				</div>
			))}
		</div>
	);
};

export default Accordion;
