"""Convert PNG graphs to PDF format for portfolio submission"""

import os
from PIL import Image

def convert_png_to_pdf(input_dir="results", output_dir="Running an LLM"):
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    graph_files = [
        "accuracy_by_subject.png",
        "overall_accuracy.png",
        "timing_comparison.png",
        "error_analysis.png",
        "accuracy_heatmap.png"
    ]
    
    for graph_file in graph_files:
        input_path = os.path.join(input_dir, graph_file)
        if os.path.exists(input_path):
            output_path = os.path.join(output_dir, graph_file.replace(".png", ".pdf"))
            try:
                img = Image.open(input_path)
                img.save(output_path, "PDF", resolution=300.0)
                print(f"Converted: {graph_file} -> {output_path}")
            except Exception as e:
                print(f"Error converting {graph_file}: {e}")
        else:
            print(f"File not found: {input_path}")

if __name__ == "__main__":
    convert_png_to_pdf()

