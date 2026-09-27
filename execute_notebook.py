"""Execute and save the notebook from a clean Python kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "solution.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
def record_progress(cell, cell_index, **kwargs):
    print(f"Completed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}},
                        on_cell_executed=record_progress)
client.execute()
nbformat.write(notebook, path)
print("Executed and saved:", path)
