# Notebook Runner

Structured and reusable framework for executing Jupyter notebooks programmatically.

## Features

- **Sequential Execution**: Executes all cells in order, preserving state
- **Dependency Management**: Maintains shared namespace across cells
- **Error Handling**: Comprehensive error tracking and reporting
- **Figure Generation**: Automatically tracks and reports generated figures
- **Statistics**: Detailed execution statistics and timing
- **CLI Interface**: Easy-to-use command-line interface
- **Reusable API**: Can be imported and used programmatically

## Usage

### Command Line

Execute all tutorial notebooks:
```bash
python notebook_runner.py --all
```

Execute specific notebooks:
```bash
python notebook_runner.py tutorials/basic_tabular.ipynb tutorials/multimodal_clusters.ipynb
```

Generate execution report:
```bash
python notebook_runner.py --all --report execution_report.txt
```

Stop on first error:
```bash
python notebook_runner.py --all --stop-on-error
```

### Python API

```python
from notebook_runner import NotebookRunner

# Create runner
runner = NotebookRunner()

# Execute single notebook
result = runner.run_notebook('tutorials/basic_tabular.ipynb')
print(f"Generated {len(result.figures_generated)} figures")

# Execute multiple notebooks
results = runner.run_multiple_notebooks([
    'tutorials/basic_tabular.ipynb',
    'tutorials/multimodal_clusters.ipynb',
])

# Generate report
report = runner.generate_report(results, output_file='report.txt')
print(report)
```

## Architecture

### Classes

- **`NotebookRunner`**: Main execution framework
- **`NotebookExecutionResult`**: Results for a complete notebook
- **`CellExecutionResult`**: Results for a single cell

### Key Methods

- `run_notebook()`: Execute a single notebook
- `run_multiple_notebooks()`: Execute multiple notebooks
- `generate_report()`: Generate execution report
- `_execute_cell()`: Execute a single cell (internal)
- `_setup_standard_imports()`: Setup execution environment

## Configuration

The runner automatically sets up:
- NumPy, Pandas, Matplotlib
- scikit-learn (datasets, preprocessing, models, metrics)
- UMAP (if available)
- dd-coresets library

All imports are available in the execution namespace.

## Error Handling

- Errors are caught and logged per cell
- Execution continues by default (use `--stop-on-error` to stop)
- Error types are tracked and summarized in reports
- Failed cells are marked but don't prevent other cells from running

## Figure Tracking

The runner automatically:
- Detects `plt.savefig()` calls in cells
- Verifies figures were actually created
- Tracks figure paths and sizes
- Reports all generated figures in execution report

## Reports

Execution reports include:
- Summary statistics (notebooks, cells, errors, figures, time)
- Per-notebook details
- Generated figures list with sizes
- Error summary by type
- Overall statistics

## Examples

### Basic Usage
```bash
# Execute all notebooks
python notebook_runner.py --all

# Execute specific notebook
python notebook_runner.py tutorials/basic_tabular.ipynb
```

### Advanced Usage
```bash
# Generate detailed report
python notebook_runner.py --all --report detailed_report.txt

# Stop on first error (useful for debugging)
python notebook_runner.py --all --stop-on-error

# Reset namespace between notebooks
python notebook_runner.py --all --reset-namespace
```

### Programmatic Usage
```python
from notebook_runner import NotebookRunner

runner = NotebookRunner(stop_on_error=True)

# Execute with custom configuration
results = runner.run_notebook(
    'tutorials/basic_tabular.ipynb',
    reset_namespace=False
)

# Check results
if results.success:
    print(f"✅ Generated {len(results.figures_generated)} figures")
else:
    print(f"❌ {results.errors} errors occurred")
```

## Integration

The runner can be integrated into:
- CI/CD pipelines
- Documentation build processes
- Testing frameworks
- Development workflows

## Dependencies

- Python 3.8+
- Jupyter notebook format (JSON)
- Standard data science libraries (auto-imported)
- dd-coresets library

## Notes

- The runner uses a shared namespace across all cells
- Cells are executed sequentially in order
- State is preserved between cells (variables, imports, etc.)
- Figures are saved to paths specified in `plt.savefig()` calls
- Execution time is tracked per cell and per notebook

