#!/usr/bin/env python3
"""
Notebook Runner - Structured and reusable notebook execution framework.

This module provides a robust framework for executing Jupyter notebooks
programmatically, ensuring all cells run in order and all outputs (including
figures) are generated correctly.

Usage:
    from notebook_runner import NotebookRunner
    
    runner = NotebookRunner()
    results = runner.run_notebook('tutorials/basic_tabular.ipynb')
    runner.generate_report(results)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CellExecutionResult:
    """Result of executing a single notebook cell."""
    cell_index: int
    cell_type: str
    executed: bool
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    figures_generated: List[str] = field(default_factory=list)
    execution_time: float = 0.0


@dataclass
class NotebookExecutionResult:
    """Result of executing a complete notebook."""
    notebook_path: str
    success: bool
    cells_executed: int
    cells_total: int
    errors: int
    figures_generated: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    cell_results: List[CellExecutionResult] = field(default_factory=list)
    error_summary: Dict[str, int] = field(default_factory=dict)


class NotebookRunner:
    """
    Robust notebook execution framework.
    
    Handles:
    - Sequential cell execution with state preservation
    - Dependency management
    - Error handling and reporting
    - Figure generation tracking
    - Execution statistics
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        setup_imports: bool = True,
        stop_on_error: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Initialize notebook runner.
        
        Args:
            base_path: Base path for resolving relative imports
            setup_imports: Whether to setup standard imports automatically
            stop_on_error: Whether to stop execution on first error
            timeout: Maximum execution time per cell (seconds)
        """
        self.base_path = base_path or os.getcwd()
        self.setup_imports = setup_imports
        self.stop_on_error = stop_on_error
        self.timeout = timeout
        
        # Execution namespace (shared across all cells)
        self.namespace = {}
        
        # Statistics
        self.stats = {
            'notebooks_run': 0,
            'cells_executed': 0,
            'errors': 0,
            'figures_generated': 0,
        }
        
        if self.setup_imports:
            self._setup_standard_imports()
    
    def _setup_standard_imports(self):
        """Setup standard imports in execution namespace."""
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.datasets import make_blobs, make_classification, make_moons
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
        
        # Try to import umap
        try:
            import umap
            HAS_UMAP = True
        except ImportError:
            HAS_UMAP = False
            umap = None
        
        # Import dd-coresets
        sys.path.insert(0, str(Path(self.base_path).parent.parent))
        from dd_coresets import (
            fit_ddc_coreset,
            fit_random_coreset,
            fit_stratified_coreset,
            fit_ddc_coreset_by_label,
        )
        
        # Add to namespace
        self.namespace.update({
            '__name__': '__main__',
            '__builtins__': __builtins__,
            'np': np,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'make_blobs': make_blobs,
            'make_classification': make_classification,
            'make_moons': make_moons,
            'StandardScaler': StandardScaler,
            'PCA': PCA,
            'NearestNeighbors': NearestNeighbors,
            'train_test_split': train_test_split,
            'LogisticRegression': LogisticRegression,
            'roc_auc_score': roc_auc_score,
            'roc_curve': roc_curve,
            'brier_score_loss': brier_score_loss,
            'fit_ddc_coreset': fit_ddc_coreset,
            'fit_random_coreset': fit_random_coreset,
            'fit_stratified_coreset': fit_stratified_coreset,
            'fit_ddc_coreset_by_label': fit_ddc_coreset_by_label,
            'HAS_UMAP': HAS_UMAP,
            'umap': umap,
            'os': os,
            'Path': Path,
            'sys': sys,
        })
        
        logger.info("Standard imports setup complete")
    
    def _should_execute_cell(self, cell: Dict[str, Any]) -> bool:
        """
        Determine if a cell should be executed.
        
        Args:
            cell: Notebook cell dictionary
            
        Returns:
            True if cell should be executed
        """
        if cell.get('cell_type') != 'code':
            return False
        
        source = ''.join(cell.get('source', []))
        
        # Skip empty cells
        if not source.strip():
            return False
        
        # Skip pure comment cells (unless they have plt.savefig)
        if source.strip().startswith('#') and 'plt.savefig' not in source:
            return False
        
        return True
    
    def _extract_figures_from_cell(self, source: str) -> List[str]:
        """
        Extract figure paths from plt.savefig calls in cell source.
        
        Args:
            source: Cell source code
            
        Returns:
            List of figure file paths
        """
        import re
        figures = []
        matches = re.findall(r"plt\.savefig\(['\"]([^'\"]+)['\"]", source)
        for match in matches:
            if os.path.exists(match):
                figures.append(match)
        return figures
    
    def _execute_cell(
        self,
        cell: Dict[str, Any],
        cell_index: int,
    ) -> CellExecutionResult:
        """
        Execute a single notebook cell.
        
        Args:
            cell: Notebook cell dictionary
            cell_index: Index of cell in notebook
            
        Returns:
            CellExecutionResult with execution details
        """
        import time
        
        result = CellExecutionResult(
            cell_index=cell_index,
            cell_type=cell.get('cell_type', 'code'),
            executed=False,
            success=False,
        )
        
        source = ''.join(cell.get('source', []))
        
        start_time = time.time()
        
        try:
            # Execute cell in shared namespace
            exec(source, self.namespace)
            
            result.executed = True
            result.success = True
            
            # Extract generated figures
            result.figures_generated = self._extract_figures_from_cell(source)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            if result.figures_generated:
                logger.info(
                    f"Cell {cell_index}: Generated {len(result.figures_generated)} figures"
                )
        
        except Exception as e:
            result.executed = True
            result.success = False
            result.error = str(e)
            result.error_type = type(e).__name__
            result.execution_time = time.time() - start_time
            
            logger.warning(
                f"Cell {cell_index} error ({result.error_type}): {result.error[:80]}"
            )
        
        return result
    
    def run_notebook(
        self,
        notebook_path: str,
        reset_namespace: bool = False,
    ) -> NotebookExecutionResult:
        """
        Execute a complete notebook.
        
        Args:
            notebook_path: Path to notebook file
            reset_namespace: Whether to reset namespace before execution
            
        Returns:
            NotebookExecutionResult with execution details
        """
        import time
        
        notebook_path = os.path.abspath(notebook_path)
        
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        logger.info(f"Executing notebook: {notebook_path}")
        
        if reset_namespace:
            self._setup_standard_imports()
        
        # Load notebook
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        result = NotebookExecutionResult(
            notebook_path=notebook_path,
            success=False,
            cells_executed=0,
            cells_total=len(nb['cells']),
            errors=0,
        )
        
        start_time = time.time()
        
        # Execute all cells in order
        for i, cell in enumerate(nb['cells']):
            if not self._should_execute_cell(cell):
                continue
            
            cell_result = self._execute_cell(cell, i)
            result.cell_results.append(cell_result)
            
            if cell_result.executed:
                result.cells_executed += 1
                self.stats['cells_executed'] += 1
            
            if cell_result.success:
                result.figures_generated.extend(cell_result.figures_generated)
            else:
                result.errors += 1
                self.stats['errors'] += 1
                
                # Track error types
                error_type = cell_result.error_type or 'Unknown'
                result.error_summary[error_type] = (
                    result.error_summary.get(error_type, 0) + 1
                )
                
                if self.stop_on_error:
                    logger.error(f"Stopping execution due to error in cell {i}")
                    break
        
        result.execution_time = time.time() - start_time
        result.success = result.errors == 0
        result.figures_generated = list(set(result.figures_generated))  # Deduplicate
        
        self.stats['notebooks_run'] += 1
        self.stats['figures_generated'] += len(result.figures_generated)
        
        logger.info(
            f"Notebook execution complete: {result.cells_executed}/{result.cells_total} "
            f"cells, {result.errors} errors, {len(result.figures_generated)} figures"
        )
        
        return result
    
    def run_multiple_notebooks(
        self,
        notebook_paths: List[str],
        reset_namespace: bool = False,
    ) -> List[NotebookExecutionResult]:
        """
        Execute multiple notebooks.
        
        Args:
            notebook_paths: List of notebook file paths
            reset_namespace: Whether to reset namespace between notebooks
            
        Returns:
            List of NotebookExecutionResult objects
        """
        results = []
        
        for nb_path in notebook_paths:
            try:
                result = self.run_notebook(nb_path, reset_namespace=reset_namespace)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute {nb_path}: {e}")
                # Create error result
                error_result = NotebookExecutionResult(
                    notebook_path=nb_path,
                    success=False,
                    cells_executed=0,
                    cells_total=0,
                    errors=1,
                )
                error_result.error_summary['ExecutionError'] = 1
                results.append(error_result)
        
        return results
    
    def generate_report(
        self,
        results: List[NotebookExecutionResult],
        output_file: Optional[str] = None,
    ) -> str:
        """
        Generate execution report.
        
        Args:
            results: List of NotebookExecutionResult objects
            output_file: Optional file path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("NOTEBOOK EXECUTION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total_notebooks = len(results)
        successful = sum(1 for r in results if r.success)
        total_cells = sum(r.cells_executed for r in results)
        total_errors = sum(r.errors for r in results)
        total_figures = sum(len(r.figures_generated) for r in results)
        total_time = sum(r.execution_time for r in results)
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append(f"Notebooks executed: {total_notebooks}")
        report_lines.append(f"Successful: {successful}")
        report_lines.append(f"Failed: {total_notebooks - successful}")
        report_lines.append(f"Total cells executed: {total_cells}")
        report_lines.append(f"Total errors: {total_errors}")
        report_lines.append(f"Total figures generated: {total_figures}")
        report_lines.append(f"Total execution time: {total_time:.2f}s")
        report_lines.append("")
        
        # Per-notebook details
        report_lines.append("DETAILS")
        report_lines.append("-" * 70)
        
        for result in results:
            status = "✅" if result.success else "❌"
            report_lines.append(f"\n{status} {os.path.basename(result.notebook_path)}")
            report_lines.append(f"   Cells: {result.cells_executed}/{result.cells_total}")
            report_lines.append(f"   Errors: {result.errors}")
            report_lines.append(f"   Figures: {len(result.figures_generated)}")
            report_lines.append(f"   Time: {result.execution_time:.2f}s")
            
            if result.figures_generated:
                report_lines.append("   Generated figures:")
                for fig in result.figures_generated:
                    size = os.path.getsize(fig) / 1024 if os.path.exists(fig) else 0
                    report_lines.append(f"     - {os.path.basename(fig)} ({size:.1f} KB)")
            
            if result.error_summary:
                report_lines.append("   Errors:")
                for error_type, count in result.error_summary.items():
                    report_lines.append(f"     - {error_type}: {count}")
        
        # Statistics
        report_lines.append("")
        report_lines.append("STATISTICS")
        report_lines.append("-" * 70)
        for key, value in self.stats.items():
            report_lines.append(f"{key}: {value}")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")
        
        return report


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Execute Jupyter notebooks and generate figures'
    )
    parser.add_argument(
        'notebooks',
        nargs='*',
        help='Notebook file paths (or use --all for all tutorials)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Execute all tutorial notebooks'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop execution on first error'
    )
    parser.add_argument(
        '--reset-namespace',
        action='store_true',
        help='Reset namespace between notebooks'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Save execution report to file'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path for resolving imports'
    )
    
    args = parser.parse_args()
    
    # Determine notebooks to execute
    if args.all:
        base_dir = args.base_path or os.path.dirname(__file__)
        notebooks = [
            os.path.join(base_dir, 'tutorials', 'basic_tabular.ipynb'),
            os.path.join(base_dir, 'tutorials', 'multimodal_clusters.ipynb'),
            os.path.join(base_dir, 'tutorials', 'adaptive_distances.ipynb'),
            os.path.join(base_dir, 'tutorials', 'label_aware_classification.ipynb'),
            os.path.join(base_dir, 'tutorials', 'high_dimensional.ipynb'),
        ]
    elif args.notebooks:
        notebooks = args.notebooks
    else:
        parser.error("Either specify notebook paths or use --all")
    
    # Create runner
    runner = NotebookRunner(
        base_path=args.base_path,
        stop_on_error=args.stop_on_error,
    )
    
    # Execute notebooks
    results = runner.run_multiple_notebooks(
        notebooks,
        reset_namespace=args.reset_namespace,
    )
    
    # Generate report
    report = runner.generate_report(results, output_file=args.report)
    print(report)
    
    # Exit code based on success
    if all(r.success for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

