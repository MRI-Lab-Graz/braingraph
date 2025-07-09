# braingraph

Import and analyse brain-derived connectomics using graph theory

## Overview

**braingraph** is a Python toolkit for extracting and analyzing graph-theoretic metrics from brain connectome data. It processes connectivity matrices and computes a comprehensive set of network measures including global efficiency, modularity, clustering coefficient, small-worldness, and nodal metrics.

Key features:
- 🧠 **Brain-specific**: Designed for neuroimaging connectome data
- 📊 **Comprehensive metrics**: Global, nodal, and network-level measures
- 🔄 **Parallel processing**: Multi-core support for large datasets
- ✅ **Data validation**: Automatic shape consistency and format checking
- 📈 **Statistical analysis**: Mixed-effects modeling with configurable formulas

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/braingraph.git
   cd braingraph
   ```

2. **Install dependencies:**
   ```bash
   ./install.sh
   ```

3. **Activate the environment:**
   ```bash
   source braingraph/bin/activate
   ```

The `install.sh` script will:
- Create a virtual environment named `braingraph`
- Install all required packages including `networkx`, `bctpy`, `pandas`, `statsmodels`, and visualization libraries
- Set up PyTorch for optional GPU acceleration

### Manual Installation

If you prefer manual setup:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv braingraph
source braingraph/bin/activate

# Install packages
uv pip install numpy pandas networkx matplotlib seaborn statsmodels bctpy python-louvain torch scikit-learn
```

## Data Preparation

### Input Format

**braingraph** expects connectivity matrices in tab-separated text files with the following structure:

```
data    data    node1   node2   node3   ...
data    data    SFG_L   SFG_R   MFG_L   ...
value1  label1  0       2.24    1.56    ...
value2  label2  2.24    0       0.38    ...
...
```

**Format Configuration:**
- **Header rows**: Number of rows to skip (default: 2 for DSI Studio format)
- **Metadata columns**: Number of columns to skip (default: 3 for DSI Studio format)
- **Remaining data**: Square connectivity matrix (n_nodes × n_nodes)

The script is flexible and can handle different software formats by configuring `skip_rows` and `skip_cols` parameters.

### File Naming Convention

Files should follow the pattern: `sub-XX_ses-Y.txt`
- `XX`: Subject number (e.g., 01, 02, ..., 22)
- `Y`: Session/timepoint number (e.g., 1, 2, 3, 4)

### Directory Structure

Organize your data as follows:
```
your_project/
├── connectomes/
│   ├── sub-01_ses-1.txt
│   ├── sub-01_ses-2.txt
│   ├── sub-02_ses-1.txt
│   └── ...
├── config.json
└── results/
```

### Data Requirements

- **Square matrices**: All connectivity matrices must be square (n × n)
- **Consistent dimensions**: All files must have identical matrix dimensions
- **Symmetric matrices**: Connectivity should be symmetric (undirected graphs)
- **Non-negative values**: All connection weights should be ≥ 0
- **No self-connections**: Diagonal elements should be 0

## Usage

### 1. Check Input Format

Before processing, verify your data format:

```bash
python extract_graph.py --checkinput -i connectomes/
```

This will:
- Show the first 10 lines of your data
- Display matrix dimensions after preprocessing
- Verify the parsing is correct

### 2. Extract Graph Metrics

#### Basic Usage
```bash
python extract_graph.py -i connectomes/ -o results/
```

#### With Configuration File
```bash
python extract_graph.py --config config.json
```

#### Advanced Options
```bash
python extract_graph.py \
    -i connectomes_data/ \
    -o results/ \
    --n_cpus 8 \
    --skip_rows 2 \
    --skip_cols 3
```

#### Different Software Formats
```bash
# For ConnectomeDB format (hypothetical: 1 header row, 2 metadata columns)
python extract_graph.py \
    -i connectomedb_data/ \
    --skip_rows 1 \
    --skip_cols 2

# For FreeSurfer format (hypothetical: 0 header rows, 1 metadata column)
python extract_graph.py \
    -i freesurfer_data/ \
    --skip_rows 0 \
    --skip_cols 1
```

### 3. Statistical Analysis

Analyze extracted metrics with mixed-effects models:

```bash
python graph_stats.py --config config.json
```

## Configuration

Create a `config.json` file to customize processing:

```json
{
    "input_dir": "./connectomes",
    "output_dir": "./results",
    "output_prefix": "graph_metrics",
    "n_cpus": 4,
    "skip_rows": 2,
    "skip_cols": 3,
    "min_nonzero": 10,
    "min_sum": 1e-4,
    "model_formula": "metric ~ timepoint + (1|subject)",
    "data_file": "results/graph_metrics_global.csv",
    "output_file": "results/statistical_results.csv",
    "metrics": ["global_efficiency", "modularity", "clustering_coef"]
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dir` | Directory containing connectome files | `"./connectomes"` |
| `output_dir` | Output directory for results | `"."` |
| `output_prefix` | Prefix for output filenames | `"graph_metrics"` |
| `n_cpus` | Number of CPU cores to use | All available |
| `skip_rows` | Number of header rows to skip | `2` (DSI Studio) |
| `skip_cols` | Number of metadata columns to skip | `3` (DSI Studio) |
| `min_nonzero` | Minimum non-zero connections | `10` |
| `min_sum` | Minimum sum of connection weights | `1e-4` |
| `model_formula` | Statistical model formula | - |
| `metrics` | List of metrics to analyze | All metrics |

## Output

### Graph Metrics Files

The extraction generates two main output files:

1. **`graph_metrics_TIMESTAMP_global.csv`**: Global network measures
   - `global_efficiency`: Network's global efficiency
   - `char_path_length`: Characteristic path length
   - `modularity`: Modularity (community structure)
   - `clustering_coef`: Average clustering coefficient
   - `density`: Network density
   - `assortativity`: Degree assortativity
   - `transitivity`: Network transitivity
   - `small_world`: Small-worldness sigma

2. **`graph_metrics_TIMESTAMP_nodal.csv`**: Node-level measures
   - `degree`: Node degree
   - `strength`: Node strength (weighted degree)
   - `eigenvector_centrality`: Eigenvector centrality
   - `betweenness`: Betweenness centrality
   - `local_efficiency`: Local efficiency
   - `clustering`: Local clustering coefficient

### Statistical Results

Statistical analysis produces:
- **`statistical_results.csv`**: Model results with coefficients, p-values, and confidence intervals
- **Plots**: Visualization of significant effects (saved to `plots/` directory)

## Examples

### Example 1: Process DSI Studio Data

```bash
# Check data format
python extract_graph.py --checkinput -i dsi_connectomes/

# Extract metrics
python extract_graph.py -i dsi_connectomes/ -o results/ --n_cpus 8

# Analyze timepoint effects
python graph_stats.py --config timepoint_analysis.json
```

### Example 2: Custom Analysis Pipeline

```bash
# 1. Verify 116x116 atlas data
python extract_graph.py --checkinput -i aal_connectomes/

# 2. Extract with custom thresholds
python extract_graph.py \
    -i aal_connectomes/ \
    -o aal_results/ \
    --config aal_config.json

# 3. Statistical modeling
python graph_stats.py --config aal_stats.json
```

### Example 3: Different Software Formats

```bash
# Check format for different software
python extract_graph.py --checkinput -i data_folder/ --skip_rows 1 --skip_cols 2

# Extract with custom format parameters
python extract_graph.py \
    -i mrtrix_data/ \
    -o results/ \
    --skip_rows 0 \
    --skip_cols 1

# Use configuration file for reproducibility
echo '{
    "input_dir": "./custom_data",
    "skip_rows": 1,
    "skip_cols": 2,
    "n_cpus": 6
}' > custom_config.json

python extract_graph.py --config custom_config.json
```

### Example 4: Quality Control

```bash
# Check for shape inconsistencies
python extract_graph.py -i mixed_data/ -o qc_results/
# Will stop if matrices have different dimensions

# Process only after confirming consistency
python extract_graph.py -i clean_data/ -o final_results/ --n_cpus 12
```

## Computed Metrics

### Global Metrics
- **Global Efficiency**: Average inverse shortest path length
- **Characteristic Path Length**: Average shortest path length
- **Modularity**: Community structure strength (Louvain algorithm)
- **Clustering Coefficient**: Average local clustering
- **Density**: Proportion of existing connections
- **Assortativity**: Tendency for similar nodes to connect
- **Transitivity**: Global clustering coefficient
- **Small-worldness**: Balance between clustering and path length

### Nodal Metrics
- **Degree**: Number of connections per node
- **Strength**: Sum of connection weights per node
- **Eigenvector Centrality**: Influence based on connections to important nodes
- **Betweenness Centrality**: Fraction of shortest paths through each node
- **Local Efficiency**: Efficiency of node's neighborhood
- **Local Clustering**: Clustering coefficient per node

## Troubleshooting

### Common Issues

**Shape mismatch errors:**
```
SHAPE MISMATCH DETECTED!
All matrices must have the same shape.
```
- Ensure all connectivity matrices have identical dimensions
- Check for corrupted or incomplete files
- Verify consistent atlas/parcellation across subjects

**Memory issues with large datasets:**
- Reduce `n_cpus` parameter
- Process data in smaller batches
- Consider using a machine with more RAM

**Import errors:**
- Ensure virtual environment is activated: `source braingraph/bin/activate`
- Reinstall dependencies: `./install.sh`

### Performance Tips

- Use `--checkinput` first to validate data format
- Set `n_cpus` to match your system (default uses all cores)
- For very large datasets (>1000 nodes), consider subsampling
- GPU acceleration (PyTorch) is automatically used when available

## Citation

If you use braingraph in your research, please cite:

```bibtex
@software{braingraph,
  title = {braingraph: Graph-theoretic analysis of brain connectomes},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/braingraph}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- 📖 **Documentation**: See examples and configuration options above
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Ask questions in GitHub Discussions
