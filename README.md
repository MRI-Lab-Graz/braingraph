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
   git clone https://github.com/MRI-Lab-Graz/braingraph.git
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

### 4. Specialized Running Training Analysis

For the specific experimental design with 22 subjects × 4 sessions where training starts after session 2, use the specialized analysis script:

```bash
python running_analysis.py --config running_config.json
```

#### Configuration File

Create a `running_config.json` file:

```json
{
    "input_file": "graph_metrics_global.csv",
    "output_folder": "running_results", 
    "metrics": [
        "global_efficiency",
        "modularity",
        "clustering_coef",
        "transitivity",
        "char_path_length",
        "small_world"
    ]
}
```

#### Alternative Usage

You can also specify parameters directly via command line:

```bash
# Direct command line usage
python running_analysis.py --data graph_metrics_global.csv --output running_results/

# Mix config file with command line overrides
python running_analysis.py --config running_config.json --metrics global_efficiency modularity
```

This script implements three complementary statistical approaches:

#### Analysis Methods

1. **Contrast Analysis** (Primary approach)
   - Compares control period (T1→T2) vs training period (T2→T4)
   - Paired t-tests and effect size calculations
   - Baseline-adjusted analyses

2. **Piecewise Analysis**
   - Tests for slope changes at training onset
   - Separate linear trends for control and training periods
   - Mixed-effects modeling with random intercepts

3. **Dose-Response Analysis**
   - Examines linear relationship with training exposure
   - Focuses on training period sessions (T2, T3, T4)
   - Tests for cumulative training effects

#### Running Analysis Options

```bash
# Recommended: Use configuration file
python running_analysis.py --config running_config.json

# Analyze specific metrics with config override
python running_analysis.py \
    --config running_config.json \
    --metrics global_efficiency modularity clustering_coef

# Direct command line (without config file)
python running_analysis.py \
    --data graph_metrics_global.csv \
    --output running_results/ \
    --metrics global_efficiency modularity
```

#### Output Files

The running analysis generates:
- **Summary report**: `running_analysis_summary_TIMESTAMP.txt`
- **Individual plots**: Comprehensive 4-panel visualizations per metric
- **Change scores**: `{metric}_changes.csv` files with period-specific changes
- **Statistical details**: Model summaries and effect sizes

#### Interpretation Guidelines

**Significant Training Effects:**
- p < 0.05 for training vs control contrast
- Effect size (Cohen's d): Small (0.2), Medium (0.5), Large (0.8)
- Positive slope change in piecewise analysis
- Dose-response correlation with training sessions

**Example Results Interpretation:**
```
Global Efficiency:
  Training vs Control difference: 0.0234
  P-value: 0.023
  Effect size (Cohen's d): 0.68
  Significance: *
```
This indicates a medium-to-large training effect on global efficiency.

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
    "input_file": "graph_metrics_global.csv",
    "output_folder": "results",
    "model": "{metric} ~ timepoint",
    "type": "global",
    "metrics": ["global_efficiency", "modularity", "clustering_coef"]
}
```

**Note**: Different scripts use slightly different config keys:
- `extract_graph.py` and `graph_stats.py`: Use the main `config.json`
- `running_analysis.py`: Uses `running_config.json` (optimized for training analysis)
- Command line arguments always override config file settings

**Note**: The configuration file serves both `extract_graph.py` and `graph_stats.py`, so it contains parameters for both scripts.

### Configuration Parameters

| Parameter | Description | Default | Used By |
|-----------|-------------|---------|---------|
| `input_dir` | Directory containing connectome files | `"./connectomes"` | extract_graph.py |
| `output_dir` | Output directory for results | `"."` | extract_graph.py |
| `output_prefix` | Prefix for output filenames | `"graph_metrics"` | extract_graph.py |
| `n_cpus` | Number of CPU cores to use | All available | extract_graph.py |
| `skip_rows` | Number of header rows to skip | `2` (DSI Studio) | extract_graph.py |
| `skip_cols` | Number of metadata columns to skip | `3` (DSI Studio) | extract_graph.py |
| `min_nonzero` | Minimum non-zero connections | `10` | extract_graph.py |
| `min_sum` | Minimum sum of connection weights | `1e-4` | extract_graph.py |
| `input_file` | Path to graph metrics CSV file | - | graph_stats.py, running_analysis.py |
| `output_folder` | Output directory for statistical results | `"results"` | graph_stats.py, running_analysis.py |
| `model` | Statistical model formula (use `{metric}` placeholder) | - | graph_stats.py |
| `type` | Analysis type (`"global"` or `"nodal"`) | `"global"` | graph_stats.py |
| `metrics` | List of metrics to analyze | All metrics | All scripts |

### Statistical Model Examples

The `model` parameter uses statsmodels formula syntax with `{metric}` as a placeholder:

**Fixed Effects Models:**
```json
"model": "{metric} ~ timepoint"                    // Simple time effect
"model": "{metric} ~ timepoint + group"            // Time + group effects  
"model": "{metric} ~ timepoint * group"            // Time × group interaction
```

**Mixed Effects Models (requires subject grouping):**
```json
"model": "{metric} ~ timepoint + (1|subject)"           // Random intercepts
"model": "{metric} ~ timepoint + group + (1|subject)"   // Fixed + random effects
"model": "{metric} ~ timepoint + (timepoint|subject)"   // Random slopes
```

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

### Example 3: Complete Running Training Analysis

```bash
# 1. Check your data format
python extract_graph.py --checkinput -i connectomes/

# 2. Extract graph metrics
python extract_graph.py --config config.json

# 3. Run specialized running training analysis
python running_analysis.py --config running_config.json

# 4. Optional: Standard statistical analysis for comparison
python graph_stats.py --config config.json
```

### Example 4: Different Software Formats

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

### Example 5: Quality Control

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
