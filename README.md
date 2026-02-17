# ad-astrAI

A multi-agent AI framework for autonomous exoplanet atmospheric analysis. Detect molecular compositions from spectral data (FITS/CSV) and generate comparative spectral fingerprint visualizations (PKL) using a coordinated team of specialized AI agents with LLM-powered reasoning.

## Quick Start

**For first-time users, follow these steps:**

1. **Install dependencies**: `uv sync`
2. **Configure API keys**: Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY`
3. **Start MLflow** (Terminal 1): `uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlruns.db`
4. **Start Spectral Service** (Terminal 2): `cd "Spectral Service" && uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`
5. **Start Web UI** (Terminal 3): `uv run streamlit run app.py`
6. **Upload test data** at http://localhost:8501:
   - **Spectral Analysis**: Upload FITS/CSV files from `test_data/spectral/` (e.g., `earth_ir.fits`)
   - **Graphical Analysis**: Upload PKL files from `test_data/graphical/` (e.g., `jupiter_combined.pkl`)

**Note**: If models are missing, train them first using the instructions in the [Training Models](#training-models) section.

**Reference**: See `Planetary Atmospheric Composition.pdf` for NASA/research-verified element probabilities used in model training.

## Multi-Agent Architecture

This project uses **LangGraph** to coordinate 6 specialized agents:

1.  **Orchestrator Agent**: The brain. Analyzes input files and routes them to the correct analysis pipeline.
    *   *Routes*: `spectral` (FITS/CSV → ML models) or `image` (PKL → visualizations).
2.  **Spectral Model Agent**: Specializes in molecular composition prediction from UV/IR spectral data (FITS/CSV).
3.  **Image Model Agent**: Generates comparative spectral fingerprint visualizations (barcodes, similarity heatmaps, dendrograms) from planetary PKL data.
4.  **Inference Agent**: The synthesizer. Consolidates predictions from spectral models and builds a dynamic **Knowledge Base**.
5.  **Validator Agent**: The quality control. Checks confidence thresholds and flags consistency issues.
6.  **Reporter Agent**: The communicator. Generates human-readable scientific reports using Google Gemini LLM.

### Architecture Diagrams

**Multi-Agent Architecture**
![Multi-Agent Architecture](Multi-Agent%20Architecture.png)

**Data-Model Flow Diagram**
![Data-Model Flow Diagram](Data-model%20flow%20diagram.png)

## Key Features

### 🔬 Spectral Analysis (ML-Powered)
- **Molecular Detection**: Identify 28 UV species and 22 IR species using trained MLP models
- **Domain Flexibility**: Works with UV-only, IR-only, or combined data
- **Physics-Based Training**: Models trained with 75x augmentation per planet (noise, baseline shift, resolution variation)
- **Multi-Modal Validation**: Cross-validates UV and IR predictions for overlapping molecules
- **Confidence Scoring**: Threshold-based filtering with validation flags
- **LLM Reporting**: Natural language scientific reports via Google Gemini

### 🎨 Graphical Analysis (Visualization)
- **Spectral Fingerprints**: Combined UV+IR barcode visualizations showing absorption patterns
- **Similarity Analysis**: Cosine distance heatmap with numerical values for planet comparison
- **Hierarchical Clustering**: Dendrogram showing spectral groupings and relationships
- **Interactive Chat**: LLM-powered Q&A about visualization patterns and planetary similarities

### 📚 Scientific Data Sources
Training labels and validation data sourced from:
- **NASA Planetary Fact Sheets**: Verified atmospheric compositions
- **Peer-reviewed spectroscopy papers**: JWST, HST, and ground-based observations
- **Reference Document**: See `Planetary Atmospheric Composition.pdf` for complete element probability tables

**Supported Molecules**: CO₂, H₂O, CH₄, O₃, N₂, O₂, Ar, SO₂, H₂S, NH₃, HCl, and 18+ additional species

## Technologies

**Core Stack:**
- **LangGraph** - Multi-agent orchestration and state management
- **Google Gemini LLM** - Natural language reasoning and report generation
- **PyTorch + Scikit-learn** - ML model training and inference
- **FastAPI** - High-performance backend API
- **Streamlit** - Interactive web interface
- **MLflow** - Experiment tracking and model registry

**Scientific Libraries:**
- **Astropy** - FITS file handling and astronomical data processing
- **NumPy/Pandas** - Numerical computing and data manipulation
- **Matplotlib/Seaborn** - Scientific visualization
- **SciPy** - Signal processing (Savitzky-Golay filtering, hierarchical clustering)

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` (Fast Python package manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1.  Clone the repository:
    ```bash
    git clone <repo-url>
    cd ad-astrAI
    ```

2.  Install dependencies:
    ```bash
    uv sync
    ```

3.  Configure API keys in `.env` (see `.env.example`):
    ```ini
    GOOGLE_API_KEY=your_gemini_key
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    SPECTRAL_SERVICE_URL=http://localhost:8001
    ```

### Environment Configuration

Create a `.env` file in the project root with the following variables:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Gemini API key for LLM-powered agents | Yes | - |
| `MLFLOW_TRACKING_URI` | MLflow server URL for experiment tracking | No | `http://127.0.0.1:5000` |
| `SPECTRAL_SERVICE_URL` | Spectral Service backend URL | No | `http://localhost:8001` |

**How to get a Gemini API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

## Usage

The application consists of three main services that need to be started in separate terminal windows:

### 1. Start MLflow Server (Required for Tracing)
Start the local MLflow server to track agent execution traces and model training experiments.

```bash
uv run mlflow server --port 5000 --backend-store-uri sqlite:///mlruns.db
```
*View Trace UI at: http://127.0.0.1:5000*

### 2. Start Spectral Service (Backend API)
The Spectral Service provides the machine learning backend for spectral analysis. It runs a FastAPI server that hosts the trained UV and IR spectral models.

**Navigate to the Spectral Service directory and start the server:**
```bash
cd "Spectral Service"
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
*API Documentation available at: http://localhost:8001/docs*

**Note**: The service will automatically load trained models from `Spectral Service/models/`:
- `uv_mlp.pt` - UV spectral model (28 species)
- `ir_mlp.pt` - IR spectral model (22 species)

If models are not found, you will need to train them first (see Training Models section below).

### 3. Run the Web UI
Launch the Streamlit interface for interactive analysis.

**In a new terminal, navigate back to the project root:**
```bash
cd ..
uv run streamlit run app.py
```
*Web UI will open at: http://localhost:8501*

### 4. How to Use the UI

#### **Two Analysis Modes:**

**🔬 Spectral Analysis** (Recommended for Unknown Exoplanets)
- **Upload**: Single FITS or CSV file containing spectral data
- **Works with**: UV-only OR IR-only data
- **Output**: Molecular composition predictions (CO₂, H₂O, CH₄, O₃, etc.)
- **Test Files**: Use files from `test_data/spectral/` (e.g., `earth_ir.fits`, `mars_uv.csv`)

**🎨 Graphical Analysis** (Known Planets Only)
- **Upload**: PKL files containing both UV and IR spectral data
- **Requires**: Complete UV+IR data for comparative analysis
- **Output**: Spectral fingerprint visualizations (barcodes, similarity heatmaps, hierarchical clustering)
- **Test Files**: Use files from `test_data/graphical/` (e.g., `jupiter_combined.pkl`, `earth_uv.pkl` + `earth_ir.pkl`)

#### **View Results:**

**For Spectral Analysis:**
- **Mission Report**: AI-generated scientific summary
- **Agent Trace**: Execution path (Orchestrator → Spectral → Inference → Validator → Reporter)
- **Consolidated Predictions**: Element detection table with confidence scores
- **Knowledge Base**: Multi-modal element validation across UV/IR domains
- **Chat with Data**: Ask questions about molecular composition (e.g., *"Which molecules were detected with high confidence?"*)

**For Graphical Analysis:**
- **Spectral Fingerprints**: UV+IR barcode visualizations for each planet
- **Similarity Matrix**: Cosine distance heatmap with values showing spectral similarity
- **Hierarchical Clustering**: Dendrogram grouping planets by spectral patterns
- **Chat with Visualizations**: Ask about patterns, planet similarities, and molecular fingerprints

## Test Data

The `test_data/` directory contains sample files for both analysis modes:

### Spectral Analysis Test Files (`test_data/spectral/`)
- **FITS format**: `earth_ir.fits`, `mars_uv.fits` - Standard astronomical spectral data
- **CSV format**: `jupiter_ir.csv`, `venus_uv.csv` - Tabular wavelength/flux data
- **Use case**: Test molecular composition prediction on unknown exoplanets

### Graphical Analysis Test Files (`test_data/graphical/`)
- **Combined files**: `jupiter_combined.pkl`, `mars_combined.pkl` - Single file with UV+IR data
- **Separate files**: `earth_uv.pkl` + `earth_ir.pkl` - Upload both for complete analysis
- **Use case**: Generate comparative spectral fingerprint visualizations

**File Format Requirements:**
- **FITS**: Must contain wavelength and flux columns
- **CSV**: Requires `wavelength` (or `wave`) and `flux` (or `radiance`) columns
- **PKL**: Python pickle format with structured numpy arrays or dictionaries

## Training Models

The spectral analysis system uses two separate machine learning models trained on real planetary spectroscopic data with physics-based augmentation.

### Prerequisites for Training
- Real spectral data files must be present in `Spectral Service/data/real/`
  - UV spectra: `*_uv.pkl` files
  - IR spectra: `*_ir.pkl` files

### Train UV Model
```bash
cd "Spectral Service/training"
uv run python train_uv.py
```

**Model specifications:**
- **Input**: UV spectral data (3-channel preprocessed: normalized, 1st derivative, 2nd derivative)
- **Output**: 28 molecular species detection probabilities
- **Architecture**: Multi-Layer Perceptron with Optuna hyperparameter optimization
- **Training**: Physics-based augmentation (75 variations per planet) with planet-level validation split

### Train IR Model
```bash
cd "Spectral Service/training"
uv run python train_ir.py
```

**Model specifications:**
- **Input**: IR spectral data (3-channel preprocessed)
- **Output**: 22 molecular species detection probabilities
- **Architecture**: Multi-Layer Perceptron with Optuna hyperparameter optimization
- **Training**: Physics-based augmentation (75 variations per planet) with planet-level validation split

**Note**: Training uses MLflow for experiment tracking. Ensure MLflow server is running to view training metrics, hyperparameters, and model artifacts.

**Expected Training Time**:
- UV Model: ~30-60 minutes (depends on number of Optuna trials)
- IR Model: ~30-60 minutes

**Trained models will be saved to:**
- `Spectral Service/models/uv_mlp.pt` + `uv_config.json`
- `Spectral Service/models/ir_mlp.pt` + `ir_config.json`

## Project Structure

```
astraAI/
├── agent/                          # Multi-agent system
│   ├── agents/                     # Source code for all 6 agents
│   │   ├── orchestrator.py         # Routing agent
│   │   ├── spectral_model.py       # Spectral model inference agent
│   │   ├── image_model.py          # Spectral fingerprint visualization agent
│   │   ├── inference_agent.py      # Prediction consolidation
│   │   ├── validator_agent.py      # Quality control
│   │   └── reporter_agent.py       # LLM-powered report generation
│   ├── graph.py                    # LangGraph definitions and routing logic
│   └── state.py                    # Shared state schema
│
├── Spectral Service/               # Backend ML service
│   ├── app/                        # FastAPI application
│   │   ├── main.py                 # API entry point
│   │   ├── routers/
│   │   │   └── analyze.py          # Spectral analysis endpoints
│   │   └── utils/
│   │       └── io.py               # FITS/CSV/PKL data loaders
│   ├── training/                   # Model training pipeline
│   │   ├── train_uv.py             # UV model training
│   │   ├── train_ir.py             # IR model training
│   │   ├── augmentation.py         # Physics-based augmentation
│   │   ├── expanded_species.py     # Species definitions & planet labels
│   │   └── mlflow_utils.py         # MLflow integration
│   ├── data/
│   │   └── real/                   # Real planetary spectra (training data)
│   │       ├── earth_uv.pkl        # Earth UV spectrum
│   │       ├── earth_ir.pkl        # Earth IR spectrum
│   │       ├── jupiter_uv.pkl      # Jupiter UV spectrum
│   │       ├── jupiter_ir.pkl      # Jupiter IR spectrum
│   │       └── ...                 # Other planets (Mars, Venus, etc.)
│   └── models/                     # Trained models (generated after training)
│       ├── uv_mlp.pt               # UV model weights
│       ├── uv_config.json          # UV model configuration
│       ├── ir_mlp.pt               # IR model weights
│       └── ir_config.json          # IR model configuration
│
├── test_data/                      # Sample test files for users
│   ├── spectral/                   # Test files for Spectral Analysis
│   │   ├── earth_ir.fits           # Earth IR spectrum (FITS)
│   │   ├── mars_uv.csv             # Mars UV spectrum (CSV)
│   │   └── ...                     # Other test spectra
│   └── graphical/                  # Test files for Graphical Analysis
│       ├── jupiter_combined.pkl    # Jupiter combined UV+IR
│       ├── earth_uv.pkl            # Earth UV only (requires pair)
│       ├── earth_ir.pkl            # Earth IR only (requires pair)
│       └── ...                     # Other planet PKL files
│
├── app.py                          # Streamlit frontend application
├── experiments/                    # Jupyter notebooks for prototyping
├── pyproject.toml                  # Project dependencies (uv package manager)
├── .env                            # Environment configuration (API keys)
├── GCP_DEPLOY.md                   # Google Cloud Platform deployment guide
├── Planetary Atmospheric Composition.pdf  # NASA/research reference data
└── README.md                       # This file
```

## Observability

This project uses **MLflow Tracing** for deep observability.
- **Spans**: Track every agent's execution time and inputs/outputs.
- **Metrics**: Monitor token usage, latency, and tool calls.
- **Artifacts**: Store generated reports and data snapshots.

## Deployment

### Local Development
Follow the [Quick Start](#quick-start) section above.

### Google Cloud Platform
For production deployment on GCP Virtual Machines, see the comprehensive guide:

📘 **[GCP_DEPLOY.md](GCP_DEPLOY.md)** - Complete deployment instructions including:
- VM setup with firewall configuration
- UV package manager installation
- tmux-based service management
- Cost optimization (starts at $10/month with spot instances)
- Troubleshooting and monitoring

**Quick Deploy Summary:**
1. Create GCP VM (n1-standard-2 recommended)
2. Open firewall ports: 5000 (MLflow), 8001 (Spectral Service), 8501 (Streamlit)
3. SSH into VM and clone repository
4. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
5. Install dependencies: `uv sync`
6. Start services in tmux (see GCP_DEPLOY.md for commands)
7. Access via `http://YOUR_VM_IP:8501`

## Troubleshooting

### Spectral Service Connection Error
**Issue**: Web UI shows "Failed to connect to Spectral Service"

**Solution**:
1. Ensure Spectral Service is running on port 8001
2. Check `.env` file has `SPECTRAL_SERVICE_URL=http://localhost:8001`
3. Verify models are trained and present in `Spectral Service/models/`

### Models Not Found
**Issue**: Spectral Service returns "Model not found" error

**Solution**:
1. Train the models using `train_uv.py` and `train_ir.py`
2. Verify `.pt` and `.json` files exist in `Spectral Service/models/`
3. Restart the Spectral Service after training

### Validation Loss = 0.0000
**Issue**: During training, validation loss shows exactly 0.0000

**Solution**: This indicates data leakage or overfitting:
- For **planet-level split**: Need 4+ planets minimum
- For **≤3 planets**: System uses sample-level split (expected behavior)
- Add more real planetary spectra to `Spectral Service/data/real/`

### High Validation Loss (>1.0)
**Issue**: Validation loss is very high during training

**Solution**:
- Check if you have enough training data (recommended: 4+ planets)
- Verify spectral data quality in `.pkl` files
- Increase `N_AUGMENT_PER_PLANET` parameter in training scripts

### LLM Routing Errors
**Issue**: Agent fails to route correctly or returns JSON parsing errors

**Solution**:
1. Verify `GOOGLE_API_KEY` is valid and active
2. Check Gemini API quota limits
3. Review MLflow traces to see exact LLM responses

### Port Already in Use
**Issue**: "Address already in use" error when starting services

**Solution**:
- **MLflow (5000)**: Change port in command: `mlflow server --port 5001`
- **Spectral Service (8001)**: Change port in command and update `.env`
- **Streamlit (8501)**: Streamlit will auto-increment to 8502

### Graphical Analysis Requires Both UV and IR Data
**Issue**: "Missing: planet_ir.pkl" error when uploading single PKL file

**Solution**:
- **Option 1**: Upload a combined file (e.g., `jupiter_combined.pkl`)
- **Option 2**: Upload both UV and IR files together (e.g., `jupiter_uv.pkl` + `jupiter_ir.pkl`)
- **Option 3**: For unknown exoplanets with incomplete data, use **Spectral Analysis** mode instead

**Note**: Graphical Analysis is designed for comparative visualization of known planets with complete UV+IR data. For molecular composition prediction on single-domain data, use Spectral Analysis mode.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns and structure
- All tests pass before submitting
- Documentation is updated for new features
- Commit messages are clear and descriptive

## License

This project is part of an academic research initiative.
