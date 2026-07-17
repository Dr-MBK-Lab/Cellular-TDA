FROM quay.io/jupyter/scipy-notebook:2025-12-31

USER root

# 1. Install system-level dependencies via Mamba
# Includes 'python-lsp-server' for autocompletion logic
RUN mamba install -y -c conda-forge \
        gudhi \
        py-opencv \
        python-lsp-server && \
    mamba clean -a -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# 2. Switch to user for library installs
USER ${NB_UID}
ENV PIP_NO_CACHE_DIR=1

# Install Jupyter AI, Gemini provider, and dotenv for key management
RUN pip install --no-cache-dir \
        "jupyter-ai>=2.0" \
        langchain-google-genai \
        python-dotenv \
        nodejs-bin \
        "ipywidgets>=8,<9" \
        jupyterlab_widgets \
        ripser \
        tadasets \
        xgi \
        "plotly>=5,<7" \
        "kaleido>=0.2,<1"

# 3. Configure Settings & Auto-Load .env
USER root

# Create all necessary setting directories
RUN mkdir -p \
        /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/notebook-extension \
        /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension \
        /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/completer-extension \
        /home/jovyan/.jupyter/lab/user-settings/@krassowski/jupyterlab-lsp \
        /home/jovyan/.ipython/profile_default/startup/

# Enable Autocompletion (Core & LSP) and Theme
RUN printf '%s\n' '{"autoCompletion": true}' \
        > /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/completer-extension/manager.jupyterlab-settings && \
    printf '%s\n' '{"codeCellConfig": {"fontSize": 16, "lineNumbers": true}}' \
        > /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings && \
    printf '%s\n' '{"theme":"JupyterLab Dark"}' \
        > /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings && \
    printf '%s\n' '{"continuousHinting": true}' \
        > /home/jovyan/.jupyter/lab/user-settings/@krassowski/jupyterlab-lsp/completer.jupyterlab-settings

RUN printf "from dotenv import load_dotenv\nload_dotenv('.env')\n" \
        > /home/jovyan/.ipython/profile_default/startup/00-load-env.py
RUN fix-permissions "/home/${NB_USER}"

USER ${NB_UID}
