FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch torchvision torchaudio triton \
    --index-url https://download.pytorch.org/whl/rocm6.1

RUN cat << 'EOF' > /tmp/requirements.txt
contourpy==1.3.3
cycler==0.12.1
filelock==3.29.0
fonttools==4.62.1
fsspec==2026.4.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.4
opencv-python==4.13.0.92
packaging==26.2
pandas==3.0.2
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
EOF

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
