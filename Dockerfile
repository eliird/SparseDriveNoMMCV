FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

# Install Python and essential tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 11.6 support
# Pin typing-extensions to version compatible with Python 3.8
RUN pip3 install typing-extensions==4.8.0 && \
    pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install build dependencies for flash-attn and other packages
RUN pip3 install --upgrade pip setuptools wheel packaging ninja

# Copy and install Python requirements (excluding flash-attn first)
COPY requirement.txt /workspace/
RUN grep -v "flash-attn" requirement.txt > /tmp/requirements_no_flash.txt && \
    pip3 install -r /tmp/requirements_no_flash.txt

# Install flash-attn separately with proper CUDA environment
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=4
RUN pip3 install flash-attn==2.3.2 --no-build-isolation

CMD ["/bin/bash"]
