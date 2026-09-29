# syntax=docker/dockerfile:1
# Trusted operator build only. The agent development, rehearsal and formal container.
# Python and packages match the official AgentBundle policy environment exactly: one
# wheelhouse (pinned torch/warp, the pinned cuRobo fork and robodojo_toolkit) serves
# both this image and the XPolicyLab submission. Agent containers never use the network.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*
# Bind-mount the wheelhouse so its 4 GB never becomes an image layer.
RUN --mount=type=bind,source=wheelhouse,target=/tmp/wheelhouse \
    python -m pip install --no-cache-dir --no-index --find-links /tmp/wheelhouse -r /tmp/wheelhouse/constraints.txt \
    && python -m pip install --no-cache-dir --no-index --no-deps /tmp/wheelhouse/nvidia_curobo-*.whl /tmp/wheelhouse/robodojo_toolkit-*.whl \
    && python -m pip check
# Prebuilt portable (sm_80+ PTX) cuRobo kernels, the same cache shipped with submissions.
COPY warp-cache /opt/robodojo/warp-cache
# Development-only asset inspection for workspace task_source/ (usd-core opens the
# original .usdz files; trimesh from the wheelhouse reads the exported mesh.obj).
# Not in the submission wheelhouse, so bundles must not import pxr.
RUN --mount=type=bind,source=inspect-wheels,target=/tmp/inspect-wheels \
    python -m pip install --no-cache-dir --no-index --no-deps /tmp/inspect-wheels/*.whl \
    && python -m pip check
ENV PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    ROBODOJO_TOOLKIT_WARP_SEED=/opt/robodojo/warp-cache
# Launchers always override the UID and enforce the actual runtime restrictions.
USER 65534:65534
