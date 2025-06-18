# start from a clean base image (replace <version> with the desired release)
FROM runpod/worker-comfyui:5.1.0-base
RUN comfy-node-install segment-anything


