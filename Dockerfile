# start from a clean base image (replace <version> with the desired release)
FROM runpod/worker-comfyui:5.1.0-base

# install custom nodes using comfy-cli
RUN comfy-node-install comfyui-kjnodes 
RUN comfy-node-install comfyui-florence2 
RUN comfy-node-install comfyui-videohelpersuite 
RUN comfy-node-install comfyui-segment-anything-2

