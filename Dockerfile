FROM runpod/worker-comfyui:5.0.0-base
RUN comfy-node-install comfyui-kjnodes comfyui-florence2 comfyui-videohelpersuite comfyui-segment-anything-2
