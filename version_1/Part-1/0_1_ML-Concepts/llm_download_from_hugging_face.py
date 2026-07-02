from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="baidu/ERNIE-4.5-VL-28B-A3B-Thinking",
    local_dir="/home/algo-4/Projects/models/ERNIE-4.5-VL-28B-A3B-Thinking",
    revision="main"
)
