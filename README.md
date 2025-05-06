# For training models
conda activate recommendation

# For running the API
conda activate api
cd /home/ens/projects/loyalty-recom
uvicorn api.app:app --reload --port 8000