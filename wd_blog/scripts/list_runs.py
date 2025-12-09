import wandb
api = wandb.Api()
runs = api.runs("marin-community/optimizer-scaling")
for run in runs:
    print(f"Run: {run.name}, ID: {run.id}, State: {run.state}")
