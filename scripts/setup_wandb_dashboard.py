import wandb_workspaces.workspaces as ws

workspace = ws.Workspace(
   name="Example W&B Workspace",
   entity="your-entity",
   project="your-project",
   sections=[
      ws.Section(
            name="Validation Metrics",
            panels=[
               wr.LinePlot(x="Step", y=["val_loss"]),
               wr.BarPlot(metrics=["val_accuracy"]),
               wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
            ],
            is_open=True,
      ),
   ],
).save()