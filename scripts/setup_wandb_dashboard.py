# install once: pip install wandb wandb-workspaces
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "your-entity"
PROJECT = "your-project"

wandb.login()

# 1) Create a shareable Report with the charts you want
report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="My Training Dashboard",
    description="Key metrics with my preferred layout & sorting.",
)
report.blocks = [
    wr.PanelGrid(panels=[
        wr.LinePlot(x="step", y=["train/loss", "val/loss"], title="Loss"),
        wr.LinePlot(x="step", y=["val/accuracy"], title="Val Accuracy"),
        wr.ScalarChart(metric="val/accuracy", title="Final Val Acc"),
    ])
]
report.save()
print("Report URL:", report.url)

# 2) (Optional) Create a Saved Workspace view and sort runs by a metric
workspace = ws.Workspace(
    name="Sorted Runs View",
    entity=ENTITY,
    project=PROJECT,
    sections=[
        ws.Section(
            name="Main",
            panels=[
                wr.LinePlot(x="step", y=["val/accuracy"]),
                wr.LinePlot(x="step", y=["train/loss"]),
            ],
            is_open=True,
        ),
    ],
    runset_settings=ws.RunsetSettings(
        # filter/group if you like; here we just sort by summary metric:
        order=[ws.Ordering(ws.Summary("val/accuracy"), ascending=False)]
    ),
)
workspace.save()
print("Workspace URL:", workspace.url)