import os
import sys
import pathlib
import supervisely as sly
from dotenv import load_dotenv


logger = sly.logger

load_dotenv("tapnet/supervisely/serve/debug.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))
api = sly.Api()
my_app = sly.AppService()
task_id = my_app.task_id
sly.fs.clean_dir(my_app.data_dir)
team_id = int(os.environ["context.teamId"])
workspace_id = int(os.environ["context.workspaceId"])
checkpoint_path = "./checkpoint/checkpoint.npy"
save_video_dir = "./input_data"
