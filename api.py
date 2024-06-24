# pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="LeB9AMuPwOcjuJWQxTE4")
project = rf.workspace("ds-lxa2d").project("apples-daz2v")
version = project.version(2)
dataset = version.download("yolov8")
