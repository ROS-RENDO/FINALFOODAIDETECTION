# Download the dataset from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="K1epz6Y0VhIcXlo3HLLj")
project = rf.workspace("fruit-detection-byxkd").project("freshness-fruits-and-vegetables-nojcz")
version = project.version(1)
dataset = version.download("yolov8")
                