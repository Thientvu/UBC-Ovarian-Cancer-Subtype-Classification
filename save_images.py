import shutil
import pandas as pd
import os

df = pd.read_csv("path to csv file")

root = "path to folder with all images"
new_root = "path to root folder"

for i in range(len(df)):
    _,image_id,label,_,_,_ = df.iloc[i].to_numpy()

    origin = root + image_id
    destination = new_root + label
    if not os.path.exists(destination):
        os.makedirs(destination)
    if os.path.exists(origin):
        shutil.move(origin,destination)