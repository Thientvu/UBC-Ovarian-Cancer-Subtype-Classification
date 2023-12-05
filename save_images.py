import shutil
import pandas as pd
import os

df = pd.read_csv("/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/UBC-Ovarian-Cancer-subtypE-Classification/train.csv")

root = "/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/train_thumbnails/"
new_root = "/Users/thientoanvu/Desktop/Classes/CS184A/Final-project/train_thumbnails/"


for i in range(len(df)):
    _,image_id,label,_,_,_ = df.iloc[i].to_numpy()

    origin = root + image_id
    print(origin)
    destination = new_root + label
    if not os.path.exists(destination):
        os.makedirs(destination)
    if os.path.exists(origin):
        shutil.move(origin,destination)