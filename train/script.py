import pandas as pd
import re
data = pd.read_excel("C:\\Users\\HP\\Desktop\\Projets\\Deep Learning\\face_recognition\\train\\student_data.xlsx")
data["Vector_Path"] = data["Vector Path"].apply(lambda x: re.sub(r"C:\\Users\\HP\\Desktop\\Projets\\Deep Learning\\face_recognition\\", "", x))
data = data.drop("Vector Path", axis = 1)
data.to_excel("Students_Data.xlsx", index=False)