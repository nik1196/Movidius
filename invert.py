import os, cv2, numpy as np

for root, dirs, files in os.walk("Test/Outside"):
        for file in files:
            img = cv2.bitwise_not(cv2.imread(os.path.join(root,file)))          
            cv2.imwrite(os.path.join(root,file), img)
