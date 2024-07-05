#STEP 1 
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


# STEP 2
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))


# STEP 3
img1 = cv2.imread("iu1.jpg")
img2 = cv2.imread("iu2.jpg")

# STEP 4
faces1 = app.get(img1)
faces2 = app.get(img2)

assert len(faces1)==1 #얼굴이 하나여야함
assert len(faces2)==1
print(faces1[0])

# STEP 5
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./iu_output.jpg", rimg)



# then print all-to-all face similarity 여러개의 얼굴
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)


feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T) #array를 행렬곱 --> 코사인 유사도  .T를 하면 뒤집어짐(행렬곱을 위해서 뒤집어야함)
print(sims) 
#0.5828101 --> -1~1에서 얼마나 비슷한지(퍼센트 아님)
#  타인과 동일인은 어떤 방식으로든 겹치는데 여기서 어떠한 기준을 잡아서 겹치지 않도록 할것인지 기준을 잡아줘야함
