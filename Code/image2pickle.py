import pickle
import cv2

# Đọc ảnh từ file
image_path = '../1.jpg'
image = cv2.imread(image_path)

# Tạo file pickle
pickle_path = '123.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(image, f)
