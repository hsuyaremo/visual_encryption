import cv2
import numpy as np 

ref_mat = np.array([[7,3,1, 5,9,8, 4,2,6],
					[5,9,8, 4,2,6, 7,3,1],
					[4,2,6, 7,3,1, 5,9,8],

					[3,1,7, 9,8,5, 2,6,4],
					[9,8,5, 2,6,4, 3,1,7],
					[2,6,4, 3,1,7, 9,8,5],

					[1,7,3, 8,5,9, 6,4,2],
					[8,5,9, 6,4,2, 1,7,3],
					[6,4,2, 1,7,3, 8,5,9]])

# Rescaled matrix
res_mat = np.zeros((9,9),dtype = int)
for i in range(9):
	for j in range(9):
		res_mat[i][j] = ((ref_mat[i][j]-1.0)/8.0) * 255

def Encrypt(img):
	'''
	Changes pixel values using rescaled sudo matrix.
	chages position by mapping (x,y) in image to (block,digit) in sudoku matrix
	'''  

	r, c = img.shape

	Encrypted_img = np.array(img)

	for i in range(r/9):
		for j in range(c/9):
			patch = np.zeros((9,9),dtype= np.uint8)
			for k in range(9):
				for l in range(9):
					v1 = img[i*9:(i+1)*9,j*9:(j+1)*9][k][l]
					v2 = res_mat[k][l]
					x = (k/3)*3 + (l/3)
					y = ref_mat[k][l] - 1
					patch[x][y] = (v1 + v2)%256
			Encrypted_img[i*9:(i+1)*9,j*9:(j+1)*9] = patch

	return Encrypted_img

def Decrypt(img):

	r, c = img.shape

	Decrypted_img = np.array(img)

	# Reverse map for decryption
	rev_map = np.zeros((9,9,2), dtype = int )

	for k in range(9):
		for l in range(9):
			x = (k/3)*3 + (l/3)
			y = ref_mat[k][l] - 1
			rev_map[x][y][0] = k
			rev_map[x][y][1] = l
 
	for i in range(r/9):
		for j in range(c/9):
			patch = np.zeros((9,9),dtype= np.uint8)
			for k in range(9):
				for l in range(9):
					x, y = rev_map[k][l] 
					v1 = img[i*9:(i+1)*9,j*9:(j+1)*9][k][l]
					v2 = res_mat[x][y]
					patch[x][y] = (v1 + 256 - v2)%256
			Decrypted_img[i*9:(i+1)*9,j*9:(j+1)*9] = patch

	return Decrypted_img

input_image = "A.png"

# read image in gray scale
img = np.array(cv2.resize(cv2.imread(input_image, 0),(270,270)))
Encrypted_img =img
for i in range(2):
	Encrypted_img = Encrypt(Encrypted_img)
Decrypted_img = Encrypted_img
for j in range(2):
	Decrypted_img = Decrypt(Decrypted_img)

cv2.imshow("Input image",img)
cv2.imshow("Encrypted",Encrypted_img)
cv2.imshow("Decrypted",Decrypted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
