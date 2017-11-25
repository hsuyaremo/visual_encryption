import cv2
import numpy as np 

d=2

def Encrypt(img):
	'''
	takes img as input image in black and white.
	Makes 2 shares of image such that only one share does not guve any info.
	To see the full info of image one must use both shares.
	'''

	r, c = img.shape

	mat_for_white = np.array([[1,0],[1,0]])
	mat_for_black = np.array([[1,0],[0,1]])
	shares = np.zeros((2,r,d*c), dtype = np.uint8)

	for i in range(r):
		for j in range(c):
			# Black Pixel
			if img[i][j] == 0:
				idx = [0,1]
				np.random.shuffle(idx)
				for k in range(d):
					shares[0][i][d*j + k] = mat_for_black[0][idx[k]]
					shares[1][i][d*j + k] = mat_for_black[1][idx[k]]
			else :
				idx = [0,1]
				np.random.shuffle(idx)
				for k in range(d):
					shares[0][i][d*j + k] = mat_for_white[0][idx[k]]
					shares[1][i][d*j + k] = mat_for_white[1][idx[k]]

	return shares

def Decrypt(shares):
	'''
	Adds both shares to get orginal image.
	'''
	r, c = shares[0].shape

	Decr = np.zeros((r,c), dtype = np.uint8)
	for i in range(r):
		for j in range(c):
			Decr[i][j] += shares[0][i][j] + shares[1][i][j]
			Decr[i][j] = (1- Decr[i][j]) *255

	return Decr

input_image = "A.png"

# read image in gray scale
img = cv2.resize(cv2.imread(input_image, 0), None, fx = .5, fy = .5) 
# convert in black and white
img_bw = np.array(cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
# Encryption 
shares = Encrypt(img_bw)

Encrypt1 = np.array(shares[0])
Encrypt2 = np.array(shares[1])

for i in range(Encrypt1.shape[0]):
	for j in range(Encrypt2.shape[1]):
		Encrypt1[i][j] = (1-Encrypt1[i][j])*255
		Encrypt2[i][j] = (1-Encrypt2[i][j])*255

Decrypt_img = Decrypt(shares)

cv2.imshow("Input image",img_bw)
cv2.imshow("Encrypt1",Encrypt1)
cv2.imshow("Encrypt2",Encrypt2)
cv2.imshow("Decrypt_img",Decrypt_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
