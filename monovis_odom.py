from distutils.log import debug
from math import atan2, factorial
import cv2
import numpy as np
from timeit import default_timer as timer

def sharpener(image):
  blurred = cv2.GaussianBlur(image, (5, 5), 1, 1)
  weighed = cv2.addWeighted(image, 1.5, blurred, -0.5, 0, image)
  return weighed

def getFeaturesHarris(image):
  sharpened = sharpener(image)
  gray = cv2.cvtColor(sharpened,cv2.COLOR_BGR2GRAY) 
  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,2,3,0.04)
  dst = cv2.dilate(dst,None)
  # image[dst>0.01*dst.max()]=[0,0,0]
  cv2.imshow('Gray', dst) 
  cv2.waitKey(0)
  
def getFeaturesORB(image):
  orb = cv2.ORB_create(nfeatures=200)
  kp = orb.detect(image, None)
  kp, des = orb.compute(image, kp)
  image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
  cv2.imshow('Surf', image) 
  cv2.waitKey(0)

def matchFeaturesORB(image1, image2):
  orb = cv2.ORB_create(nfeatures=500)
  kp1, des1 = orb.detectAndCompute(sharpie1, None)
  kp2, des2 = orb.detectAndCompute(sharpie2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  matches = sorted(matches, key=lambda x: x.distance)
  match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches, None)
  cv2.imshow('original image', image1)
  cv2.imshow('test image', image2)
  cv2.imshow('Matches', match_img)
  cv2.waitKey(0)

def normalize(dataset):
  mx, my = getCentroid(dataset)
  norm, dist = getDistance(dataset, [mx, my])
  normalization_matrix = np.array([[1/norm, 0, -mx/norm  ],
                                  [0, 1/norm, -my/norm  ],
                                  [0,      0,        1  ]])
  
  ones_col = np.ones([len(dataset), 1])
  dataset = np.append(dataset, ones_col, axis=1)
  
  normalized = np.matmul(dataset, normalization_matrix)
  return normalized
    
def getCentroid(matched_points):
  mx, my = 0, 0
  for point in matched_points:
    mx += point[0]
    my += point[1]
  num_of_points = len(matched_points)
  mx /= num_of_points
  my /= num_of_points
  return mx, my 
  
def getDistance(matched_points, centroid):
  dst = np.zeros([len(matched_points),1])
  norm = 0 
  for i, point in enumerate(matched_points):
    dst[i] = ((point[0]-centroid[0])**2 + (point[1]-centroid[1])**2) **0.5
    norm += ((point[0]-centroid[0])**2 + (point[1]-centroid[1])**2) **0.5
  norm /= len(dst)  
  return norm, dst    

def getVelocity(p1, p2, t):
  velx = (p1[0] - p2[0]) / t
  vely = (p1[1] - p2[1]) / t
  return velx, vely

def getAngVel(p1, p2, t):
  return atan2(p1[0]-p2[0], p1[1] - p2[1]) / t
  

def filterRANSAC(dataset, iterations, samplesize, threshold, minpoints):
  bestscore = 0
  bestfit = [0, 0]
  bestpoints = []
  # iterations = factorial(iterations)
  # while(bestscore < minpoints):
  for iteration in range(0, iterations):
    indexes = np.random.randint(0, len(dataset)-1, size=samplesize)
    passingpoints = np.empty([0, 2])
    othersset = dataset
    sampleset = np.zeros([samplesize, 2])
    for i, index in enumerate(indexes):
      sampleset[i] = dataset[index]
      np.delete(othersset, index)
      
    b, k = fitLSR(sampleset)
    
    inliers = 0
    for i, point in enumerate(othersset): 
      # A, B, C = 1, -1/k, b/k
      # d = abs(A*point[0] + B*point[1] + C) / (A**2 + B**2)**0.5
      d = abs(point[1] - point[0]*k - b)
      if(d < threshold): 
        inliers += 1
        passingpoints = np.vstack((passingpoints, point))
        
    if(inliers > bestscore):
      bestscore = inliers
      bestfit = [b, k]
      bestpoints = np.vstack((passingpoints, sampleset))
      if(bestscore >= minpoints):
        break
          
  if(bestscore < minpoints):
    raise ValueError("RANSAC no solution found")
    
  bestfit = fitLSR(bestpoints)
  
  return bestscore, bestfit, bestpoints
  
def fitLSR(dataset):
  x_set = np.zeros([len(dataset), 1])
  y_set = np.zeros([len(dataset), 1])
  for i, data in enumerate(dataset):
    x_set[i] = data[0]
    y_set[i] = data[1]
  
  ones_mat = np.ones([len(x_set), 1])
  design_mat = np.hstack((ones_mat, x_set))
  
  xx = np.linalg.inv(np.matmul(np.transpose(design_mat), design_mat))
  xy = np.matmul(np.transpose(design_mat), y_set)
  
  b0, b1 = np.matmul(xx, xy)
  return b0, b1
          
def getAverage(dataset):
  sum = 0
  for data in dataset:
    sum += data[1]
  avg = sum/len(dataset)
  return avg

total_time = 0
counter = 0
last_time = 0
time_delta = 0 

velocities_x = []
velocities_y = [] 
velocities_ang = []

cap = cv2.VideoCapture('marsh.avi')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
ret, second_img = cap.read()
last_time = cap.get(cv2.CAP_PROP_POS_MSEC)
cv2.imshow("Frame",second_img)
cv2.waitKey(0)

while(counter <= length):
  counter += 1
  ret, frame = cap.read()
  
  if(ret and counter!= 0):
    first_img = second_img
    second_img = frame
    cv2.imshow("Frame",second_img)
    cv2.waitKey(1)
    time_delta = 1/30 * 4
    # time_delta = (cap.get(cv2.CAP_PROP_POS_MSEC) - last_time) * 1000
    # last_time = cap.get(cv2.CAP_PROP_POS_MSEC) * 1000
    print("\n-------Loop", int(counter/4), "------")
  else:
    continue

  start = timer()
  gray1 = cv2.Laplacian(first_img, None, 1, 7)
  gray2 = cv2.Laplacian(second_img, None, 1, 7)
  end = timer()  
  total_time += end - start
  print('Laplacian: ', end - start)

  start = timer()
  orb = cv2.ORB_create(nfeatures=500)
  kp1, des1 = orb.detectAndCompute(gray1, None)
  kp2, des2 = orb.detectAndCompute(gray2, None)
  end = timer()  
  total_time += end - start
  print('ORB: ', end - start)

  start = timer()
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  matched_img = cv2.drawMatches(gray1, kp1, gray2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  cv2.imshow("Matches", matched_img)
  cv2.waitKey(1)
  end = timer()  
  total_time += end - start
  print('Matcher: ', end - start)

  start = timer()
  set1 = np.zeros([len(matches), 2])
  set2 = np.zeros([len(matches), 2])
  kp1 = np.array(kp1)
  kp2 = np.array(kp2)
  for i, matched_point in enumerate(matches):
    set1[i] = kp1[matched_point.queryIdx].pt
    set2[i] = kp2[matched_point.trainIdx].pt
  end = timer()  
  total_time += end - start
  print('Arrays filtering: ', end - start)

  start = timer()
  velocity = np.zeros_like(set1)
  for i in range(0, len(set1)):
    velocity[i] = getVelocity(set1[i], set2[i], time_delta)

  velocityAng = np.zeros([len(set1), 1])
  for i in range(0, len(set1)):
    velocityAng[i] = getAngVel(set1[i], set2[i], time_delta)
    
  xvels, yvels = [], []
  for vel in velocity:
    xvels.append(vel[0])
    yvels.append(vel[1])
  points_rangex = [x for x in range(0, len(set1))]
  
  xvels_set = list(zip(points_rangex, xvels))
  yvels_set = list(zip(points_rangex, yvels))
  AngVel_set = list(zip(points_rangex, velocityAng))
  
  end = timer()  
  total_time += end - start
  print('Speed calculations: ', end - start)

  iters, size, delta, min = 1000, 2, 1, 50
  start =  timer()
  try:
    score, fit, xvels_filtered = filterRANSAC(xvels_set, iters, size, delta, min)
    velocities_x.append(getAverage(xvels_filtered))
    print('X Result:', getAverage(xvels_filtered))
  except:
    print('X not found. Result:', getAverage(xvels_set))
    velocities_x.append(getAverage(xvels_set))
    
  try:
    score, fit, yvels_filtered = filterRANSAC(yvels_set, iters, size, delta, min)
    velocities_y.append(getAverage(yvels_filtered))
    print('Y Result:', getAverage(yvels_set))
  except:
    print('Y not found. Result:', getAverage(yvels_set))
    velocities_y.append(getAverage(yvels_set))
  
  try:
    score, fit, angVel_filtered = filterRANSAC(AngVel_set, iters, size, delta, min)
    velocities_ang.append(getAverage(angVel_filtered))
    print('ANG Result:', getAverage(angVel_filtered))
  except:
    print('ANG not found. Result:', getAverage(AngVel_set))
    velocities_ang.append(getAverage(AngVel_set))
    
  end = timer()  
  total_time += end - start
  print('3 RANSACs: ', end - start)

cap.release()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt

points_rangex = [x for x in range(0, len(velocities_x))]
points_rangey = [x for x in range(0, len(velocities_y))]
points_rangeang = [x for x in range(0, len(velocities_ang))]

plt.figure(1)
plt.subplot(211)
plt.scatter(points_rangex, velocities_x, color='r')
plt.subplot(212)
plt.scatter(points_rangey, velocities_y, color='b')
plt.figure(2)
plt.scatter(points_rangey, velocities_ang, color='g')

# plt.figure(2)
# plt.scatter(points_rangeang, velocities_ang, color='g')
plt.show()

print('Total: ', total_time)
print('Time to cut: ', total_time - 1/30)