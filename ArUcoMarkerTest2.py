import cv2
from cv2 import aruco
import numpy as np
import pdb

def findRect(contour):
    epsilon = cv2.arcLength(contour, True) * 0.02
    rect = cv2.approxPolyDP(contour, epsilon, True)
    if len(rect) == 4:
        return rect
    else:
        return [0]
def grid2Bits(grids):
    return ["".join(map(str, grids[b])) for b in range(len(grids))]
def distance(a,b):
    distance = 0
    if len(a) != len(b):
        print('Error: Dimensios of two data must be equal!')
        return
    else:
        for i in range(len(a)):
            distance += (a[i]-b[i])**2
    return distance

def findID(bytes):

    min_dist = 100
    marker_id = -1
    rot_id = -1
    for id in range(1000):
        #aruco_marker = aruco.drawMarker(dictionary=aruco_dict, id=200, sidePixels= 8, borderBits=1)
        #temp = aruco_marker[1:7,1:7]
        temp = aruco_dict_bytes[id].reshape(4,5)
        for r in range(4):
            dist_tmp = distance(temp[r],bytes)
            if dist_tmp < min_dist:
                min_dist = dist_tmp
                marker_id = id
                rot_id = r
        if min_dist == 0:
            break
    return marker_id, rot_id

def detection(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    #_, bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU) # Binarization => cv2.adaptiveThreshold    
    bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 2) # Binarization => cv2.adaptiveThreshold

    ctr, _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)# Contour Detection
    rects = []
    rects_aruco = []

    grid_matrices = []
    chk4marker = [False] * 4
    rot4marker = [-1] * 4
    coord4marker = [[] for _ in range(4)]
    for contour in ctr:
        rect = findRect(contour)    
#        print(rect)
        if len(rect) == 4:
            area = cv2.contourArea(rect)
         #   print(h, w, area)
            rect_tmp = []
            
            for i in range(4):
                rect_tmp.append(rect[i][0])
            rect_tmp = np.array(rect_tmp)
            if area > 0.002 * (h*w) and area < 0.5 *(h*w):
               # print([rect])
               # cv2.drawContours(frame, [rect], -1, (0,255,0), 1)
               # print(rect)
                #print([rect])
                sm = rect_tmp.sum(axis = 1)
                dif = np.diff(rect_tmp, axis=1)
                rects.append(np.float32([rect[np.argmin(sm)], rect[np.argmin(dif)], rect[np.argmax(sm)], rect[np.argmax(dif)]]))

            #    cv2.imshow('contour',frame)
    #print(rects)
    cnt_ar = 0
    for r in rects: # Perspective Transform
        marker__bytes = []
        marker_bits = []
        trnsmat = cv2.getPerspectiveTransform(r, rectcoord) 
        marker = cv2.warpPerspective(img, trnsmat, (length_mark, length_mark))
        _, bin2 = cv2.threshold(marker, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        grid = np.zeros((8,8), dtype=int) # Marker Griding
       # grid6 = np.zeros((6,6), dtype=int)
        boundary_white = False
        for y in range(0, length_mark, 10):
            for x in range(0, length_mark, 10):
                num_white = 0
                for yy in range(10):
                    for xx in range(10):
                         if bin2[y+yy][x+xx] == 255:
                             num_white += 1
                if num_white > 50:
                    if y == 0 or y == length_mark-10 or x == 0 or x == length_mark-1:
                        boundary_white = True
                        break
                    grid[int(y/10)][int(x/10)] = 1
            
            if boundary_white:
                break
        if boundary_white:
            continue
        grid_matrices.append(grid[1:7,1:7])
        #print(grid6)
        # Marker ID and Coner    
        marker_bits = grid2Bits(grid[1:7,1:7])
        marker_bits_full = str("".join(map(str, marker_bits)))
       # print(marker_bits)
        marker__bytes = [int(str('0b'+marker_bits_full[b:b+8]), 2) for b in range(0, len(marker_bits_full), 8)]
        marker_id, rot_id = findID(marker__bytes)
        if marker_id >= 0:
            
   #     print('ch', r, r[2][0])
            if rot_id == 0:
                corner = r[0][0]
                c1 = corner
                c2  = r[1][0]
                c3  = r[2][0]
                c4  = r[3][0]
                cv2.rectangle(frame, (int(corner[0]-2), int(corner[1]-2)), (int(corner[0]+2), int(corner[1]+2)), (0,0,255), 2)        
                rects_aruco.append(([[[c1[0], c1[1]]],[[c2[0], c2[1]]],[[c3[0], c3[1]]],[[c4[0], c4[1]]]]))
            elif rot_id == 1:
                corner = r[3][0]   
                c1 = corner
                c2  = r[0][0]
                c3  = r[1][0]
                c4  = r[2][0]
                cv2.rectangle(frame, (int(corner[0]-2), int(corner[1]-2)), (int(corner[0]+2), int(corner[1]+2)), (0,0,255), 2)              
                rects_aruco.append(([[[c1[0], c1[1]]],[[c2[0], c2[1]]],[[c3[0], c3[1]]],[[c4[0], c4[1]]]]))

            elif rot_id == 2:
                corner = r[2][0] 
                c1 = corner
                c2  = r[3][0]
                c3  = r[0][0]
                c4  = r[1][0]
                cv2.rectangle(frame, (int(corner[0]-2), int(corner[1]-2)), (int(corner[0]+2), int(corner[1]+2)), (0,0,255), 2)
                rects_aruco.append(([[[c1[0], c1[1]]],[[c2[0], c2[1]]],[[c3[0], c3[1]]],[[c4[0], c4[1]]]]))

            elif rot_id == 3:
                corner = r[1][0]
                c1 = corner
                c2  = r[2][0]
                c3  = r[3][0]
                c4  = r[0][0]
                cv2.rectangle(frame, (int(corner[0]-2), int(corner[1]-2)), (int(corner[0]+2), int(corner[1]+2)), (0,0,255), 2)
                rects_aruco.append(([[[c1[0], c1[1]]],[[c2[0], c2[1]]],[[c3[0], c3[1]]],[[c4[0], c4[1]]]]))
            center_x, center_y = 0, 0
            for ri in range(4):
                for x, y in r[ri]:
                    center_x += x
                    center_y += y
            center_y /= 4
            center_x /= 4
            #print(center_x, center_y)
          #  cv2.putText(frame, str('id=' + str(marker_id)), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 0), 2, cv2.LINE_8)
            tmp_aruco = np.array(rects_aruco[cnt_ar], dtype=np.int32)
            if marker_id == 0:
                chk4marker[0] = True
                rot4marker[0] = rot_id
                coord4marker[0] = c1
            elif marker_id == 4:
                chk4marker[1] = True
                rot4marker[1] = rot_id
                coord4marker[1] = c2
            elif marker_id == 34:
                chk4marker[2] = True
                rot4marker[2] = rot_id
                coord4marker[2] = c3
            elif marker_id == 30:
                chk4marker[3] = True
                rot4marker[3] = rot_id
                coord4marker[3] = c4
            print(chk4marker, rot4marker, coord4marker)
            #print('aruco:',tmp_aruco)
          #  cv2.drawContours(frame, [tmp_aruco], -1, (0,255,0), 1)
            cnt_ar += 1
            #print(marker_id, rot_id)
    rects_aruco = np.array(rects_aruco, dtype=np.float64)
    print(chk4marker, rot4marker, coord4marker)
    if rot4marker[0] == rot4marker[1] and rot4marker[2] == rot4marker[3] and rot4marker[1] == rot4marker[2] and False not in chk4marker:
        if checkClockwise(coord4marker, rot4marker[0]):
            frame = veilImage(frame, drawing, coord4marker, drawing_coord)
    #if rects_aruco.ndim == 4:
    #    rects2 = np.transpose(rects_aruco, (0,2,1,3))
    #    frame = poseEstimation(frame, rects2)
      #  print(rects2, corners)
      #  print(rects2.shape,corners2.shape)
    #rects2 = rects[:][:][0][0]
    #print(rects2.shape, rects2)
    cv2.imshow('Frame',frame)
#    cv2.waitKey(30)
    return frame

def poseEstimation(img, corners):
    rot = np.array(camMat['mat'])
    #print(rot)
    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, length_mark, rot, camMat['dist'])
    for cnt in range(rvecs.shape[0]):
        img	= aruco.drawAxis(img, camMat['mat'], camMat['dist'], rvecs[cnt], tvecs[cnt], 50)
    return img

def checkClockwise(coord, rot):
    mean_x = (coord[0][0] + coord[1][0] + coord[2][0] + coord[3][0])/4
    mean_y = (coord[0][1] + coord[1][1] + coord[2][1] + coord[3][1])/4
    if rot == 0:
        if coord[0][0] < mean_x and coord[3][0] < mean_x and coord[1][0] > mean_x and coord[2][0] > mean_x and coord[0][1] < mean_y and coord[3][1] > mean_y and coord[1][1] < mean_y and coord[2][1] > mean_y:
            return True
    if rot == 1:
        if coord[0][0] < mean_x and coord[3][0] > mean_x and coord[1][0] < mean_x and coord[2][0] > mean_x and coord[0][1] > mean_y and coord[3][1] > mean_y and coord[1][1] < mean_y and coord[2][1] < mean_y:
            return True
    if rot == 2:
        if coord[0][0] > mean_x and coord[3][0] > mean_x and coord[1][0] < mean_x and coord[2][0] < mean_x and coord[0][1] > mean_y and coord[3][1] < mean_y and coord[1][1] > mean_y and coord[2][1] < mean_y:
            return True
    if rot == 3:
        if coord[0][0] > mean_x and coord[3][0] < mean_x and coord[1][0] > mean_x and coord[2][0] < mean_x and coord[0][1] < mean_y and coord[3][1] < mean_y and coord[1][1] > mean_y and coord[2][1] > mean_y:
            return True
    return False

def veilImage(frm, img, coord, drawing_coord):
    print(drawing_coord, coord)
    trnsmat = cv2.getPerspectiveTransform(drawing_coord, np.array(coord,dtype=np.float32)) 
    marker = cv2.warpPerspective(img, trnsmat, (w, h))
    mask = np.zeros(frm.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask,np.array(coord,dtype=np.int32), (255,255,255))
    mask = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(frm, frm, mask=mask)
    frm = cv2.bitwise_or(marker, masked_image)
    #cv2.imshow('affine', frm)
    return frm


length_mark = 80
rectcoord = np.float32([[0, 0], [length_mark-1, 0], [length_mark-1, length_mark-1], [0, length_mark-1]])
drawing = cv2.imread('test_drawing.jpg')
img_h, img_w, _ = drawing.shape
drawing_coord = np.float32([[0, 0], [img_w-1, 0], [img_w-1, img_h-1], [0, img_h-1]])
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
aruco_marker = aruco.drawMarker(dictionary=aruco_dict, id=200, sidePixels= 8, borderBits=1)
aruco_dict_bytes = aruco_dict.bytesList

aruco_param = aruco.DetectorParameters_create()
camMat = np.load('camMatrix.npz')
cap = cv2.VideoCapture(0)
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output2.mp4', fourcc, fps/3, (w, h))

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        frame_new = detection(frame)
        cv2.imshow('Frame',frame_new)
        out.write(frame_new)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
