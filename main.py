import cv2
import numpy as np
#cap = cv2.VideoCapture("./20240426_162822_600x600.mp4")
cap = cv2.VideoCapture("./162329-600x600.mp4")
if(cv2.__version__ != '3.4.18'):
    raise Exception("Versão do OpenCV incompatível. Utilize a versão 3.4.18.65")

def handleHoughLines(input_image, output_image):
    lines = cv2.HoughLinesP(input_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=150, maxLineGap=50)
    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            your_line = np.array([x2-x1, y2-y1])
            dot_product = np.dot([0, 1], your_line)
            angle_2_x = np.arccos(dot_product / np.linalg.norm(your_line))
            degrees = np.rad2deg(angle_2_x)
            print("degrees", degrees)
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

def processaLadoEsquerdo(roiEsquerdo):
    gray = cv2.cvtColor(roiEsquerdo, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 100, 100)
    threshold1 = 20
    threshold2 = 100
    thresholdDark = 170
    thresh = cv2.threshold(blur, thresholdDark, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('threshEsquerdo', thresh)
    edges = cv2.Canny(blur, threshold1, threshold2, apertureSize=3)
    areaThreshold = 1000
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cv2.drawContours(roiEsquerdo,cnts,-1,(255,0,255),3)
    angles = np.array([])
    for c in cnts:
        if cv2.contourArea(c) < areaThreshold:
            continue
        
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(roiEsquerdo,[box],0,(0,0,255),2)
        
        angle = rect[2]
        angle = np.deg2rad(angle)
        angles = np.append(angles, angle)
        cv2.putText(roiEsquerdo, str(np.rad2deg(angle)), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        ROIwidth, ROIheight = roiEsquerdo.shape[:2]
        x_inicial = int(ROIheight/2)+int(int(ROIheight/2) * np.mean(angles)) # centro da faixa
        y_inicial = 0
        x_final = int(ROIheight/2)
        y_final = ROIwidth
        cv2.line(roiEsquerdo, (x_inicial, y_inicial), (x_final, y_final), (255, 255, 0), 3)
    return np.mean(angles)
def processaLadoDireito(roiDireito):
    gray = cv2.cvtColor(roiDireito, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 100, 100)
    threshold1 = 20
    threshold2 = 100
    thresholdDark = 150
    thresh = cv2.threshold(blur, thresholdDark, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('threshDireito', thresh)
    #thresh = cv2.dilate(thresh, None, iterations=2)
    edges = cv2.Canny(blur, threshold1, threshold2, apertureSize=3)
    areaThreshold = 500
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cv2.drawContours(roiDireito,cnts,-1,(255,0,255),3)
    angles = np.array([])
    for c in cnts:
        if cv2.contourArea(c) < areaThreshold:
            continue
        
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(roiDireito,[box],0,(0,255,0),2)
        
        angle = -90 + rect[2]
        angle = np.deg2rad(angle)
        angles = np.append(angles, angle)
        cv2.putText(roiDireito, str(np.rad2deg(angle)), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        ROIwidth, ROIheight = roiDireito.shape[:2]
        
        x_inicial = int(ROIheight/2)+int(int(ROIheight/2) * np.sin(np.mean(angles))) # centro da faixa
        y_inicial = 0
        x_final = int(ROIheight/2)
        y_final = ROIwidth
        cv2.line(roiDireito, (x_inicial, y_inicial), (x_final, y_final), (255, 255, 0), 3)
    return np.mean(angles)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    clean = frame.copy()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    height, width = frame.shape[:2]
    startHeight = int(height/2)
    endHeight = height
    startWidth = int(0)
    endWidth = int(width/2)
    roiEsquerdo = frame[startHeight:endHeight, startWidth:width - endWidth]
    leftAngles = processaLadoEsquerdo(roiEsquerdo)
    roiDireito = frame[startHeight:endHeight, endWidth:width]
    rightAngles = processaLadoDireito(roiDireito)
    
    angulo = (leftAngles + (-rightAngles))/2
    print("angulo", np.rad2deg(angulo))
    
    x_inicial = int(height/2)+int(int(height/2) * np.sin(angulo)) # centro da faixa
    y_inicial = 0
    x_final = int(height/2)
    y_final = width
    cv2.putText(clean, str(np.rad2deg(angulo)), (int(height/2), int(width/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(clean, (x_inicial, y_inicial), (x_final, y_final), (255, 0, 255), 3)
    
    #handleHoughLines(thresh, output_image=roi)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('blur', blur)
    #cv2.imshow('edges', edges)
    cv2.imshow('clean', clean)
    cv2.imshow('roiEsquerdo', roiEsquerdo)
    cv2.imshow('roiDireito', roiDireito)
    if cv2.waitKey(100) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()