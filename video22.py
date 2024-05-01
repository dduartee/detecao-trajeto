import cv2
import numpy as np
#cap = cv2.VideoCapture("./20240426_162822_600x600.mp4")
cap = cv2.VideoCapture("./162329-600x600.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
if(cv2.__version__ != '3.4.18'):
    raise Exception("Versão do OpenCV incompatível. Utilize a versão 3.4.18.65")
def tratarRoi(roi, thresholdDark):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 100, 100)
    thresh = cv2.threshold(blur, thresholdDark, 255, cv2.THRESH_BINARY)[1]
    return thresh

"""
A representação pode não estar correta, por conta da limitação do seno, sei lá
"""
def representacaoLinhaAngulo(angle, width, height, output_image, color=(255, 255, 0)):
    x_inicial = int(width/2)+int(int(width/2) * np.sin(angle)) # cateto oposto
    y_inicial = 0
    x_final = int(width/2) # centro do eixo x
    y_final = height
    cv2.line(output_image, (x_inicial, y_inicial), (x_final, y_final), color, 3)
    cv2.putText(output_image, str(round(np.rad2deg(angle), 0)), (int(width/4), int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
def handleHoughLines(input_image, minHoughLineLengthValue, maxLineGapValue, output_image):
    #cv2.imshow('thresh'+str(minHoughLineLengthValue), input_image)
    #cv2.waitKey(0)
    lines = cv2.HoughLinesP(input_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=minHoughLineLengthValue, maxLineGap=maxLineGapValue)
    #  limit to 20 lines
    if lines is not None and len(lines) > 40:
        lines = lines[:40]
    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            your_line = np.array([x2-x1, y2-y1])
            dot_product = np.dot([0, 1], your_line)
            angle_2_x = np.arccos(dot_product / np.linalg.norm(your_line))
            #print("angle_2_x", np.rad2deg(angle_2_x))
            if(angle_2_x > np.pi/2):
                angle_2_x = np.pi - angle_2_x
            angles.append(-angle_2_x)
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(output_image, str(np.rad2deg(angle_2_x)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    if(np.isnan(np.mean(angles)) or len(angles) == 0):
        return None
    ROIwidth, ROIheight = output_image.shape[:2]
    representacaoLinhaAngulo(np.mean(angles), ROIwidth, ROIheight, output_image, color=(0, 255, 0))
    return np.mean(angles)

def processaLadoEsquerdo(thresh, areaThreshold, output_image):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    #cv2.drawContours(roiEsquerdo,cnts,-1,(255,0,255),3)
    angles = np.array([])
    height, width = output_image.shape[:2]
    # desenhando margem para os contornos
    heightMargin = 5
    widthMargin = 5
    cv2.line(output_image, (0, heightMargin), (width, heightMargin), (255, 0, 0), 3)
    cv2.line(output_image, (widthMargin, 0), (widthMargin, height), (255, 0, 0), 3)
    cv2.line(output_image, (width-widthMargin, 0), (width-widthMargin, height), (255, 0, 0), 3)
    cv2.line(output_image, (0, height-heightMargin), (width, height-heightMargin), (255, 0, 0), 3)
    #contador = 0
    for c in cnts:
        #contador += 1
        #print("area"+str(contador), cv2.contourArea(c))
        if cv2.contourArea(c) < areaThreshold:
            continue
        
        rect = cv2.minAreaRect(c)
        cv2.circle(output_image, (int(rect[0][0]), int(rect[0][1])), 20, (0, 255, 0), -1)
        
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # verifica se os pontos estão dentro da margem
        valid = all([point[0] > widthMargin and point[0] < width-widthMargin and point[1] > heightMargin and point[1] < height-heightMargin for point in box])
        
        if not valid:
            cv2.drawContours(output_image,[box],0,(0,0,255),2) # contorno vermelho para invalidos
            continue
        cv2.drawContours(output_image,[box],0,(0,255,0),2)  # contorno verde para validos
        
        for vertex in box:
            cv2.circle(output_image, (vertex[0], vertex[1]), 5, (0, 255, 0), -1) # pontos da caixa
            
        cv2.circle(output_image, (box[0][0], box[0][1]), 5, (0, 0, 0), -1) # ponto de origem
        
        angle = rect[2] # angulo do retangulo segue a orientacao aqui https://i0.wp.com/theailearner.com/wp-content/uploads/2020/11/movie1-1.gif de https://theailearner.com/2020/11/03/opencv-minimum-area-rectangle/
        
        angle = np.deg2rad(angle)
        angles = np.append(angles, angle)
        
        cv2.putText(output_image, str(np.rad2deg(angle)), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    anglesMean = np.mean(angles)
    if(np.isnan(anglesMean) or len(angles) == 0):
        return None
    
    representacaoLinhaAngulo(anglesMean, width, height, output_image)
    return anglesMean

def processaLadoDireito(roiDireito):
    gray = cv2.cvtColor(roiDireito, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 100, 100)
    threshold1 = 20
    threshold2 = 100
    thresholdDark = 150
    thresh = cv2.threshold(blur, thresholdDark, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('threshDireito', thresh)
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
        
        ROIheight, ROIwidth= roiDireito.shape[:2]
        
        representacaoLinhaAngulo(angle, ROIwidth, ROIheight, roiDireito)
    return np.mean(angles)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    clean = frame.copy()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    height, width = frame.shape[:2]
    
    # cortes para a região de interesse ROI (de cima para baixo) e (da esquerda para a direita)
    startHeight = int(height/3)
    endHeight = height
    startWidth = int(0)
    endWidth = int(width/2)
    
    roiEsquerdo = frame[startHeight:endHeight, startWidth:width - endWidth]
    roiDireito = frame[startHeight:endHeight, endWidth:width]
    
    thresholdDarkEsquerdo = 170 # valores impiricos
    thresholdDarkDireito = 180 # valores impiricos
    
    threshRoiEsquerdo = tratarRoi(roiEsquerdo, thresholdDarkEsquerdo)
    threshRoiDireito = tratarRoi(roiDireito, thresholdDarkDireito)
    
    # imagens para debug
    cv2.imshow('threshRoiEsquerdo', threshRoiEsquerdo)
    cv2.imshow('threshRoiDireito', threshRoiDireito)
    
    leftAngles = processaLadoEsquerdo(threshRoiEsquerdo, areaThreshold=1000, output_image=roiEsquerdo) # encontra os angulos usando contornos
    rightAngles = handleHoughLines(threshRoiDireito, minHoughLineLengthValue=150, maxLineGapValue=20, output_image=roiDireito) # encontra os angulos usando houghlines
    
    # caso não há nenhum angulo encontrado, o valor é None
    if(leftAngles != None and rightAngles != None): 
        angulo = leftAngles + rightAngles
        representacaoLinhaAngulo(leftAngles, width/2, height, clean, color=(0, 255, 255))
        representacaoLinhaAngulo(rightAngles, width*2, height, clean, color=(255, 255, 0)) 
        
        representacaoLinhaAngulo(angulo, width, height, clean, color=(255, 0, 255)) # linha de direção
        representacaoLinhaAngulo(0, width, height, clean, color=(255, 255, 255)) # linha de referencia
        
    # exibição das imagens
    cv2.imshow('clean', clean)
    cv2.imshow('roiDireito', roiDireito)
    cv2.imshow('roiEsquerdo', roiEsquerdo)
    
    if cv2.waitKey(0) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()