# Import the required libraries
import cv2
import numpy as np
#frame = cv2.imread("inclinado.jpg")
camera = cv2.VideoCapture(0)
camera.set(3, 320)
camera.set(4, 240)
for i in range(0,20):
    (grabbed, frame) = camera.read()

while True:
        (grabbed, frame) = camera.read()
        cv2.imshow('frame', frame)
        if (grabbed):
            yf=200
            xf=0
            height=480
            width=640
            crop = frame[yf:height, xf:width]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            canny = cv2.Canny(blur, 100, 450)
            ret, thresh = cv2.threshold(canny, 100, 100, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(crop, contours, -1, (0,255,0), 3)
            qtdContornos = 0
            DirecaoASerTomada = 0
            angles = []
            for c in contours:

                qtdContornos += 1
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                #cv2.drawContours(crop, [box], 0, (0, 0, 255), 2)

                rows, cols = crop.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 1, 1) # vx e vy vetores unitarios

                if abs(vx) < 1e-6:  # Se vx é próximo de zero
                    vx = np.array([1e-2])
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                print("lefty", lefty)
                print("righty", righty)
                print("vx", vx)
                cv2.line(crop, (cols - 1, righty), (0, lefty), (255, 0, 0), 2)

                your_line = np.array([vx, vy])
                dot_product = np.dot([0, 1], your_line)
                angle_2_x = np.arccos(dot_product/np.linalg.norm(your_line))
                degrees = np.rad2deg(angle_2_x)
                angles.append(degrees)
            if(len(angles) > 0):
                anglesMean = np.mean(angles)
                angulo_em_radianos = np.deg2rad(anglesMean)
                print("media de angulo", angulo_em_radianos)
                #cv2.line(crop, (320, 0), (320, 480), (255, 0, 255), 1)
                x_inicial = 320 + int(480 * np.cos(angulo_em_radianos)) # centro da faixa
                y_inicial = 0
                x_final = 320
                y_final = 480
                print("x_inicial", x_inicial)
                cv2.line(frame, (x_final, y_final), (x_inicial, y_inicial), (255, 255, 0), 3)
                cv2.imshow('fram1e', frame)
                cv2.imshow('canny', canny)
                cv2.imshow('blur', blur)
                cv2.imshow('crop', crop)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# After the loop release the cap object
camera.release()
# Destroy all the windows
cv2.destroyAllWindows()