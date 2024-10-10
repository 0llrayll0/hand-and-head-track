import cv2
import os
import mediapipe as mp
import numpy as np  # Adicionando a importação do NumPy

def main():
    # Carregar o classificador Haar Cascade para detecção de rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Inicializar o mediapipe para detecção de mãos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Inicializar a captura de vídeo da câmera
    cap = cv2.VideoCapture(0)

    # Verificar se a câmera foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    # Definir a altura e a largura da janela do boneco virtual
    virtual_window_width = 300
    virtual_window_height = 400

    # Criar uma nova janela para o boneco virtual
    cv2.namedWindow('Boneco Virtual', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Boneco Virtual', virtual_window_width, virtual_window_height)

    # Loop principal
    while True:
        # Capturar um frame da câmera
        ret, frame = cap.read()

        # Verificar se o frame foi capturado corretamente
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Criar uma imagem para o boneco virtual
        virtual_image = 255 * np.ones((virtual_window_height, virtual_window_width, 3), dtype=np.uint8)

        # Converter a imagem para tons de cinza para a detecção de rostos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Para cada rosto detectado
        for (x, y, w, h) in faces:
            # Desenhar um retângulo ao redor do rosto detectado na imagem principal
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Desenhar um retângulo ao redor do rosto detectado no boneco virtual
            cv2.rectangle(virtual_image, (50, 50), (250, 350), (255, 0, 0), 2)

        # Converter a imagem para RGB para a detecção de mãos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar mãos na imagem
        results = hands.process(rgb_frame)

        # Verificar se mãos foram detectadas
        if results.multi_hand_landmarks:
            # Para cada mão detectada
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar pontos de referência das mãos na imagem principal
                for landmark in hand_landmarks.landmark:
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Desenhar pontos de referência das mãos no boneco virtual
                for landmark in hand_landmarks.landmark:
                    # Converter as coordenadas para a janela do boneco virtual
                    virtual_x = int(landmark.x * virtual_window_width)
                    virtual_y = int(landmark.y * virtual_window_height)

                    # Desenhar os pontos de referência no boneco virtual
                    cv2.circle(virtual_image, (virtual_x, virtual_y), 5, (0, 255, 0), -1)

        # Exibir o frame com os rostos e as mãos identificadas
        cv2.imshow('Video', frame)

        # Exibir o boneco virtual na segunda janela
        cv2.imshow('Boneco Virtual', virtual_image)

        # Verificar se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
