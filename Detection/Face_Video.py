import cv2
import time

# Cargar el clasificador Haar preentrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verificar si el clasificador Haar se ha cargado correctamente
if face_cascade.empty():
    print("Error al cargar el clasificador Haar.")
else:
    print("Clasificador Haar cargado correctamente.")

# Iniciar la captura de video desde la cámara.
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
else:
    print("Cámara conectada y funcionando.")

# Crear un objeto VideoWriter para grabar el video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de video
out = cv2.VideoWriter('video_detectado.avi', fourcc, 20.0, (640, 480))

# Variables para contar las fotos tomadas
photo_count = 0
max_photos = 2

# Registrar el tiempo de inicio para controlar la duración del video (5 segundos)
start_time = time.time()

# Capturar los cuadros del video en un bucle
while True:
    # Leer un cuadro del video
    ret, frame = cap.read()

    # Si no se puede leer el cuadro, se termina el bucle
    if not ret:
        print("Error al leer el cuadro del video.")
        break

    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar las caras en el cuadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un rectángulo alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        # Añadir un margen alrededor de la cara para que la foto sea más amplia
        margin = 50  # Puedes ajustar este valor según la cantidad de contexto que quieras alrededor de la cara
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, frame.shape[1] - x)
        h = min(h + 2 * margin, frame.shape[0] - y)

        # Recortar la cara con el margen
        face_image = frame[y:y + h, x:x + w]

        # Si aún no se han tomado 3 fotos, tomar la foto
        if photo_count < max_photos:
            filename = f'face_detected_{photo_count + 1}.png'  # Nombre único para cada foto
            cv2.imwrite(filename, face_image)  # Guardar la foto
            print(f"¡Cara detectada! Foto {photo_count + 1} guardada como '{filename}'.")
            photo_count += 1

        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Escribir el cuadro en el archivo de video
    out.write(frame)

    # Mostrar el cuadro con las caras detectadas
    cv2.imshow('Face Detection in Real-Time', frame)

    # Salir del bucle después de 10 segundos
    if (time.time() - start_time) >= 10:
        print("PILLADO 👌. ")
        break

    # Salir del bucle si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Después de 10 segundos, tomar las 3 fotos restantes
while photo_count < max_photos:
    # Leer un nuevo cuadro para capturar la cara
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el cuadro del video.")
        break

    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar las caras en el cuadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Si hay caras detectadas, tomar una foto
    for (x, y, w, h) in faces:
        # Añadir un margen alrededor de la cara para que la foto sea más amplia
        margin = 50  # Puedes ajustar este valor según la cantidad de contexto que quieras alrededor de la cara
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, frame.shape[1] - x)
        h = min(h + 2 * margin, frame.shape[0] - y)

        face_image = frame[y:y + h, x:x + w]  # Recortar la cara con el margen
        filename = f'face_detected_{photo_count + 1}.png'  # Nombre único para cada foto
        cv2.imwrite(filename, face_image)  # Guardar la foto
        print(f"¡Cara detectada! Foto {photo_count + 1} guardada como '{filename}'.")
        photo_count += 1

        # Romper después de tomar una foto
        if photo_count >= max_photos:
            break

# Liberar la cámara y cerrar las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()
