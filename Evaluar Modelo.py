import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from text_to_speech import text_to_speech
import pygame

# Ajustar rutas para asegurar que se usa el directorio correcto
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))  # Ruta donde se encuentra el script actual
MODELS_PATH = os.path.join(ROOT_PATH, "modelos")
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
MAX_LENGTH_FRAMES = 15
MIN_LENGTH_FRAMES = 5

# Funciones para Mediapipe y traducción
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def evaluate_model(model, actions, threshold=0.7):
    count_frame = 0
    kp_sequence, sentence = [], []
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))

            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
            else:
                if count_frame >= MIN_LENGTH_FRAMES:
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                        print(f"Palabra detectada: {sent}")
                    count_frame = 0
                    kp_sequence = []

            cv2.rectangle(image, (0, image.shape[0] - 35), (image.shape[1], image.shape[0]), (255, 255, 255), -1)
            if sentence:
                cv2.putText(image, sentence[0], (10, image.shape[0] - 10), FONT, FONT_SIZE, (0, 0, 0))

            cv2.imshow('Traductor', image)
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break
        video.release()
        cv2.destroyAllWindows()

def select_topic():
    """Interfaz para seleccionar un tópico"""
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Seleccionar Tópico")
    font = pygame.font.Font(None, 36)

    topics = [f for f in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, f))]
    selected = 0

    running = True
    while running:
        screen.fill((30, 30, 30))
        title = font.render("Selecciona un tópico con ↑ ↓ y presiona Enter", True, (255, 255, 255))
        screen.blit(title, (50, 20))

        for i, topic in enumerate(topics):
            color = (255, 255, 255) if i == selected else (100, 100, 100)
            text = font.render(topic, True, color)
            screen.blit(text, (50, 80 + i * 40))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(topics)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(topics)
                elif event.key == pygame.K_RETURN:
                    running = False

    pygame.quit()
    return topics[selected]

def main():
    selected_topic = select_topic()
    topic_path = os.path.join(MODELS_PATH, selected_topic)
    print(f"Tópico seleccionado: {selected_topic}")

    # Cargar modelo
    model_path = os.path.join(topic_path, f"actions_{MAX_LENGTH_FRAMES}.keras")
    print(f"Cargando modelo desde: {model_path}")
    model = load_model(model_path)

    # Cargar acciones desde los datos
    actions_path = os.path.join(ROOT_PATH, "datos", selected_topic)
    actions = [f.replace(".h5", "") for f in os.listdir(actions_path) if f.endswith(".h5")]
    print(f"Acciones cargadas: {actions}")

    evaluate_model(model, actions)

if __name__ == "__main__":
    main()
