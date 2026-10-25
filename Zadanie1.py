import cv2
import matplotlib.pyplot as plt

image_path = "Lemon.jpg"

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {image_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.title("Oryginalny obraz")
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

plt.figure(figsize=(6, 4))
plt.title("Po zmniejszeniu, grayscale i obrocie o 90°")

plt.imshow(img_rotated, cmap="gray")
plt.axis("off")
plt.show()


print("Wymiary macierzy wynikowej (wysokość, szerokość):", img_rotated.shape)
print("Typ danych pikseli:", img_rotated.dtype)
print("\nMacierz obrazu (wartości pikseli 0-255):\n")
print(img_rotated)

# Jeśli chcesz podejrzeć dokładny fragment (np. wycinek 5x5 pikseli):
print("\nWycinek macierzy [pierwsze 5x5 pikseli]:\n", img_rotated[:5, :5])