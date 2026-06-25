import cv2
import matplotlib.pyplot as plt

from peakpinpoint.skyline import load_image, detect_edges, extract_skyline, draw_skyline

image = load_image("images/motorbike.jpg")

edges = detect_edges(image)

skyline = extract_skyline(edges)

result = draw_skyline(image, skyline)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.title("PeakPinpoint Skyline Detection Test v0.1")
plt.axis("off")
plt.show()