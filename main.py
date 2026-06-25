import cv2
import matplotlib.pyplot as plt

from peakpinpoint.skyline import load_image, detect_edges, extract_candidate_ridges, draw_candidates

image = load_image("images/motorbike.jpg")

edges = detect_edges(image)

#cv2.imwrite("edges.png", edges)

candidates = extract_candidate_ridges(edges)

result = draw_candidates(image, candidates)

print()
for i, candidate in enumerate(candidates[:10]):
    print(f"{i+1}: " f"score={candidate['score']:.2f}" f" x={candidate['x']}" f" y={candidate['y']}" f" w={candidate['w']}" f" h={candidate['h']}" f" area={candidate['area']:.2f}")

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.title("PeakPinpoint Skyline Detection Test v0.2.1")
plt.axis("off")
plt.show()