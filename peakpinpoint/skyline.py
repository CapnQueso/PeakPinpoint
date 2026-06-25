import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    
    return image

def detect_edges(image):
    # convert the image to edge map using Canny edge detection provided by OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# def extract_skyline(edges):
#     height, width = edges.shape
#     skyline = []

#     for x in range(width):
#         column = edges[:, x]
#         # find edge pixels
#         edge_locations = np.where(column > 0)[0]

#         if len(edge_locations) > 0:
#             skyline.append(edge_locations[0])
#         else:
#             skyline.append(height)
#     return np.array(skyline)

def extract_candidate_ridges(edges):
    # finds candidate ridge contours and returns a list of said contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []

    height, width = edges.shape

    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        score = w * 2 - y + area * 0.01

        candidates.append({"contour": contour, "score": score, "x": x, "y": y, "w": w, "h": h, "area": area})
    
    candidates.sort(key=lambda c: c["score"], reverse=True)

    return candidates[:20]

def draw_candidates(image, candidates):
    output = image.copy()
    for candidate in candidates:
        cv2.drawContours(output, [candidate["contour"]], -1, (0, 0, 255), 2)
    return output

def draw_skyline(image, skyline):
    output = image.copy()
    for x, y in enumerate(skyline):
        cv2.circle(output, (x, int(y)), 1, (0, 0, 255), -1)
    return output