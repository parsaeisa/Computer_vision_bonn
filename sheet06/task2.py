import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans

CLIP_MARGIN = 25
BOUNDING_BOX_X_START = 70
BOUNDING_BOX_X_END = 280
BOUNDING_BOX_Y_START = 30

def pre_process_image(image):
    clipped_image = np.clip(image, CLIP_MARGIN, 255-CLIP_MARGIN)
    channels = cv.split(clipped_image)
    equalized_channels = [cv.equalizeHist(ch) for ch in channels]
    equalized_image = cv.merge(equalized_channels)

    return equalized_image

def read_image(filename):
    image = cv.imread(filename)
    image = pre_process_image(image)
    image = image / 255

    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[BOUNDING_BOX_Y_START:, BOUNDING_BOX_X_START:BOUNDING_BOX_X_END, :] = 1
    bb_width, bb_height = height - BOUNDING_BOX_Y_START, BOUNDING_BOX_X_END - BOUNDING_BOX_X_START

    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    return image, foreground, background

class GMM:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.weights = None
        self.means = None
        self.covariances = None

    def gaussian_scores(self, data):
        scores = []
        for k in range(self.n_components):
            diff = data - self.means[k]
            scores.append(
                self.weights[k]
                * np.exp(-0.5 * np.sum(diff**2 / self.covariances[k], axis=1))
                / np.sqrt((2 * np.pi) ** data.shape[1] * np.prod(self.covariances[k]))
            )
        return np.array(scores).T

    def estep(self, data):
        scores = self.gaussian_scores(data)
        r = scores / np.sum(scores, axis=1, keepdims=True)
        return r

    def mstep(self, data, r):
        Nk = np.sum(r, axis=0)
        self.weights = Nk / len(data)
        self.means = (r.T @ data) / Nk[:, np.newaxis]
        self.covariances = []
        for k in range(self.n_components):
            diff = data - self.means[k]
            self.covariances.append(
                np.sum(r[:, k][:, np.newaxis] * (diff**2), axis=0) / Nk[k]
            )
        self.covariances = np.array(self.covariances)

    def em_algorithm(self, data, n_iterations=10):
        for _ in range(n_iterations):
            r = self.estep(data)
            self.mstep(data, r)

    def kmeans_init(self, data):
        kmeans = KMeans(n_clusters=self.n_components).fit(data)
        self.means = kmeans.cluster_centers_
        self.weights = np.ones(self.n_components) / self.n_components
        self.covariances = np.ones((self.n_components, data.shape[1]))

    def probability(self, data):
        scores = self.gaussian_scores(data)
        return np.sum(scores, axis=1)

    def train(self, data):
        self.kmeans_init(data)
        self.em_algorithm(data)

image, foreground, background = read_image('person.jpg')

gmm_foreground = GMM(n_components=4)
gmm_background = GMM(n_components=4)

gmm_foreground.train(foreground)
gmm_background.train(background)

data = image.reshape((-1, 3))
background_probs = gmm_background.probability(data)

tau = 0.05
mask = background_probs.reshape(image.shape[:2]) < tau

output = np.ones_like(image, dtype=np.uint8) * 255

output[mask == False] = 0

cv.imshow('Background Subtracted', output)
cv.waitKey(0)
cv.destroyAllWindows()
