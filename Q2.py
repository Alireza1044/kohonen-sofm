import numpy as np
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, lr, epochs):
        self.map = np.random.uniform(0, 1, size=(40, 40, 3))
        self.initial_lr = lr
        self.initial_radius = 20
        self.epochs = epochs
        self.landa = 0
        self.compute_landa()
        self.radius = 0
        self.compute_radius()
        self.lr = 0
        self.compute_lr()
        self.influence = 0

    def compute_landa(self):
        self.landa = self.epochs / np.log(self.initial_radius)

    def compute_radius(self, epoch=0):
        self.radius = self.initial_radius * np.exp(-epoch / self.landa)

    def compute_lr(self, epoch=0):
        self.lr = self.initial_lr * np.exp(-epoch / self.landa)

    def compute_influence(self, distance):
        return np.exp(-distance / (2 * (self.radius ** 2)))

    def compute_distance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def find_BMU(self, x):
        arg_min = (-1, -1)
        dist = float("inf")
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if np.sqrt(np.sum((self.map[i][j] - x) ** 2)) < dist:
                    dist = np.sqrt(np.sum((self.map[i][j] - x) ** 2))
                    arg_min = (i, j)
        return arg_min

    def update_weights(self, minX, minY, x):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                dist = self.compute_distance(i, j, minX, minY)
                if dist < self.radius:
                    influence = self.compute_influence(dist)
                    self.map[i][j] += influence * self.lr * (x - self.map[i][j])

    def train(self, input):
        for epoch in range(self.epochs + 1):
            self.compute_radius(epoch)
            self.compute_lr(epoch)
            x = input[epoch % 1600]
            minX, minY = self.find_BMU(x)
            self.update_weights(minX, minY, x)
            print("Epoch: ", epoch / self.epochs * 100, "%")
            if epoch % 400 == 0:
                plt.subplot(2, 2, (int((epoch + 1) / 400) % 4) + 1)
                plt.axis('off')
                plt.title("Epoch " + str(epoch))
                plt.imshow(self.map)
                if epoch in [1200, 2800]:
                    plt.savefig("epoch" + str(epoch) +".png")
                    plt.show()

        plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.title("Final Result")
        plt.imshow(self.map)
        plt.savefig("final.png")
        plt.show()


def generate_data():
    data = np.ndarray((1600, 3), dtype=float)
    for i in range(len(data)):
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        data[i] = [r, g, b]
    return data


if __name__ == '__main__':
    input = generate_data()
    input = input / input.max()
    som = SOM(0.04, 3200)
    som.train(input)
