import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from sys import setrecursionlimit
from abc import ABC, abstractmethod

setrecursionlimit(10000)


class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def isPointInCircle(self, point: tuple[float, float]):
        px, py = point
        dist = ((self.x - px) ** 2 + (self.y - py) ** 2) ** 0.5
        return dist <= self.r

    @staticmethod
    def isCirclesIntersect(a: 'Circle', b: 'Circle'):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) <= (a.r + b.r) ** 2

    def __str__(self):
        return f"{self.r=}, {self.x=}, {self.y=}"


def euclid(a, b, c, d):
    return ((a - c)**2 + (b - d)**2)**0.5


class OccupancyGrid:
    def __init__(self, obstacles: list[Circle], sizeOG: int,
                 sizeMap: tuple[float, float]):
       
        self.sizeOG = sizeOG
        self.og = [[0 for _ in range(sizeOG)] for _ in range(sizeOG)]

        sizeMapX, sizeMapY = sizeMap
        self.blockSize = [sizeMapX / sizeOG, sizeMapY / sizeOG]

        self.obstacles = obstacles

        for posY in range(sizeOG):
            for posX in range(sizeOG):
                cx = (posX + 0.5) * self.blockSize[0]
                cy = (posY + 0.5) * self.blockSize[1]
                for obst in obstacles:
                    if obst.isPointInCircle((cx, cy)):
                        self.og[posY][posX] = 1
                        break

    def printMatrix(self, matrix):
        for row in matrix:
            print(''.join(str(cell) for cell in row))

    def printPath(self, path):
        copyOG = deepcopy(self.og)
        for x, y in path:
            if 0 <= y < self.sizeOG and 0 <= x < self.sizeOG:
                copyOG[y][x] = "P"
        self.printMatrix(copyOG)


    def boundingCircleForPoint(self, point: tuple[int, int]) -> Circle | None:
        px = (point[0] + 0.5) * self.blockSize[0]
        py = (point[1] + 0.5) * self.blockSize[1]

        sizeMapX = self.sizeOG * self.blockSize[0]
        sizeMapY = self.sizeOG * self.blockSize[1]

        best_circle = Circle(0, 0, 0)
        max_r = 0.0

        for row in range(self.sizeOG):          
            for col in range(self.sizeOG):       
                cx = col * self.blockSize[0]     
                cy = row * self.blockSize[1]     

                dist_to_point = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

                l = dist_to_point              
                r = self.sizeOG * max(self.blockSize) + 100.0 

                best_local = 0.0
                while abs(r - l) > 1e-6:
                    radius = (l + r) / 2.0

                    if (cx - radius < 0 or cx + radius > sizeMapX or
                        cy - radius < 0 or cy + radius > sizeMapY):
                        r = radius
                        continue

                    pred_circle = Circle(cx, cy, radius)
                    good = True
                    for obs in self.obstacles:
                        if Circle.isCirclesIntersect(obs, pred_circle):
                            good = False
                            break
                       

                    if good:
                        l = radius
                        best_local = radius
                    else:
                        r = radius

                if best_local > max_r + 1e-9:
                    max_r = best_local
                    best_circle = Circle(cx, cy, best_local)

        return best_circle

    def boundingCircles(self, path: list[int, int]) -> list[Circle]:
        result = []
        for cell in path:
            result.append(self.boundingCircleForPoint(cell))
        return result

    def visualize(self, path, bcircles):
        """Рисует препятствия, путь (точки) и найденные круги."""
        fig, ax = plt.subplots(figsize=(10, 10))

        for i in range(self.sizeOG + 1):
            ax.axhline(i * self.blockSize[1], color='lightgray', linewidth=0.5)
            ax.axvline(i * self.blockSize[0], color='lightgray', linewidth=0.5)
        for y in range(self.sizeOG):
            for x in range(self.sizeOG):
                if self.og[y][x] == 1:
                    rect = plt.Rectangle(
                        (x * self.blockSize[0], y * self.blockSize[1]),
                        self.blockSize[0], self.blockSize[1],
                        facecolor='red', alpha=0.3, edgecolor='darkred'
                    )
                    ax.add_patch(rect)

        for obs in self.obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.r, color='red', alpha=0.5, label='Препятствия')
            ax.add_patch(circle)
            ax.plot(obs.x, obs.y, 'ro', markersize=4)

        if path:
            phys_points = [((p[0] + 0.5) * self.blockSize[0], (p[1] + 0.5) * self.blockSize[1]) for p in path]
            xs, ys = zip(*phys_points)
            ax.plot(xs, ys, 'b-', linewidth=2, label='Путь')
            ax.scatter(xs, ys, color='blue', s=50, zorder=5, label='Точки пути')

        if bcircles:
            for i, circ in enumerate(bcircles):
                if circ is None:
                    continue
                circle_patch = plt.Circle(
                    (circ.x, circ.y), circ.r,
                    color='green', fill=False, linestyle='--', linewidth=2,
                    label='Круги' if i == 0 else ""
                )
                ax.add_patch(circle_patch)
                ax.plot(circ.x, circ.y, 'g+', markersize=10)
                ax.annotate(
                    f'{circ.r:.2f}',
                    (circ.x, circ.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )

        ax.set_xlim(0, self.sizeOG * self.blockSize[0])
        ax.set_ylim(0, self.sizeOG * self.blockSize[1])
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Визуализация препятствий, пути и ограничивающих кругов')
        ax.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()



if __name__ == "__main__":

    obstacles = [Circle(1.7, 1, 0.19), Circle(1.7, 1, 0.14)]

    og = OccupancyGrid(obstacles, 5, (2, 2))

    path = og.getInitPath((0, 0), (1.2, 1.3))
    og.printPath(path)

    boundingCircles = og.boundingCircles(path)

    for i in boundingCircles:
        print(str(i))

    og.visualize(path, boundingCircles)