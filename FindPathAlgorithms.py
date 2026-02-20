from abc import abstractmethod, ABC
from OccupancyGrid import OccupancyGrid
from collections import defaultdict
import heapq
from math import inf


class FindPathAlgorithm(ABC):
    
    @abstractmethod
    def getP(self, grid: OccupancyGrid, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        pass


class DFS(FindPathAlgorithm):

    def getP(self, grid: OccupancyGrid, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:

        self.grid = grid
        self._start_x, self._start_y = tuple(int(i // grid.blockSize[0]) for i in list(start))
        self._end_x, self._end_y = tuple(int(i // grid.blockSize[1]) for i in list(end))
        if grid.og[self._start_y][self._start_x] == 1:
            print("Старт в препятствии!")
            return []
        if grid.og[self._end_y][self._end_x] == 1:
            print("Цель в препятствии!")  
            return []

        for depth in range(grid.sizeOG * grid.sizeOG):  
            self._visited = defaultdict(lambda: 0)
            self._path = []
            if self._dfs_limited((self._start_x, self._start_y), depth):
                self._path = list(reversed(self._path))
                return self._path
        return []  

    def _dfs_limited(self, v: tuple[int, int], limit: int) -> bool:
        
        x, y = v
        self._visited[v] = 1

        if v == (self._end_x, self._end_y):
            self._path.append(v)
            return True

        if limit <= 0:
            return False

        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.sizeOG and 0 <= ny < self.grid.sizeOG:
                if self._visited[(nx, ny)] == 0 and self.grid.og[ny][nx] != 1:
                    if self._dfs_limited((nx, ny), limit - 1):
                        self._path.append(v)
                        return True
        return False


class AStar(FindPathAlgorithm):
    def heuristic(self, p1: tuple[int, int], p2: tuple[int, int]) -> int:
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    def pathRecovery(self, parent, v1):
        path = []
        while v1 != -1:
            path.append(v1)
            v1 = parent[v1]
        return list(reversed(path))

    def getP(self, grid: OccupancyGrid, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[int, int]]:
        start_idx = (int(start[0] // grid.blockSize[0]), int(start[1] // grid.blockSize[1]))
        end_idx = (int(end[0] // grid.blockSize[0]), int(end[1] // grid.blockSize[1]))

        if grid.og[start_idx[1]][start_idx[0]] == 1:
            print("Старт в препятствии!")
            return []
        if grid.og[end_idx[1]][end_idx[0]] == 1:
            print("Цель в препятствии!")
            return []

        A = defaultdict(lambda: float('inf'))
        f = defaultdict(lambda: float('inf'))
        parent = defaultdict(lambda: -1)
        visited = defaultdict(lambda: 0)

        heap = []
        heapq.heappush(heap, (self.heuristic(start_idx, end_idx), start_idx))
        A[start_idx] = 0
        f[start_idx] = self.heuristic(start_idx, end_idx)
        parent[start_idx] = -1

        moves = [(-1, 0), (0, -1), (1, 0), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while heap:
            _, v = heapq.heappop(heap)

            if visited[v] == 1:
                continue

            if v == end_idx:
                break

            visited[v] = 1

            x, y = v
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid.sizeOG and 0 <= ny < grid.sizeOG):
                    continue
                if grid.og[ny][nx] == 1:
                    continue

                if dx != 0 and dy != 0:
                    if grid.og[y][nx] == 1 or grid.og[ny][x] == 1:
                        continue

                neighbor = (nx, ny)
                tentative = A[v] + 1 

                if tentative < A[neighbor]:
                    parent[neighbor] = v
                    A[neighbor] = tentative
                    f[neighbor] = tentative + self.heuristic(neighbor, end_idx)
                    heapq.heappush(heap, (f[neighbor], neighbor))

        recovered = self.pathRecovery(parent, end_idx)
        return recovered

   