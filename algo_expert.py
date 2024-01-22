'''
# pygame is 2d graphics module
#Left click to select
#Right click to un-select
#C for Clear

# D for Dijkstra
# B for BFS
# F for DFS
# S for Best First Search
# A for A* Algorithm

'''


from collections import deque
import math
import pygame
from heapq import *

window_size = 800
WINDOW = pygame.display.set_mode((window_size, window_size))
grid_size = 40

pygame.display.set_caption("AlgoExpert\n")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 139)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
PURPLE = (120, 0, 160)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
GOLD = (255, 223, 0)

WINDOW.fill(WHITE)
gap = window_size // grid_size
blocksize = gap


class Block:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.xcor = row * gap
        self.ycor = col * gap
        self.color = BLACK
        self.neighbors = []

    def make_block(self):
        pygame.draw.rect(WINDOW, self.color, (self.xcor, self.ycor, gap, gap))

    def assign_neighbors(self, grid):
        self.neighbors = []
        if self.row > 0 and grid[self.row - 1][self.col].color != WHITE:                # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.row < grid_size - 1 and grid[self.row + 1][self.col].color != WHITE:    # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.col > 0 and grid[self.row][self.col - 1].color != WHITE:                # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])
        if self.col < grid_size - 1 and grid[self.row][self.col + 1].color != WHITE:    # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])


def make_gridlines():
    for i in range(grid_size):
        pygame.draw.line(WINDOW, GREY, (0, i * gap), (window_size, i * gap))
        for j in range(grid_size):
            pygame.draw.line(WINDOW, GREY, (j * gap, 0), (j * gap, window_size))


def draw(grid):
    WINDOW.fill(BLACK)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid[i][j].make_block()

    make_gridlines()
    pygame.display.update()


def dijkstra(draw, grid, st, end):
    dist = {block: float('inf') for row in grid for block in row}
    path_tracker = {}
    count = 0
    h = []
    dist[st] = 0
    heappush(h, (0, count, st))
    while h:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = heappop(h)[2]
        if current == end:
            while current in path_tracker:
                current = path_tracker[current]
                current.color = RED
                draw()
            return True

        for i in current.neighbors:
            if dist[current] + 1 < dist[i]:
                path_tracker[i] = current
                dist[i] = dist[current] + 1
                i.color = GREEN
                count += 1
                heappush(h, (dist[i], count, i))

        draw()
        if current != st:
            current.color = ORANGE

    return False


def bfs01(draw, grid, st, end):
    vis = {block: False for row in grid for block in row}
    queue = deque([])
    path_tracker = {}
    vis[st] = True
    queue.append(st)
    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        curr = queue.popleft()
        if curr == end:
            while curr in path_tracker:
                curr = path_tracker[curr]
                curr.color = RED
                draw()
            return True

        for i in curr.neighbors:
            if not vis[i]:
                vis[i] = True
                i.color = GREEN
                path_tracker[i] = curr
                queue.append(i)

        draw()
        if curr != st:
            curr.color = ORANGE

    return False


def dfs_using_stack(draw, grid, st, end):
    vis = {block: False for row in grid for block in row}
    stack = deque([])
    stack.append(st)
    vis[st] = True
    path_tracker = {}
    while len(stack) != 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        curr = stack.pop()
        if curr == end:
            while curr in path_tracker:
                curr = path_tracker[curr]
                curr.color = RED
                draw()
            return True

        draw()
        for i in curr.neighbors:
            if not vis[i]:
                path_tracker[i] = curr
                i.color = GREEN
                vis[i] = True
                stack.append(i)

        if curr != st:
            curr.color = ORANGE

    return False


def euclidean_distance(a, b):
    return ((a.xcor - b.xcor)**2 + (a.ycor - b.ycor)**2)**0.5


def best_first_search(draw, grid, st, end):
    vis = {block: False for row in grid for block in row}
    val = {block: euclidean_distance(block, end) for row in grid for block in row}
    path_tracker = {}
    have = []
    cnt = 0
    vis[st] = True
    heappush(have, (val[st], cnt, st))
    while have:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = heappop(have)[2]
        if current == end:
            while current in path_tracker:
                current = path_tracker[current]
                current.color = RED
                draw()
            return True

        for i in current.neighbors:
            if not vis[i]:
                i.color = GREEN
                path_tracker[i] = current
                vis[i] = True
                cnt += 1
                heappush(have, (euclidean_distance(i, end), cnt, i))

        draw()
        if current != st:
            current.color = ORANGE

    return False


def heuristic(a, b):
    # Manhattan Distance
    return abs(a.row - b.row) + abs(a.col - b.col)


def a_star(draw, grid, start, end):
    count = 0
    open_set = []
    heappush(open_set, (0, count, start))
    came_from = {}

    g_score = {block: float("inf") for row in grid for block in row}
    g_score[start] = 0

    f_score = {block: float("inf") for row in grid for block in row}    # stores predicted score
    f_score[start] = heuristic(start, end)

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = heappop(open_set)[2]

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.color = RED
                draw()

            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, end)

                if neighbor not in [block[2] for block in open_set]:
                    count += 1
                    heappush(open_set, (f_score[neighbor], count, neighbor))
                    neighbor.color = GREEN

        draw()

        if current != start:
            current.color = ORANGE

    return False


def main():
    grid = [[Block(i, j) for j in range(grid_size)] for i in range(grid_size)]

    st = False
    end = False
    flag = True
    while flag:
        draw(grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False

            if event.type == pygame.KEYDOWN:
                for i in range(grid_size):
                    for j in range(grid_size):
                        grid[i][j].assign_neighbors(grid)

                if event.key == pygame.K_c:
                    grid = [[Block(i, j) for j in range(grid_size)] for i in range(grid_size)]
                    st = False
                    end = False
                else:
                    if event.key == pygame.K_d:  # Dijkstra
                        dijkstra(lambda: draw(grid), grid, st, end)
                        continue
                    if event.key == pygame.K_b:  # BFS
                        bfs01(lambda: draw(grid), grid, st, end)
                        continue
                    if event.key == pygame.K_f:  # DFS
                        dfs_using_stack(lambda: draw(grid), grid, st, end)
                        continue
                    if event.key == pygame.K_s:  # Best First Search
                        best_first_search(lambda: draw(grid), grid, st, end)
                        continue
                    if event.key == pygame.K_a:  # A* algorithm
                        a_star(lambda: draw(grid), grid, st, end)
                        continue

            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                i = x // gap
                j = y // gap

                if not st and grid[i][j] != end:
                    st = grid[i][j]
                    grid[i][j].color = YELLOW

                elif not end and grid[i][j] != st:
                    end = grid[i][j]
                    grid[i][j].color = PURPLE

                elif grid[i][j] != st and grid[i][j] != end:
                    grid[i][j].color = WHITE

            else:
                if pygame.mouse.get_pressed()[2]:
                    x, y = pygame.mouse.get_pos()
                    i = x // gap
                    j = y // gap

                    if grid[i][j] == st:
                        st = False
                        grid[i][j].color = BLACK
                    elif grid[i][j] == end:
                        end = False
                        grid[i][j].color = BLACK
                    else:
                        grid[i][j].color = BLACK

    pygame.quit()


main()
