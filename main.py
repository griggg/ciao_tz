import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as mplCircle
import numpy as np

from CIAO import CIAO
from OccupancyGrid import Circle
from FindPathAlgorithms import *

matplotlib.use('tkagg')  

start_point = None
goal_point = None
obstacles = [] 

RADIUS = 0.3   
SIZE = 2.0      

def on_click(event):
    """Обработка кликов мыши"""
    global start_point, goal_point, obstacles
    
    if event.inaxes is None:
        return
    
    x, y = event.xdata, event.ydata
    
    if event.button == 1:
        obstacles.append((x, y, RADIUS))
        # Рисуем круг сразу
        circle = mplCircle((x, y), RADIUS, color='red', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.plot(x, y, 'ro', markersize=3)
        plt.draw()
        print(f"Добавлено препятствие: ({x:.2f}, {y:.2f})")
    
    elif event.button == 3:
        if obstacles:
            distances = [np.sqrt((x - ox)**2 + (y - oy)**2) for (ox, oy, _) in obstacles]
            nearest_idx = np.argmin(distances)
            removed = obstacles.pop(nearest_idx)
            print(f"Удалено препятствие: ({removed[0]:.2f}, {removed[1]:.2f})")
            redraw_plot()

def on_key(event):
    """Обработка нажатий клавиш"""
    global start_point, goal_point
    
    if event.inaxes is None:
        return
    
    x, y = event.xdata, event.ydata
    
    if event.key == '1':
        start_point = (x, y)
        redraw_plot()
        print(f"Старт установлен: ({x:.2f}, {y:.2f})")
    
    elif event.key == '2':
        goal_point = (x, y)
        redraw_plot()
        print(f"Цель установлена: ({x:.2f}, {y:.2f})")
    
    elif event.key == '3':
        if start_point is None or goal_point is None:
            print("Ошибка: не заданы старт или цель!")
            return
        plt.close() 

def redraw_plot():
    plt.cla()
    plt.xlim(0, SIZE)
    plt.ylim(0, SIZE)
    plt.grid(True, alpha=0.3)
   
    
    for (ox, oy, r) in obstacles:
        circle = mplCircle((ox, oy), r, color='red', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.plot(ox, oy, 'ro', markersize=3)
    
    if start_point:
        sx, sy = start_point
        plt.plot(sx, sy, 'go', markersize=10, markeredgecolor='black')
        plt.annotate('СТАРТ', (sx, sy), xytext=(10, 10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))
    if goal_point:
        gx, gy = goal_point
        plt.plot(gx, gy, 'b*', markersize=15, markeredgecolor='black')
        plt.annotate('ЦЕЛЬ', (gx, gy), xytext=(10, -15),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white'))
    
    plt.draw()



def test_case_1():
    obstacles_data = [
        (0.93, 0.63), (0.97, 1.47), (1.36, 1.73),
        (0.68, 0.59), (0.95, 0.38), (1.36, 0.57)
    ]
    obstacles = [Circle(x, y, 0.3) for x, y in obstacles_data]

    start = (0.05806451612903224, 0.035714285714285754)
    goal  = (1.8935483870967744, 1.9155844155844157)

    ciao = CIAO(
        START=start,
        GOAL=goal,
        SIZEMAP=(2, 2),
        obstacles=obstacles,
        GRIDSIZE=20,
        tf=10
    )
    ciao.process()
    ciao.visualize()

def test_case_2():
    
    obstacles_data = [
        (1.44, 1.19), (0.73, 1.21), (1.24, 0.96), (1.15, 1.01)
    ]
    obstacles = [Circle(x, y, 0.3) for x, y in obstacles_data]

    start = (0.3967741935483871, 0.09415584415584421)
    goal  = (1.6, 1.720779220779221)

    ciao = CIAO(
        START=start,
        GOAL=goal,
        SIZEMAP=(2, 2),
        obstacles=obstacles,
        GRIDSIZE=120,
        tf=50
    )
    ciao.process()
    ciao.visualize()

def test_case_3():
    obstacles_data = [
        (0.68, 1.44), (0.63, 1.43), (0.46, 1.64), (0.60, 1.80),
        (0.67, 1.56), (0.67, 1.55), (0.75, 1.28), (0.80, 1.19),
        (0.85, 1.16), (0.93, 1.10), (1.05, 0.98), (1.06, 0.97),
        (1.61, 1.43), (1.66, 1.43), (1.80, 1.41), (1.95, 1.38)
    ]
    obstacles = [Circle(x, y, 0.3) for x, y in obstacles_data]

    start = (0.11612903225806454, 0.09415584415584421)
    goal = (1.8806451612903226, 1.9155844155844157)

    ciao = CIAO(
        START=start,
        GOAL=goal,
        SIZEMAP=(2, 2),
        obstacles=obstacles,
        GRIDSIZE=120,
        tf=50,
        
    )
    ciao.process()
    ciao.visualize()

def test_case_4():
    obstacles_data = [
        (0.87, 1.50), (0.99, 1.40), (1.26, 1.19), (1.38, 0.96),
        (0.94, 0.82), (0.69, 1.07), (0.96, 1.23), (1.09, 0.99)
    ]
    obstacles = [Circle(x, y, 0.3) for x, y in obstacles_data]

    start = (0.14193548387096777, 0.10389610389610393)
    goal = (1.8806451612903226, 1.8214285714285716)

    ciao = CIAO(
        START=start,
        GOAL=goal,
        SIZEMAP=(2, 2),
        obstacles=obstacles,
        GRIDSIZE=20,
        tf=20
    )
    ciao.process()
    ciao.visualize()


def main_interactive():
    global start_point, goal_point, obstacles

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.text(0.02, 0.98,  # координаты в долях фигуры (левый верхний угол)
         "- ЛКМ: добавить препятствие ;"
         "- ПКМ: удалить ближайшее\n"
         "- 1: установить старт;"
         "- 2: установить цель\n"
         "- 3: завершить и запустить расчёт",
         transform=fig.transFigure,  # координаты относительно окна
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.grid(False) 
    ax.grid(True, alpha=0.3)
    ax.set_title('Интерактивный редактор')

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    if start_point is None or goal_point is None:
        print("Ошибка: не заданы старт или цель!")
        return

    print("\n--- СОБРАННЫЕ ДАННЫЕ ---")
    print(f"START = {start_point}")
    print(f"GOAL = {goal_point}")
    print(f"OBSTACLES ({len(obstacles)}):")
    for i, (ox, oy, r) in enumerate(obstacles):
        print(f"  {i+1}: ({ox:.2f}, {oy:.2f}), r={r}")

    obs_circles = [Circle(x, y, r) for (x, y, r) in obstacles]

    ciao = CIAO(START=start_point, GOAL=goal_point, SIZEMAP=(2, 2),
                obstacles=obs_circles, GRIDSIZE=30, tf=10)
    ciao.process(FIND_PATH_ALGO=AStar())
    ciao.visualize()

if __name__ == '__main__':
    import sys

    if sys.argv[1] == "custom":
        main_interactive()
    elif sys.argv[1] == "test1":
        test_case_1()
    elif sys.argv[1] == "test2":
        test_case_2()
    elif sys.argv[1] == "test3":
        test_case_3()
    elif sys.argv[1] == "test4":
        test_case_4()