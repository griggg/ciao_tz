import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('tkagg')  # для Ubuntu с Tkinter
import matplotlib.pyplot as plt
import os
from OccupancyGrid import Circle, OccupancyGrid
from FindPathAlgorithms import FindPathAlgorithm, DFS, AStar
from acados_template import get_tera

import json


# get_tera(force_download=True)


class CIAO:

    def __init__(self, START: tuple[float, float],
                 GOAL: tuple[float, float], SIZEMAP: tuple[int, int],
                 obstacles: list[Circle], GRIDSIZE: int, tf: int):
        self.GOALX, self.GOALY = GOAL
        self.obstacles = obstacles
        self.SIZE_X, self.SIZE_Y = SIZEMAP
        self.GRIDSIZE = GRIDSIZE
        self.START_X, self.START_Y = START
        self.TF = tf


    def process(self, FIND_PATH_ALGO: FindPathAlgorithm = None):
        if FIND_PATH_ALGO is None:
            FIND_PATH_ALGO = AStar()

        grid = OccupancyGrid(self.obstacles, self.GRIDSIZE, (self.SIZE_X, self.SIZE_Y))
        initPath_indices = FIND_PATH_ALGO.getP(grid, (self.START_X, self.START_Y), (self.GOALX, self.GOALY))

        self.path_points = [
            ((ix + 0.5) * grid.blockSize[0], (iy + 0.5) * grid.blockSize[1])
            for (ix, iy) in initPath_indices
        ]

        self.bcircles = grid.boundingCircles(initPath_indices)


        self.N = len(self.bcircles)
        nh = self.N

        posx = ca.SX.sym('posx')
        posy = ca.SX.sym('posy')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        a = ca.SX.sym('a')
        omega = ca.SX.sym('omega')

        refx = ca.SX.sym('refx')
        refy = ca.SX.sym('refy')
        weight = ca.SX.sym('weight')

        model = AcadosModel()
        model.name = "ciao"
        model.x = ca.vertcat(posx, posy, v, theta)
        model.u = ca.vertcat(a, omega)
        model.f_expl_expr = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), a, omega)
        model.p = ca.vertcat(refx, refy, weight)

        stage_cost = 0.2  * (a**2 + omega**2) + weight * ((posx - refx)**2 + (posy - refy)**2)
        model.cost_expr_ext_cost = stage_cost
        terminal_cost = 100 * ((posx - self.GOALX)**2 + (posy - self.GOALY)**2)
        model.cost_expr_ext_cost_e = terminal_cost

        h_expr = [(posx - c.x)**2 + (posy - c.y)**2 - c.r**2 for c in self.bcircles]
        model.con_h_expr = ca.vertcat(*h_expr)
        model.con_h_expr_e = ca.vertcat(*h_expr)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.nh = nh
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.TF

        ocp.constraints.x0 = np.array([self.START_X, self.START_Y, 0.0, 0.0])

        ocp.constraints.lh = -1e9 * np.ones(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.lh_e = -1e9 * np.ones(nh)
        ocp.constraints.uh_e = 1e9 * np.ones(nh)
       
        ocp.constraints.lbu = np.array([-2.0, -2.0])
        ocp.constraints.ubu = np.array([2.0, 2.0])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.parameter_values = np.zeros(3)

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"  

        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        print("Number of constraints (nh) =", self.solver.acados_ocp.dims.nh)

        for k in range(self.N):
            ref_x, ref_y = self.path_points[k]
            self.solver.set(k, "p", np.array([ref_x, ref_y, 0.8]))

            lh_k = -1e9 * np.ones(nh)
            uh_k = 1e9 * np.ones(nh)
            uh_k[k] = 0.001

            if k + 1 < self.N:
                self.solver.constraints_set(k + 1, "lh", lh_k)
                self.solver.constraints_set(k + 1, "uh", uh_k)


        status = self.solver.solve()
        print("Solver status:", status)


        data = {
            'x': [self.solver.get(i, "x").tolist() for i in range(self.N + 1)],
            'u': [self.solver.get(i, "u").tolist() for i in range(self.N)],
            'bcircles': [(c.x, c.y, c.r) for c in self.bcircles],
            'status': status
        }

        with open('solver_data.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print("Saved solver data to solver_data.json")

        
        
    def visualize(self, show=True):
        xs = np.zeros((self.N + 1, 2))
        us = np.zeros((self.N, 2))
        for i in range(self.N):
            full_state = self.solver.get(i, "x")   
            xs[i] = full_state[:2]                 
            us[i] = self.solver.get(i, "u")
        full_state_end = self.solver.get(self.N, "x")
        xs[self.N] = full_state_end[:2]

       

        os.makedirs("tz", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,6))

        for c in self.bcircles:
            circ = plt.Circle((c.x, c.y), c.r, color='green', fill=False, linewidth=1)
            ax.add_patch(circ)

        for c in self.obstacles:
            circ2 = plt.Circle((c.x, c.y), c.r-0.1, color='grey', fill=True, alpha=1, zorder=3)
            ax.add_patch(circ2)
            circ = plt.Circle((c.x, c.y), c.r, color='red', fill=False, alpha=1, zorder=2)
            ax.add_patch(circ)

        ax.plot(xs[:,0], xs[:,1], '-o', label='optimized trajectory')
        ax.scatter(xs[0,0], xs[0,1], c='blue', s=80, label='start')
        ax.scatter(self.GOALX, self.GOALY, c='black', s=80, marker='*', label='goal')

        ref_xs = [p[0] for p in self.path_points]
        ref_ys = [p[1] for p in self.path_points]
        ax.plot(ref_xs, ref_ys, '--', color='orange', label='reference path')
        ax.scatter(ref_xs, ref_ys, color='orange', s=10)

        for k in range(self.N):
            node = k+1
            ax.plot(xs[node,0], xs[node,1], 'bo', markersize=1.2)
        
        ax.set_xlim(0, self.SIZE_X)
        ax.set_ylim(0, self.SIZE_Y)


        dx = self.SIZE_X / self.GRIDSIZE
        dy = self.SIZE_Y / self.GRIDSIZE
        for i in range(1, self.GRIDSIZE):
            ax.axvline(i * dx, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
            ax.axhline(i * dy, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)


        ax.set_aspect('equal')
        ax.grid(False)
        ax.legend()
        plt.tight_layout()
        plt.savefig("tz/solution.png", dpi=200)
        if show:
            plt.show()   
        plt.close()
        print("\nSaved to tz/solution.png")


