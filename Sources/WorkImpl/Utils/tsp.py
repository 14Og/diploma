# ------------------------------------------------------------------------------------
# TSP - traveling salesman problem integer linear programming method
# Решатель задачи комивояжёра методом целочисленного линейного программирования
# Rebuilder 09.06.2023 https://habr.com/ru/publication/edit/740984/
# ------------------------------------------------------------------------------------
from typing import Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

import os
import math
import random
import yaml

from scipy import optimize
from scipy import sparse
from dataclasses import dataclass
from multimethod import overload
from datetime import datetime

from networkx.drawing.nx_agraph import graphviz_layout

from .functions import spherical2cartesian, cartesian2spherical, reinterpret

@dataclass
class Path:
    """
    Dataclass describing a path using:
    * list of point indices;
    * path length;
    * arr: angles
    * path name (optional).
    """

    indices: list[int]
    arr: np.ndarray
    length: float
    dir_path: str

    def save(self, path: str = None):
        if path is None:
            path = self.dir_path

        path = os.path.join(path, "path.yaml")

        to_dump = [
            {
                "elevation": float(element[0]),
                "azimuth": float(element[1]),
            }
            for element in self.arr
        ]

        with open(path, "w") as yaml_file:
            yaml.safe_dump(to_dump, yaml_file, indent=1, sort_keys=False)

class Voyager:
    mode: str = "release"
    def __init__(self, mode) -> None:
        self.path: Path = None
        self.matrix = None
        self.graph = None
        Voyager.mode = mode 

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def create_matrix(self, point_arr: np.ndarray) -> Tuple[np.ndarray, nx.Graph]:
        angle_arr = reinterpret(point_arr=point_arr)
        matrix_shape = (angle_arr.shape[0], angle_arr.shape[0])
        matrix = np.empty(matrix_shape)

        for row_index, angle in enumerate(angle_arr):
            for col_index, next_angle in enumerate(angle_arr):
                matrix[row_index, col_index] = self.angle_distance(angle, next_angle)

        graph: nx.Graph = nx.complete_graph(matrix.shape[0])
        for i, j in graph.edges():
            if j > i:
                graph[i][j]['weight'] = Voyager.angle_distance(angle=angle_arr[i], next_angle=angle_arr[j])

        return matrix, graph    

    def fast_create_path(self, point_arr: np.ndarray) -> np.ndarray:
        path = np.empty_like(prototype=point_arr, dtype=int)
        angle_arr = reinterpret(point_arr)

        matrix, _ = self.create_matrix(point_arr) # матрица смежности и ориентированный связный граф

        return path

    def sort_path(self, edges: List[Tuple]) -> list:
        next_index = edges[0][0]
        list_index = []
        list_index.append(next_index)
        length = len(edges)

        for i in range(length):
            for index, edge in enumerate(edges):
                if edge[0] == next_index:
                    next_index = edge[1]
                    edges.pop(index)
                    list_index.append(next_index)
                    break
                if index == len(edges) - 1:
                    for index, edge in enumerate(edges):
                        if edge[1] == next_index:
                            next_index = edge[0]
                            edges.pop(index)
                            list_index.append(next_index)
                            break

        return list_index

    def create_path(self, point_arr: np.ndarray) -> Path:
        self.matrix, self.graph = self.create_matrix(point_arr)
        angles = reinterpret(point_arr)

        start_time = datetime.now()
        res = Voyager.TSP_linprog(self.graph, 3, 5)
        # res = Voyager.TSP_milp(self.graph, 3, 5)
        date_diff = (datetime.now() - start_time).total_seconds()

        if self.mode == "debug":
            plt.figure(figsize=(12, 9))
            plt.axis("equal")
            print('time =', date_diff)

            nx.draw(self.graph, angles, width=0, with_labels=True, node_size=0, font_size=6, font_color="black")
            nx.draw(res, angles, width=1, edge_color="red", style="-", with_labels=False, node_size=0)

            plt.show()

        # dict = nx.to_dict_of_dicts(res)
        # pos = nx.spring_layout(res)

        edges = []
        for i in res.edges:
            edges.append(i)
        
        indices = self.sort_path(edges)
        new_angle_arr = self.preprocess(point_arr, indices) 
        
        return Path(indices, new_angle_arr, 0, None) 
        
    def preprocess(self, point_arr, indices) -> np.ndarray:
        angle_arr = reinterpret(point_arr)
        arr = np.empty_like(prototype=angle_arr, dtype=float)
        
        for i in range(len(indices)-1):
            arr[i] = angle_arr[indices[i]]
            
        return arr

    @staticmethod
    def point_distance(center: list, point: list, next_point: list) -> Tuple[float, float]:
        radius = math.hypot(point[0] - center[0], point[1] - center[1])
        distance = math.hypot(next_point[0] - point[0], next_point[1] - point[1])

        return radius, distance 

    @staticmethod
    def angle_distance(angle: list, next_angle: list):
        return math.ceil(abs(next_angle[0] - angle[0]) + abs(next_angle[1] - angle[1]))

    @staticmethod
    def matrix_count(a):
        # Возвращет размер плоского списка для квадратной симметричной матрицы
        return((a * a - a) // 2)

    @staticmethod
    def triangle_intersection(a, b, c, vertex_array, val):
        # Нужно ли убрать ребро
        if (b > a) or (c > a) or (a < 0.000001):
            return val      
        cross_size = len(vertex_array)

        key = [True] * cross_size
        for i in range(cross_size):
            if (b == vertex_array[i]):
                key[i] = False       
        p = 0.5 * (a + b + c)
        s = p * (p - a) * (p - b) * (p - c)
        r = 2 * np.sqrt(abs(s)) / a
        for i in range(cross_size):
            if key[i] and vertex_array[i] >= r:    
                val += 1    
        return val

    @staticmethod
    def algorithm_evaluation(array_key, result_sets, k):
        # Возвращет оценку какой алгоритм создаст меньше единиц
        set1 = set()
        set2 = set()

        for i in result_sets[k]:
            for j in result_sets[k]:
                if i > j and (i, j) in array_key:
                    set1.add(array_key[(i, j)])

        for l, val_l in enumerate(result_sets):
            if l != k:
                for i in result_sets[k]:
                    for j in val_l:
                        key = (i, j) if i > j else (j, i)
                        if (key in array_key):
                            set2.add(array_key[key])

        if len(set1) <= len(set2):
            return True, set1
        else:
            return False, set2

    @staticmethod
    def TSP_linprog(init_graph: nx.Graph, circles, cross) -> nx.Graph:
        # Находит минимальный путь в графе методом целочисленного линейного программирования

        # print('Run TSP linprog', end='')
        n = len(init_graph)
        # Количество переменных солвера
        flat_n = Voyager.matrix_count(n)
        # Разворачиваем граф как матрицу смежности
        adjacency_matrix = nx.adjacency_matrix(init_graph, dtype=np.float64).todense()
        # Сортируем матрицу смежности по строкам
        adjacency_matrix.sort(axis=1)
        # Выбираем только часть матрицы
        vertex_array = adjacency_matrix[:,1:circles + 1]
        del adjacency_matrix

        graph_of_neighbors = nx.Graph(())    
        for i in range(n):
            for j in range(n):
                if i > j:                
                    value = 0
                    for k in range(n):
                        if (k != i) and (k != j): 
                            value = Voyager.triangle_intersection(init_graph[i][j]['weight'], init_graph[i][k]['weight'], init_graph[j][k]['weight'], vertex_array[k], value)
                            if value > cross:
                                break
                    if value <= cross:
                        graph_of_neighbors.add_edge(i, j)
        del vertex_array  

        graph_united = nx.Graph(())    
        for i, j in list(graph_of_neighbors.edges()):      
            for k in list(graph_of_neighbors.adj[i]):
                if j != k and not graph_united.has_edge(j, k):
                    graph_united.add_edge(j, k)
            for k in list(graph_of_neighbors.adj[j]):
                if i != k and not graph_united.has_edge(i, k):
                    graph_united.add_edge(i, k)

        for i, j in list(graph_of_neighbors.edges()):
            if not graph_united.has_edge(i, j):
                graph_united.add_edge(i, j)
        del graph_of_neighbors

        # Ограничение на диаппазон переменных
        variable_bounds = (0, 1)
        #     variable_bounds = (0, None)

        # Заполняем основные элементы
        edges_list = list(graph_united.edges())
        n_var = len(edges_list)
        array_index = {}
        array_key = {}
        objective_function = np.empty(n_var, dtype=np.float64)

        # Объявление массивов ограничений
        equality_a = sparse.lil_matrix((n, n_var), dtype=np.int8)
        equality_b = [2] * n
        inequalities_a = sparse.lil_matrix((0, n_var), dtype=np.int8)
        inequalities_b = []

        for i, val in enumerate(edges_list):
            key = (val[0], val[1]) if val[0] > val[1] else (val[1], val[0])
            array_key[key] = i
            array_index[i] = key
            objective_function[i] = init_graph[key[0]][key[1]]['weight']
            equality_a[key[0], i] = 1
            equality_a[key[1], i] = 1
        del graph_united

        constraints = 0
        step = 0
        while True:       
            # Ищем решение
            if Voyager.mode == "debug":
                res = optimize.linprog(
                    c=objective_function,
                    bounds=variable_bounds,
                    A_eq=equality_a,
                    b_eq=equality_b,
                    A_ub=inequalities_a,
                    b_ub=inequalities_b,
                    method="highs",
                    integrality=1,
                    options={"disp": True}
                )
            else:
                res = optimize.linprog(
                    c=objective_function,
                    bounds=variable_bounds,
                    A_eq=equality_a,
                    b_eq=equality_b,
                    A_ub=inequalities_a,
                    b_ub=inequalities_b,
                    method="highs",
                    integrality=1,
                )
            step += 1
            print('.', end='')

            # Сохраняем результат в виде графа связей
            graph_resilt = nx.Graph()
            for i, val in enumerate(res.x):
                if round(val) >= 1:
                    graph_resilt.add_edge(*array_index[i])

            # Разбиваем граф результата на подмножества
            result_sets = list(nx.connected_components(graph_resilt))
            qty_sets = len(result_sets)

            # Решение найдено если в графе только одно множество
            if qty_sets == 1:
                break

            # Если только два множества соединяем их напрямую
            if qty_sets == 2:
                inequalities_a.resize((constraints + 1, n_var))
                inequalities_b += [-2]
                for i in result_sets[0]: 
                    for j in result_sets[1]:
                        key = (i, j) if i > j else (j, i)
                        if key in array_key:
                            inequalities_a[constraints, array_key[key]] = -1
                constraints += 1
                continue

            # Вводим столько ограничений, сколько у нас получилось множеств
            inequalities_a.resize((constraints + qty_sets, n_var))
            for i, val_i in enumerate(result_sets):
                # Выбираем тот алгоритм, который добавит меньше значений в разреженную матрицу
                value, add_set = Voyager.algorithm_evaluation(array_key, result_sets, i)
                if value:
                    inequalities_b += [len(val_i) - 1]
                    for j in add_set:
                        inequalities_a[constraints + i, j] = 1
                else:
                    inequalities_b += [-2]
                    for j in add_set:
                        inequalities_a[constraints + i, j] = -1 
            constraints += qty_sets

        return graph_resilt  # {'length' : res.fun, 'graph' : graph_resilt, 'constraints' : constraints, 'steps' : step}
    # ====================================================================================================================
    def TSP_milp(init_graph: nx.Graph, circles, cross):
        # Находит минимальный путь в графе методом целочисленного линейного программирования

        print('Run TSP milp', end='')  
        n = len(init_graph)
        # Количество переменных солвера
        flat_n = Voyager.matrix_count(n)
        # Разворачиваем граф как матрицу смежности
        adjacency_matrix = nx.adjacency_matrix(init_graph, dtype=np.float64).todense()
        # Сортируем матрицу смежности по строкам
        adjacency_matrix.sort(axis=1)
        # Выбираем только часть матрицы
        vertex_array = adjacency_matrix[:,1:circles + 1]
        del adjacency_matrix

        graph_of_neighbors = nx.Graph(())    
        for i in range(n):
            for j in range(n):
                if i > j:                
                    value = 0
                    for k in range(n):
                        if (k != i) and (k != j): 
                            value = Voyager.triangle_intersection(init_graph[i][j]['weight'], init_graph[i][k]['weight'], init_graph[j][k]['weight'], vertex_array[k], value)
                            if value > cross:
                                break
                    if value <= cross:
                        graph_of_neighbors.add_edge(i, j)
        del vertex_array  

        graph_united = nx.Graph(())
        for i, j in list(graph_of_neighbors.edges()):

            for k in list(graph_of_neighbors.adj[i]):
                if j != k and not graph_united.has_edge(j, k):
                    graph_united.add_edge(j, k)
            for k in list(graph_of_neighbors.adj[j]):
                if i != k and not graph_united.has_edge(i, k):
                    graph_united.add_edge(i, k)

        for i, j in list(graph_of_neighbors.edges()):
            if not graph_united.has_edge(i, j):
                graph_united.add_edge(i, j)
        del graph_of_neighbors

        # Заполняем основные элементы
        edges_list = list(graph_united.edges())
        n_var = len(edges_list)
        array_index = {}
        array_key = {}
        matrix_a = sparse.lil_matrix((n, n_var), dtype=np.int8)    
        objective_function = np.empty(n_var, dtype=np.float64)    
        for i, val in enumerate(edges_list):
            key = (val[0], val[1]) if val[0] > val[1] else (val[1], val[0])
            array_key[key] = i
            array_index[i] = key
            objective_function[i] = init_graph[key[0]][key[1]]['weight']
            matrix_a[key[0], i] = 1
            matrix_a[key[1], i] = 1
        del graph_united

        # Ограничение на диаппазон переменных
        variable_bounds = (0, 1)
        #     variable_bounds = (0, np.inf)

        # Начальные ограничения модели
        lower_bound  = [2] * n
        upper_bound  = [2] * n    

        constraints = 0
        step = 0
        while True:       
            # Ищем решение
            if Voyager.mode == "debug":
                res = optimize.milp(
                    c=objective_function,
                    constraints=optimize.LinearConstraint(
                        matrix_a, lower_bound, upper_bound
                    ),
                    bounds=variable_bounds,
                    integrality=1,
                    options={"disp": True}
                )
            else:
                res = optimize.milp(
                    c=objective_function,
                    constraints=optimize.LinearConstraint(
                        matrix_a, lower_bound, upper_bound
                    ),
                    bounds=variable_bounds,
                    integrality=1
                )
            step += 1
            print('.', end='')

            # Сохраняем результат в виде графа связей
            graph_resilt = nx.Graph()
            for i, val in enumerate(res.x):
                if round(val) >= 1:
                    graph_resilt.add_edge(*array_index[i])

            # Разбиваем граф результата на подмножества
            result_sets = list(nx.connected_components(graph_resilt))
            qty_sets = len(result_sets)

            # Решение найдено если в графе только одно множество
            if qty_sets == 1:
                break

            # Если только два множества соединяем их напрямую
            if qty_sets == 2:
                matrix_a.resize((n + constraints + 1, n_var))
                lower_bound += [2]
                upper_bound += [np.inf]
                for i in result_sets[0]: 
                    for j in result_sets[1]:
                        key = (i, j) if i > j else (j, i)
                        if key in array_key:
                            matrix_a[n + constraints, array_key[key]] = 1
                constraints += 1
                continue

            # Вводим столько ограничений, сколько у нас получилось множеств
            matrix_a.resize((n + constraints + qty_sets, n_var))
            for i, val_i in enumerate(result_sets):
                # Выбираем тот алгоритм, который добавит меньше значений в разреженную матрицу
                value, add_set = Voyager.algorithm_evaluation(array_key, result_sets, i)
                if value:
                    lower_bound += [0]
                    upper_bound += [len(val_i) - 1]
                else:
                    lower_bound += [2]
                    upper_bound += [np.inf]
                for j in add_set:
                    matrix_a[n + constraints + i, j] = 1
            constraints += qty_sets
        print()

        return {'length' : res.fun, 'graph' : graph_resilt, 'constraints' : constraints, 'steps' : step}
    
def distance(point1, point2):
    # Возвращет дистанцию между двумя точками на плоскости 
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

