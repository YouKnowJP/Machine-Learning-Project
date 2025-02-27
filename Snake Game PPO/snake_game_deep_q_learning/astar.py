#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:11:36 2025

@author: youknowjp
"""

import heapq
from game import Point  # Reuse the Point namedtuple from your game module
import config


def heuristic(a: Point, b: Point) -> int:
    """Manhattan distance heuristic."""
    return abs(a.x - b.x) + abs(a.y - b.y)


def get_neighbors(point: Point, w: int, h: int, block_size: int):
    """Return neighboring points (up, down, left, right) within bounds."""
    neighbors = []
    moves = [(block_size, 0), (-block_size, 0), (0, block_size), (0, -block_size)]
    for dx, dy in moves:
        neighbor = Point(point.x + dx, point.y + dy)
        if 0 <= neighbor.x < w and 0 <= neighbor.y < h:
            neighbors.append(neighbor)
    return neighbors


def reconstruct_path(came_from: dict, current: Point):
    """Reconstructs the path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(start: Point, goal: Point, obstacles: set, w: int, h: int, block_size: int):
    """
    Performs A* search from start to goal avoiding obstacles.

    Args:
        start: Starting point.
        goal: Goal point.
        obstacles: A set of Points that are not traversable (e.g. snake body).
        w: Width of the game window.
        h: Height of the game window.
        block_size: The size of each grid cell.

    Returns:
        A list of Points representing the path from start to goal (including both),
        or an empty list if no path is found.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, w, h, block_size):
            if neighbor in obstacles:
                continue  # Skip cells occupied by the snake
            tentative_g = g_score[current] + 1  # Assume uniform cost for each move
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found
