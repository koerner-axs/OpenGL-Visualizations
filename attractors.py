from typing import List, Tuple

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


initial_position = np.random.normal(0.0, 0.1, (3,))
angles = np.array([0.0, 0.0, 0.0])
translation = np.array([0.0, 0.0, -100.0])

phase_angle = 0.0
display = (1920, 1080)

simple_alg = True
debug = False


def my_attractor1(point, num_steps) -> List[Tuple[int, int, int]]:
    dt = 0.003333 # Timestep

    # sigma = 10.0
    sigma = 20.0

    # rho = 28
    rho = 88

    # beta = (8 / 3)
    beta = 7 / 3

    x, y, z = point
    points = []
    for _ in range(num_steps):
        points.append((x, y, z))

        # Update point using the differential equations
        x += (sigma * (y - x)) * dt
        y += (x * (rho - z) - y) * dt
        z += (x * y - beta * z) * dt

    return points


def my_attractor2(point, num_steps) -> List[Tuple[int, int, int]]:
    dt = 0.1 # Timestep
    precision_steps_ratio = 10

    x, y, z = point
    points = []
    for _ in range(num_steps):
        points.append((x, y, z))

        # Update point using the differential equations
        for _ in range(precision_steps_ratio):
            x += (-0.01 * y + z - 0.1 * x) * (dt / precision_steps_ratio)
            y += (1 + 0.1 * x - 0.01 * y - 0.01* z) * (dt / precision_steps_ratio)
            z += (-np.sqrt(x**2 + y**2) - x - 0.35 * z + (0.1 / (z+1))) * (dt / precision_steps_ratio)

    return points


def roessler_attractor(point, num_steps) -> List[Tuple[int, int, int]]:
    dt = 0.025  # Timestep
    precision_steps_ratio = 1

    a = 0.2
    b = 0.2
    c = 5.7

    x, y, z = point
    points = []
    for _ in range(num_steps):
        points.append((x, y, z))

        # Update point using the differential equations
        for _ in range(precision_steps_ratio):
            x += (- y - z) * (dt / precision_steps_ratio)
            y += (x + a * y) * (dt / precision_steps_ratio)
            z += (b + z * (x - c)) * (dt / precision_steps_ratio)

    return points


def lorenz_attractor(point, num_steps) -> List[Tuple[int, int, int]]:
    dt = 0.003333  # Timestep
    sigma = 10.0
    rho = 28
    beta = (8 / 3)

    x, y, z = point
    points = []
    for _ in range(num_steps):
        points.append((x, y, z))

        # Update point using the differential equations
        x += (sigma * (y - x)) * dt
        y += (x * (rho - z) - y) * dt
        z += (x * y - beta * z) * dt

    return points


simulator = lorenz_attractor  # my_attractor1


def find_center(points_array) -> Tuple[np.ndarray, float]:
    # Initialize centroid with center of the AABB of the point cloud.
    centroid = 0.5 * (np.min(points_array, axis=0) + np.max(points_array, axis=0))

    if simple_alg:
        # Simple algorithm simply takes the center of the AABB of the point cloud.
        # This is fast and given the smoothing I apply later on it is also pleasant to view.
        r2 = np.inner(centroid, centroid)
    else:
        # This iterative algorithm is designed to improve the simple AABB base centroid towards
        # the true center of the enclosing sphere.
        # For some attractors and number of points settings the convergence is too slow and
        # requires too many iterations to run in RT. The improvement of doing this the proper way
        # is largely invisible due to the rotation of the viewport and the applied smoothing.
        for i in range(100):
            points = points_array - centroid

            # Find furthest away point and furthest expanse in opposite direction.
            # Then choose the centroid as the mid point
            furthest_point = points[np.argmax(np.linalg.norm(points, ord=2, axis=-1))]
            dist_furthest_point = np.linalg.norm(furthest_point, ord=2)
            furthest_point /= dist_furthest_point
            dots = np.sum(furthest_point * points, axis=-1)
            dist_opposite_side = -np.min(dots)
            centroid = centroid + furthest_point * 0.5 * (dist_furthest_point - dist_opposite_side)

        r2 = np.inner(centroid, centroid)

        # More sophisticated, but unfortunately more unstable algorithm
        # for i in range(100):
        #     # Find furthest away point and furthest expanse in opposite direction.
        #     # Then choose the centroid as the mid point
        #     points = points_array - centroid
        #     f1 = points[np.argmax(np.linalg.norm(points, ord=2, axis=-1))]
        #     df1 = np.linalg.norm(f1, ord=2)
        #     dots1 = np.sum((f1 / df1) * points, axis=-1)
        #     dn1 = -np.min(dots1)
        #     centroid = centroid + (f1 / df1) * 0.5 * (df1 - dn1)  # First adjustment of the centroid
        #
        #     # Project points onto the orthogonal plane (orthogonal to the f1-n1 axis).
        #     # Take into account the first adjustment of the centroid.
        #     projected_points = points_array - centroid
        #     projected_points -= np.outer(np.dot(projected_points, (f1 / df1)), (f1 / df1))  # Subtract the dot product to project
        #     f2 = projected_points[np.argmax(np.linalg.norm(projected_points, ord=2, axis=-1))]
        #     df2 = np.linalg.norm(f2, ord=2)
        #     dots2 = np.sum((f2 / df2) * projected_points, axis=-1)
        #     dn2 = -np.min(dots2)
        #     centroid = centroid + (f2 / df2) * 0.5 * (df2 - dn2) * 0  # Second adjustment of the centroid
        #
        #     # Project points onto the axis orthogonal to the axes f1-n1 and f2-n2.
        #     # Take into account the first and second adjustment of the centroid.
        #     orth_vector = np.cross((f1 / df1), (f2 / df2))
        #     projected_points = np.outer(np.dot(points_array - centroid, orth_vector), orth_vector)  # orth_vector is normalized
        #     f3 = projected_points[np.argmax(np.linalg.norm(projected_points, ord=2, axis=-1))]
        #     df3 = np.linalg.norm(f3, ord=2)
        #     dots3 = np.sum((f3 / df3) * projected_points, axis=-1)
        #     dn3 = -np.min(dots3)
        #     centroid = centroid + (f3 / df3) * 0.5 * (df3 - dn3) * 0  # Final adjustment of the centroid

    return centroid, np.nan_to_num(np.sqrt(r2))


def main():
    global angles, translation

    # Initialize pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Setup OpenGL to PyGame binding
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Setup OpenGL
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 15000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Constants
    initial_points = 150
    constant_phase_num_points = 1500 # 5000
    new_points_per_update = 10

    # Initialize point list
    points = simulator(initial_position, initial_points)
    smoothed_centroid = np.array(points).mean(axis=0)
    smoothed_max_dist_to_centroid = np.max(np.array(points) - smoothed_centroid)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        clock.tick(60)

        # Compute new points and remove the oldest ones.
        points.extend(simulator(points[-1], new_points_per_update))
        if len(points) > constant_phase_num_points:
            del points[:len(points)-constant_phase_num_points]  # Delete oldest points to reach target length

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update translation and rotation.
        points_array = np.array(points)
        centroid, max_dist_to_centroid = find_center(points_array)
        if smoothed_centroid is not None:
            factor1 = 0.99
            smoothed_centroid = factor1 * smoothed_centroid + (1.0 - factor1) * centroid
        else:
            smoothed_centroid = centroid
        if smoothed_max_dist_to_centroid is not None:
            factor2 = 0.975
            smoothed_max_dist_to_centroid = factor2 * smoothed_max_dist_to_centroid + (1.0 - factor2) * max_dist_to_centroid
        else:
            smoothed_max_dist_to_centroid = max_dist_to_centroid
        points_array -= smoothed_centroid

        translation = [0.0, 0.0, -3 * smoothed_max_dist_to_centroid]
        angles += [0.0, 0.3, 0.1]

        # OpenGL rendering code
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(*translation)
        glRotatef(angles[2], 0.0, 0.0, 1.0)
        glRotatef(angles[1], 0.0, 1.0, 0.0)
        glRotatef(angles[0], 1.0, 0.0, 0.0)

        glEnable(GL_POINT_SMOOTH)
        glPointSize(1)

        glBegin(GL_POINTS)
        for idx in range(len(points)):
            pos = idx / len(points)
            color = np.array([pos, 1-0.75*pos, 1]) * np.sqrt(pos)
            glColor3f(*color)
            glVertex3d(*(points_array[idx]))
        glEnd()

        if debug:
            glPointSize(5)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3d(0.0, 0.0, 0.0)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(*(centroid - smoothed_centroid))
            glEnd()

            glPushMatrix()
            glTranslatef(*(centroid - smoothed_centroid))
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(-max_dist_to_centroid, 0.0, 0.0)
            glVertex3f(max_dist_to_centroid, 0.0, 0.0)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, -max_dist_to_centroid, 0.0)
            glVertex3f(0.0, max_dist_to_centroid, 0.0)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0.0, 0.0, -max_dist_to_centroid)
            glVertex3f(0.0, 0.0, max_dist_to_centroid)
            glEnd()
            glPopMatrix()

        glPopMatrix()

        pygame.display.flip()


main()
