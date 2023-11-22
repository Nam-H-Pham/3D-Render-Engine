import numpy as np
from abc import ABC, abstractmethod

class Point3D:
    def __init__(self, x, y, z):
        self.loc = np.array([x, y, z])

    def rotate(self, axis: str, angle: float) -> None:
        angle = np.radians(angle)
        rotation_matix = None
        if axis == 'x':
            rotation_matix = np.array([[1, 0, 0],
                                       [0, np.cos(angle), -np.sin(angle)],
                                       [0, np.sin(angle), np.cos(angle)]])
        elif axis == 'y':
            rotation_matix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                       [0, 1, 0],
                                       [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'z':
            rotation_matix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]])  
        self.loc = np.dot(rotation_matix, self.loc)
        self.loc = self.loc.round(5)

    def rotate_arbitrary(self, axis: np.array, angle: float) -> None:
        angle = np.radians(angle)
        rotation_matix = np.array([[np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)), axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle), axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                                   [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle), np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)), axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                                   [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle), axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle), np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]])
        self.loc = np.dot(rotation_matix, self.loc)
        self.loc = self.loc.round(5)

    def move(self, direction_vector: np.array) -> None:
        self.loc = self.loc + direction_vector

    def __str__(self):
        return str(self.loc)


class Shape3D(ABC):
    def __init__(self, points: list, colour=(255,255,255)) -> None:
        self.points = points
        self.colour = colour
        self.connection_matrix = np.zeros((len(points), len(points)))

    def get_axis_index(self, axis: str) -> int:
        if axis == 'x':
            return 0
        elif axis == 'y':
            return 1
        elif axis == 'z':
            return 2

    def connect(self, point1: Point3D, point2: Point3D) -> None:
        self.connection_matrix[self.points.index(point1)][self.points.index(point2)] = 1
        self.connection_matrix[self.points.index(point2)][self.points.index(point1)] = 1

    def rotate(self, axis: str, angle: float) -> None:
        for point in self.points:
            point.rotate(axis, angle)

    def rotate_arbitrary(self, axis: np.array, angle: float) -> None:
        for point in self.points:
            point.rotate_arbitrary(axis, angle)

    def move(self, direction_vector: np.array) -> None:
        for point in self.points:
            point.move(direction_vector)
            

    def __str__(self):
        return str(self.points)

class Physics_Shape3D(Shape3D):
    def __init__(self, points: list) -> None:
        super().__init__(points)

        self.physics_enabled = True
        self.velocity = np.array([0, 0, 0])

    def force(self, force_vector: np.array) -> None:
        self.velocity = self.velocity + force_vector
        
    def apply_physics(self, gravity: float, drag: float, world_directions: list) -> None:
        if self.physics_enabled:

            gravity_vector = -world_directions[1].loc * gravity
            self.force(gravity_vector)

            collision, vertical_vector = self.floor_collision(world_directions)
            if collision:
                self.normal_collision(world_directions, vertical_vector)
                self.friction(world_directions)
                self.colour = (255, 200, 200)
            else:
                self.colour = (255, 255, 255)

            self.move(self.velocity * drag)

    def floor_collision(self, world_directions) -> tuple:
        z_vector = world_directions[2].loc
        x_vector = world_directions[0].loc
        vertical_unit_vector, vertical_vector = None, None

        collision = False
        for point in self.points:
            vector_along_z_axis = z_vector * np.dot(point.loc, z_vector) / np.dot(z_vector, z_vector)
            vector_along_x_axis = x_vector * np.dot(point.loc, x_vector) / np.dot(x_vector, x_vector)
            flat_vector = vector_along_z_axis + vector_along_x_axis

            vertical_vector = point.loc - flat_vector
            vertical_vector_size = np.linalg.norm(vertical_vector)
            if vertical_vector_size == 0:
                continue
            
            vertical_unit_vector = vertical_vector / vertical_vector_size 
            if vertical_unit_vector[1] < 0:
                collision = True
                break

        return (collision, vertical_vector)

    def friction(self, world_directions) -> None:
        x_vector = world_directions[0].loc
        z_vector = world_directions[2].loc
        velocity_along_x_axis = x_vector * np.dot(self.velocity, x_vector) / np.dot(x_vector, x_vector)
        velocity_along_z_axis = z_vector * np.dot(self.velocity, z_vector) / np.dot(z_vector, z_vector)
        flat_velocity = (velocity_along_x_axis + velocity_along_z_axis) * 0.2
        self.force(-flat_velocity)
        
    def normal_collision(self, world_directions, vertical_vector) -> tuple:
        y_vector = world_directions[1].loc
        up_vector = np.linalg.norm(self.velocity) * y_vector * 0.5
        self.force(up_vector)
        self.move(-vertical_vector)

class World:
    def __init__(self, screen) -> None:
        self.shapes = []
        self.camera = Camera(screen)
        self.floor = []
        self.create_grid(5, 10)

        self.world_center = Point3D(0, 0, 0)
        self.world_directions = [Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]

        self.universal_directions = [Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]

    def create_grid(self, square_size: float, side_num: int) -> None:
        offset = (side_num * square_size) / 2
        for i in range(side_num):
            for j in range(side_num):
                new_plane = self.create_plane(i*square_size-offset, 0, j*square_size-offset, square_size, square_size)
                new_plane.colour = (50, 50, 50)
                self.floor.append(new_plane)
    
    def add_shape(self, shape: Shape3D) -> None:
        self.shapes.append(shape)
    
    def rotate(self, axis: str, angle: float) -> None:
        for point in self.world_directions:
            point.rotate(axis, angle)
        for shape in self.shapes:
            shape.rotate(axis, angle)
        for plane in self.floor:
            plane.rotate(axis, angle)

    def move_shape_axis_aligned(self, shape: Shape3D, axis: str, amount: float) -> None:
        direction = self.world_directions[self.camera.get_axis_index(axis)]
        direction = direction.loc
        shape.move(direction * amount)

    def apply_force_axis_aligned(self, shape: Shape3D, axis: str, amount: float) -> None:
        direction = self.world_directions[self.camera.get_axis_index(axis)]
        direction = direction.loc
        shape.force(direction * amount)


    def apply_force_universal_axis_aligned(self, shape: Shape3D, axis: str, amount: float) -> None:
        direction = self.universal_directions[self.camera.get_axis_index(axis)]
        direction = direction.loc
        shape.force(direction * amount)

    def update_physics(self, gravity: float, drag: float) -> None:
        for shape in self.shapes:
            if isinstance(shape, Physics_Shape3D):
                shape.apply_physics(gravity, drag, self.world_directions)

    def draw(self, screen, camera) -> None:
        for plane in self.floor:
            camera.draw_shape(screen, plane)

        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(len(self.world_directions)):
            point = self.world_directions[i]
            colour = colours[i]
            camera.draw_point(screen, point)
            pygame.draw.line(screen, colour, camera.get_projection(self.world_center), camera.get_projection(point))

        for shape in self.shapes:
            camera.draw_shape(screen, shape)

    def create_plane(self, x: float, y: float, z: float, width: float, height: float) -> Shape3D:
        points = []
        points.append(Point3D(x, y, z))
        points.append(Point3D(x + width, y, z))
        points.append(Point3D(x + width, y, z + height))
        points.append(Point3D(x, y, z + height))
        shape = Shape3D(points)
        shape.connect(points[0], points[1])
        shape.connect(points[1], points[2])
        shape.connect(points[2], points[3])
        shape.connect(points[3], points[0])
        return shape


class Camera:
    def __init__(self, screen) -> None:
        self.loc = np.array([0, 1, 10])
        self.center = np.array([screen.get_width() / 2, screen.get_height() / 2])
        self.screen = screen

    def get_axis_index(self, axis: str) -> int:
        if axis == 'x':
            return 0
        elif axis == 'y':
            return 1
        elif axis == 'z':
            return 2

    def move(self, axis: str, amount: float) -> None:
        self.loc[self.get_axis_index(axis)] += amount
        self.loc[2] = max(1, self.loc[2])
        
    def get_projection(self, point: Point3D) -> tuple: 
        x = point.loc[0]/self.loc[2] if point.loc[0] != 0 else 0
        y = point.loc[1]/self.loc[2] if point.loc[1] != 0 else 0

        multiplier = 1000
        y *= -1
        return (int(x*multiplier + self.center[0]), int(y*multiplier  + self.center[1]))

    def draw_point(self, screen, point: Point3D, colour=(255,255,255)) -> None:
        x, y = self.get_projection(point)
        pygame.draw.circle(screen, colour, (x, y), 1)

    def draw_shape(self, screen, shape: Shape3D) -> None:
        colour = shape.colour
        for point in shape.points:
            self.draw_point(screen, point, colour)
        for i in range(len(shape.connection_matrix)):
            for j in range(len(shape.connection_matrix[i])):
                if shape.connection_matrix[i][j] == 1:
                    pygame.draw.line(screen, colour, self.get_projection(shape.points[i]), self.get_projection(shape.points[j]))

def create_rect_prism(x: float, y: float, z: float, width: float, height: float, depth: float) -> Shape3D:
    points = []
    points.append(Point3D(x, y, z))
    points.append(Point3D(x + width, y, z))
    points.append(Point3D(x + width, y + height, z))
    points.append(Point3D(x, y + height, z))
    points.append(Point3D(x, y, z + depth))
    points.append(Point3D(x + width, y, z + depth))
    points.append(Point3D(x + width, y + height, z + depth))
    points.append(Point3D(x, y + height, z + depth))
    shape = Physics_Shape3D(points)
    shape.connect(points[0], points[1])
    shape.connect(points[1], points[2])
    shape.connect(points[2], points[3])
    shape.connect(points[3], points[0])
    shape.connect(points[4], points[5])
    shape.connect(points[5], points[6])
    shape.connect(points[6], points[7])
    shape.connect(points[7], points[4])
    shape.connect(points[0], points[4])
    shape.connect(points[1], points[5])
    shape.connect(points[2], points[6])
    shape.connect(points[3], points[7])
    return shape


def shape_from_obj(file_name: str) -> Shape3D:
    points = []
    with open(file_name) as file:
        for line in file:
            line = line.split()
            if len(line):
                if line[0] == 'v':
                    points.append(Point3D(float(line[1]), float(line[2]), float(line[3])))
    shape = Shape3D(points)
    with open(file_name) as file:
        for line in file:
            line = line.split()
            if len(line):
                if line[0] == 'f':
                    vertix1 = int(line[1].split('/')[0]) - 1
                    vertix2 = int(line[2].split('/')[0]) - 1
                    vertix3 = int(line[3].split('/')[0]) - 1
                    shape.connect(points[vertix1], points[vertix2])
                    shape.connect(points[vertix2], points[vertix3])
                    shape.connect(points[vertix3], points[vertix1])
    return shape

import pygame

pygame.init()
screen = pygame.display.set_mode((1000, 600), pygame.RESIZABLE)
pygame.display.set_caption('3D Matrix')
clock = pygame.time.Clock()

w = World(screen)

object = shape_from_obj('eb_house_plant_01.obj')
w.add_shape(object)


last_mouse_pos = pygame.mouse.get_pos()
rotation_state = [False, False]
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:            
                rotation_state = [True, True]
                last_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 3:
                rotation_state = [True, False]

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:            
                rotation_state = [False, False]
            elif event.button == 3:
                rotation_state[0] = False

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            if rotation_state[0]:
                x_diff = mouse_pos[0] - last_mouse_pos[0]
                w.rotate('y', x_diff/2)
            if rotation_state[1]:
                y_diff = mouse_pos[1] - last_mouse_pos[1]
                w.rotate('x', y_diff)
            last_mouse_pos = mouse_pos

        elif event.type == pygame.MOUSEWHEEL:
            scroll_amount = event.y * 2
            w.camera.move('z', -scroll_amount)

    screen.fill((0, 0, 0))

    w.update_physics(0.01, 0.9)
    w.draw(screen, w.camera)

    pygame.display.update()
    clock.tick(60)