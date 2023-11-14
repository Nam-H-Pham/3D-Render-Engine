import numpy as np

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

    def move(self, direction_vector: np.array) -> None:
        self.loc = self.loc + direction_vector

    def __str__(self):
        return str(self.loc)


class Shape3D:
    def __init__(self, points: list) -> None:
        self.points = points
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

    def move(self, axis: str, direction_vector: np.array) -> None:
        for point in self.points:
            point.move(direction_vector)

    def __str__(self):
        return str(self.points)

class World:
    def __init__(self, screen) -> None:
        self.shapes = []
        self.camera = Camera(screen)
        self.floor = []
        self.create_grid(1, 10)

        self.world_center = Point3D(0, 0, 0)
        self.world_directions = [Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]


    def create_grid(self, square_size: float, side_num: int) -> None:
        offset = (side_num * square_size) / 2
        for i in range(side_num):
            for j in range(side_num):
                self.floor.append(create_plane(i*square_size-offset, 0, j*square_size-offset, square_size, square_size))
    
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
        shape.move(axis, direction * amount)

    def draw(self, screen, camera) -> None:

        for plane in self.floor:
            camera.draw_shape(screen, plane, colour=(50, 50, 50))

        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(len(self.world_directions)):
            point = self.world_directions[i]
            colour = colours[i]
            camera.draw_point(screen, point)
            pygame.draw.line(screen, colour, camera.get_projection(self.world_center), camera.get_projection(point))

        for shape in self.shapes:
            camera.draw_shape(screen, shape)


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

    def draw_shape(self, screen, shape: Shape3D, colour=(255,255,255)) -> None:
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
    shape = Shape3D(points)
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

def create_plane(x: float, y: float, z: float, width: float, height: float) -> Shape3D:
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

def create_triangle(x: float, y: float, z: float, width: float, height: float, depth: float) -> Shape3D:
    points = []
    points.append(Point3D(x, y, z))
    points.append(Point3D(x + width, y, z))
    points.append(Point3D(x + width, y + height, z))
    shape = Shape3D(points)
    shape.connect(points[0], points[1])
    shape.connect(points[1], points[2])
    shape.connect(points[2], points[0])
    return shape

import pygame

pygame.init()
screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('3D Matrix')
clock = pygame.time.Clock()

rect = create_rect_prism(0.5, 0, 0.5, 1, 1, 1)
tri = create_triangle(-1, -1, -1, 1, 1, 1)
tri.rotate('y', 90)
rect_small = create_rect_prism(-2, -0.5, -0.5, 0.5, 0.1, 1)

w = World(screen)
w.add_shape(rect)
w.add_shape(tri)
w.add_shape(rect_small)

movement_amount = 0.1

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
            scroll_amount = event.y
            w.camera.move('z', -scroll_amount)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        w.move_shape_axis_aligned(rect, 'z', -movement_amount)
    if keys[pygame.K_s]:
        w.move_shape_axis_aligned(rect, 'z', movement_amount)
    if keys[pygame.K_a]:
        w.move_shape_axis_aligned(rect, 'x', -movement_amount)
    if keys[pygame.K_d]:
        w.move_shape_axis_aligned(rect, 'x', movement_amount)
    if keys[pygame.K_SPACE]:
        w.move_shape_axis_aligned(rect, 'y', movement_amount)
    if keys[pygame.K_LSHIFT]:
        w.move_shape_axis_aligned(rect, 'y', -movement_amount)

    tri.rotate('y', 1)
    tri.rotate('x', 1)

    rect_small.rotate('y', 0.5)
        
    screen.fill((0, 0, 0))

    w.draw(screen, w.camera)

    pygame.display.update()
    clock.tick(60)