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

    def __str__(self):
        return str(self.loc)

class Shape3D:
    def __init__(self, points: list) -> None:
        self.points = points
        self.connection_matrix = np.zeros((len(points), len(points)))

    def connect(self, point1: Point3D, point2: Point3D) -> None:
        self.connection_matrix[self.points.index(point1)][self.points.index(point2)] = 1
        self.connection_matrix[self.points.index(point2)][self.points.index(point1)] = 1

    def rotate(self, axis: str, angle: float) -> None:
        for point in self.points:
            point.rotate(axis, angle)

    def __str__(self):
        return str(self.points)

class Camera:
    def __init__(self, screen) -> None:
        self.loc = np.array([0, 0, 10])
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
        self.loc[2] = 1 if self.loc[2] == 0 else self.loc[2]
        
    def get_projection(self, point: Point3D) -> tuple: 
        # x = point.loc[0]
        # y = point.loc[1]
        # return (int(x * 100 + self.center[0]), int(y * 100 + self.center[1]))

        # https://stackoverflow.com/questions/6139451/how-can-i-convert-3d-space-coordinates-to-2d-space-coordinates

        x = point.loc[0]/self.loc[2] if point.loc[0] != 0 else 0
        y = point.loc[1]/self.loc[2] if point.loc[1] != 0 else 0

        multiplier = 1000
        return (int(x*multiplier + self.center[0]), int(y*multiplier  + self.center[1]))

    def draw_point(self, screen, point: Point3D) -> None:
        x, y = self.get_projection(point)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 1)

    def draw_shape(self, screen, shape: Shape3D) -> None:
        for point in shape.points:
            self.draw_point(screen, point)
        for i in range(len(shape.connection_matrix)):
            for j in range(len(shape.connection_matrix[i])):
                if shape.connection_matrix[i][j] == 1:
                    pygame.draw.line(screen, (255, 255, 255), self.get_projection(shape.points[i]), self.get_projection(shape.points[j]))

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
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption('3D Matrix')
clock = pygame.time.Clock()

camera = Camera(screen)
point = Point3D(0, 100, 0)

rect = create_rect_prism(0, 0, 0, 1, 1, 1)
tri = create_triangle(-1, -1, -1, 1, 1, 1)
tri.rotate('y', 90)
rect_small = create_rect_prism(-2, -0.5, -0.5, 0.5, 0.1, 1)
floor = create_plane(-1, 1, -1, 5, 5)

shapes = [rect, tri, rect_small, floor]

def rotate_all(axis: str, angle: float) -> None:
    for shape in shapes:
        shape.rotate(axis, angle)


rotation_amount = 1
movement_amount = 1

last_mouse_pos = pygame.mouse.get_pos()
rotation_state = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:            
                rotation_state = True
                last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:            
                rotation_state = False

        elif event.type == pygame.MOUSEMOTION:
            if rotation_state:
                mouse_pos = pygame.mouse.get_pos()
                x_diff = mouse_pos[0] - last_mouse_pos[0]
                y_diff = mouse_pos[1] - last_mouse_pos[1]
                rotate_all('y', x_diff)
                rotate_all('x', -y_diff)
                last_mouse_pos = mouse_pos

        elif event.type == pygame.MOUSEWHEEL:
            scroll_amount = event.y
            camera.move('z', -scroll_amount)


    screen.fill((0, 0, 0))

    for shape in shapes:
        camera.draw_shape(screen, shape)

    pygame.display.update()
    clock.tick(60)