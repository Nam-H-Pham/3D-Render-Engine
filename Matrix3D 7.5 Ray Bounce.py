import numpy as np
from abc import ABC, abstractmethod
import matplotlib.image
from tqdm import tqdm
import threading
from PIL import Image

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
    def __init__(self, points: list = [], colour=(255,255,255)) -> None:
        self.points = points
        self.colour = colour
        self.tris = []

    def get_axis_index(self, axis: str) -> int:
        if axis == 'x':
            return 0
        elif axis == 'y':
            return 1
        elif axis == 'z':
            return 2

    def create_tri(self, point1: Point3D, point2: Point3D, point3: Point3D) -> None:
        self.tris.append([point1, point2, point3])

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
        self.screen = screen
        self.ray_tracing = False
        self.fill_faces = True

        self.world_center = Point3D(0, 0, 0)
        self.world_directions = [Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]

        self.universal_directions = [Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)]
    
    def add_shape(self, shape: Shape3D) -> None:
        self.shapes.append(shape)
    
    def rotate(self, axis: str, angle: float) -> None:
        for point in self.world_directions:
            point.rotate(axis, angle)
        for shape in self.shapes:
            shape.rotate(axis, angle)

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

    def draw(self, camera) -> None:
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(len(self.world_directions)):
            point = self.world_directions[i]
            colour = colours[i]
            camera.draw_point(point)
            pygame.draw.line(self.screen , colour, camera.get_projection(self.world_center), camera.get_projection(point))

        if self.ray_tracing:
            camera.trace_pixels(self)
            self.ray_tracing = False
        else:
            camera.draw_all_edges(self)
            if self.fill_faces:
                camera.draw_all_faces(self)


    def camera_pixel_in_world(self, x: float, y: float) -> Shape3D:
        x = x - self.camera.center[0]
        y = y - self.camera.center[1]
        x = x / 1000 * self.camera.loc[2]
        y = y / 1000 * self.camera.loc[2]
        y *= -1
        return np.array([x, y, self.camera.loc[2]])

    def create_ray(self, position: np.array, direction: np.array) -> Shape3D:
        x, y, z = position
        points = []
        points.append(Point3D(x, y, z))
        points.append(Point3D(x + direction[0], y + direction[1], z + direction[2]))
        shape = Shape3D(points)
        shape.create_tri(points[0], points[1], points[1])
        return shape


class Camera:
    def __init__(self, screen) -> None:
        self.loc = np.array([0, 1, 10])
        self.screen = screen
        self.create_center(screen)
        self.ray_bounce = 1
        self.bounce_overlay_alpha = 0.2
        self.sample_reduction = 1 # 1 = no reduction, 2 = 2x2 reduction, 3 = 3x3 reduction, etc.
        self.debug = False

    def create_center(self, screen) -> None:
        self.center = np.array([self.screen.get_width() / 2, self.screen.get_height()*1.5/ 4])

    def move_center(self, x: float, y: float) -> None:
        self.center = self.center + np.array([x, y])

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

    def draw_point(self, point: Point3D, colour=(255,255,255)) -> None:
        x, y = self.get_projection(point)
        pygame.draw.circle(self.screen, colour, (x, y), 1)

    def draw_shape(self, shape: Shape3D) -> None:
        colour = shape.colour
        for tri in shape.tris:
            pygame.draw.polygon(self.screen, colour, [self.get_projection(point) for point in tri])

    def draw_all_edges(self, world) -> None:
        for shape in world.shapes:
            for tri in shape.tris:
                for i in range(3):
                    pygame.draw.line(self.screen, shape.colour, self.get_projection(tri[i]), self.get_projection(tri[(i+1)%3]))

    def draw_all_faces(self, world) -> None:
        complete_shape = Shape3D()
        for shape in world.shapes:
            for point in shape.points:
                complete_shape.points.append(point)
            complete_shape.tris = complete_shape.tris + shape.tris

        shape = complete_shape

        tris_and_depths = []
        for tri in shape.tris:
            depth = np.mean([point.loc[2] for point in tri])
            tris_and_depths.append((tri, depth))

        tris_and_depths = sorted(tris_and_depths, key=lambda x: x[1])
        farthest_depth = tris_and_depths[0][1]
        closest_depth = tris_and_depths[-1][1]
        for tri, depth in tris_and_depths:
            intensity = 255 - (depth - closest_depth) / (farthest_depth - closest_depth) * 255
            colour = (intensity, intensity, intensity)
            pygame.draw.polygon(self.screen, colour, [self.get_projection(point) for point in tri])

    def trace_pixels(self, world) -> None:
        complete_shape = Shape3D()
        for shape in world.shapes:
            for point in shape.points:
                complete_shape.points.append(point)
            complete_shape.tris = complete_shape.tris + shape.tris

        shape = complete_shape

        tri_bounds = []
        for tri in shape.tris:
            lowest_x = min([self.get_projection(point)[0] for point in tri])
            highest_x = max([self.get_projection(point)[0] for point in tri])
            lowest_y = min([self.get_projection(point)[1] for point in tri])
            highest_y = max([self.get_projection(point)[1] for point in tri])
            tri_bounds.append((lowest_x, highest_x, lowest_y, highest_y))

        image_ray_position_data = np.zeros((self.screen.get_width(), self.screen.get_height(), 3)) # Camera pixel in world coordinates
        image_ray_direction_data = np.zeros((self.screen.get_width(), self.screen.get_height(), 3)) # Camera pixel in world direction
        for x in range(self.screen.get_width()):
            for y in range(self.screen.get_height()):
                image_ray_position_data[x][y] = world.camera_pixel_in_world(x, y)
                image_ray_direction_data[x][y] = np.array([0, 0, -1])

        def render_thread(self, world, shape: Shape3D, x_portion_start, x_portion_end, y_portion_start, y_portion_end, render_pass=0):
            for x in tqdm(range(x_portion_start, x_portion_end), desc="Processing", unit="iteration", total=x_portion_end - x_portion_start, mininterval=0.5):
                for y in range(y_portion_start, y_portion_end):

                    if self.sample_reduction > 1:
                        if x % self.sample_reduction != 0 or y % self.sample_reduction != 0:
                            continue

                    final_pos = None
                    final_distance = 0
                    final_i = 0

                    if render_pass != 0:
                        previous_distance = render_passes_results[-1][x][y]
                        previous_pos = image_ray_position_data[x][y]

                        if previous_distance == 0:
                            continue
                    else:
                        previous_distance = 0
                        previous_pos = world.camera_pixel_in_world(x, y)
                        

                    for i in range(len(shape.tris)):
                        lowest_x, highest_x = tri_bounds[i][0], tri_bounds[i][1]
                        lowest_y, highest_y = tri_bounds[i][2], tri_bounds[i][3]

                        if render_pass == 0:
                            if lowest_x > x_portion_end or highest_x < x_portion_start or lowest_y > y_portion_end or highest_y < y_portion_start:
                                continue
                            

                        tri = shape.tris[i]
                        intersection_point = self.ray_triangle_intersection(image_ray_position_data[x][y], image_ray_direction_data[x][y], tri)
                        if intersection_point is not None:
                            distance = np.linalg.norm(previous_pos - intersection_point)
                        
                            if final_distance == 0 or distance <= final_distance:
                                final_distance = distance
                                final_pos = intersection_point
                                final_i = i
                                

                    if final_pos is not None:

                        if self.debug:
                            self.sample_reduction = 50
                            intial_to_final = final_pos - image_ray_position_data[x][y]
                            rayout = world.create_ray(image_ray_position_data[x][y], intial_to_final)
                            rayout.colour = (255 - render_pass * 50, 0,0)
                            world.add_shape(rayout)

                        image_ray_position_data[x][y] = final_pos

                        if render_pass != 0:
                            final_distance = previous_distance + final_distance
                        image_distance_data[x][y] = final_distance

                        incoming_ray_direction = image_ray_direction_data[x][y]
                        
                        i = final_i
                        normal = np.cross(shape.tris[i][1].loc - shape.tris[i][0].loc, shape.tris[i][2].loc - shape.tris[i][0].loc)
                        normal = normal / np.linalg.norm(normal)
                        if normal[2] > 0:
                            normal = -normal
                        outgoing_ray_direction = incoming_ray_direction - 2 * np.dot(incoming_ray_direction, normal) * normal
                        
                        image_ray_direction_data[x][y] = outgoing_ray_direction
                    

        slices = self.screen.get_width()//80
        x_portion_size = self.screen.get_width() // slices
        x_portions = [(i * x_portion_size, (i+1) * x_portion_size) for i in range(slices)]
        x_portions[-1] = (x_portions[-1][0], self.screen.get_width())

        y_portion_size = self.screen.get_height() // slices
        y_portions = [(i * y_portion_size, (i+1) * y_portion_size) for i in range(slices)]
        y_portions[-1] = (y_portions[-1][0], self.screen.get_height())
        
        threads = []

        render_passes_results = []
        for render_pass in range(self.ray_bounce + 1):
            image_distance_data = np.zeros((self.screen.get_width(), self.screen.get_height())) # distance

            for x in range(slices):
                for y in range(slices):
                    x_portion_start, x_portion_end = x_portions[x]
                    y_portion_start, y_portion_end = y_portions[y]
                    thread = threading.Thread(target=render_thread, args=(self, world, shape, x_portion_start, x_portion_end, y_portion_start, y_portion_end, render_pass))
                    thread.start()
                    threads.append(thread)

            for thread in threads:
                thread.join()

            print(">> Render Pass", render_pass, "Complete")
            image_distance_data_copy = np.copy(image_distance_data)
            render_passes_results.append(image_distance_data_copy)


        print(">> Render Complete")

        final_render_image_passes = []

    
        largest_distance = np.max(render_passes_results[0]) * 1.2 # brightness boost > 1
        min_distance = np.min(render_passes_results[0])

        for render_pass in range(self.ray_bounce + 1):

            image_data = np.zeros((self.screen.get_width(), self.screen.get_height()))
            image_distance_data = render_passes_results[render_pass]

            for x in range(self.screen.get_width()):
                for y in range(self.screen.get_height()):
                    if image_distance_data[x][y] != 0:
                        distance = image_distance_data[x][y]
                        intensity = 255 - (distance - min_distance) / (largest_distance - min_distance) * 255

                        if intensity <= 0 or distance > largest_distance:
                            continue
                            
                        image_data[x][y] = intensity

            final_render_image_passes.append(image_data)

        image_names = []
        for index in range(len(final_render_image_passes)):
            final_render_image_passes[index] = np.rot90(final_render_image_passes[index], 3)
            final_render_image_passes[index] = np.flip(final_render_image_passes[index], 1)
            name = 'test' + str(index) + '.png'
            matplotlib.image.imsave(name, final_render_image_passes[index], cmap='gray')
            image_names.append(name)

        final_render = final_render_image_passes[0]
        for index in range(1, len(final_render_image_passes)):
            final_render = self.overlay_images(final_render, final_render_image_passes[index], self.bounce_overlay_alpha)
        
        matplotlib.image.imsave('test.png', final_render, cmap='gray')


    def ray_triangle_intersection(self, ray_origin, ray_direction, tri):
        epsilon = 1e-6

        vertex0, vertex1, vertex2 = [point.loc for point in tri]
        edge1 = vertex1 - vertex0
        edge2 = vertex2 - vertex0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None  # Ray is parallel to the triangle

        f = 1.0 / a
        s = ray_origin - vertex0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None  # Intersection point is outside the triangle

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None  # Intersection point is outside the triangle

        t = f * np.dot(edge2, q)

        if t > epsilon:
            intersection_point = ray_origin + t * ray_direction
            return intersection_point

        return None  # No intersection in front of the ray origin
        
    def overlay_images(self, background, overlay, alpha, x=0, y=0):
        # Ensure the images have the same shape
        assert background.shape == overlay.shape

        # Convert uint8 to float
        background = background.astype(float)
        overlay = overlay.astype(float)

        # Image ranges from 0 to 255
        background /= 255
        overlay /= 255

        # Compute the weighted average
        background = background * (1 - alpha)
        overlay = overlay * alpha
        output = background + overlay

        # Convert back to uint8
        output *= 255
        output = output.astype(np.uint8)

        return output




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
    shape.create_tri(points[0], points[1], points[2])
    shape.create_tri(points[2], points[3], points[0])
    shape.create_tri(points[1], points[5], points[6])
    shape.create_tri(points[6], points[2], points[1])
    shape.create_tri(points[5], points[4], points[7])
    shape.create_tri(points[7], points[6], points[5])
    shape.create_tri(points[4], points[0], points[3])
    shape.create_tri(points[3], points[7], points[4])
    shape.create_tri(points[3], points[2], points[6])
    shape.create_tri(points[6], points[7], points[3])
    shape.create_tri(points[4], points[5], points[1])
    shape.create_tri(points[1], points[0], points[4])
    return shape

def create_plane(x: float, y: float, z: float, width: float, height: float) -> Shape3D:
    points = []
    points.append(Point3D(x, y, z))
    points.append(Point3D(x + width, y, z))
    points.append(Point3D(x + width, y, z + height))
    points.append(Point3D(x, y, z + height))
    shape = Shape3D(points)
    shape.create_tri(points[0], points[1], points[2])
    shape.create_tri(points[2], points[3], points[0])
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
                    line = line[1:]
                    if len(line) == 3:
                        vertex1 = int(line[0].split('/')[0]) - 1
                        vertex2 = int(line[1].split('/')[0]) - 1
                        vertex3 = int(line[2].split('/')[0]) - 1
                        shape.create_tri(points[vertex1], points[vertex2], points[vertex3])

                    else:
                        number_of_vertices = len(line)
                        for i in range(number_of_vertices - 2):
                            vertex1 = int(line[0].split('/')[0]) - 1
                            vertex2 = int(line[i+1].split('/')[0]) - 1
                            vertex3 = int(line[i+2].split('/')[0]) - 1
                            shape.create_tri(points[vertex1], points[vertex2], points[vertex3])

    return shape

import pygame

pygame.init()
screen = pygame.display.set_mode((300, 125), pygame.RESIZABLE)
pygame.display.set_caption('3D Matrix')
clock = pygame.time.Clock()

w = World(screen)

object = shape_from_obj('sphere.obj')
object.move(np.array([2, 0, 0]))
w.add_shape(create_rect_prism(-2, 0, -2, 8, 8, 0))
w.add_shape(create_plane(-2, 0, -2, 8, 8))
w.add_shape(object)

# w.add_shape(create_rect_prism(0, 0, 0, 1, 1, 1))
# w.add_shape(create_rect_prism(-2, 0, -2, 4, 4, 0))
# w.add_shape(create_plane(-2, 0, -2, 4, 4))


last_mouse_pos = pygame.mouse.get_pos()
rotation_state = False
center_movement_state = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            # show size of window in title bar
            pygame.display.set_caption('3D Matrix' + ' ' + str(event.w) + 'x' + str(event.h))
            w.camera.create_center(screen)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # Left click      
                rotation_state = True
            elif event.button == 3: # Right click
                center_movement_state = True
            last_mouse_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left click            
                rotation_state = False
            elif event.button == 3: # Right click
                center_movement_state = False

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = pygame.mouse.get_pos()
            if rotation_state:
                x_diff = mouse_pos[0] - last_mouse_pos[0]
                w.rotate('y', x_diff/2)
                y_diff = mouse_pos[1] - last_mouse_pos[1]
                w.rotate('x', y_diff)
            elif center_movement_state:
                x_diff = mouse_pos[0] - last_mouse_pos[0]
                w.camera.move_center(x_diff, 0)
                y_diff = mouse_pos[1] - last_mouse_pos[1]
                w.camera.move_center(0, y_diff)
            last_mouse_pos = mouse_pos

        elif event.type == pygame.MOUSEWHEEL:
            scroll_amount = event.y * 1
            w.camera.move('z', -scroll_amount)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                w.ray_tracing = True

            elif event.key == pygame.K_f:
                w.fill_faces = not w.fill_faces

            elif event.key == pygame.K_d:
                w.debug = not w.debug
                if w.debug:
                    w.ray_tracing = True

    screen.fill((0, 0, 0))

    #w.update_physics(0.01, 0.9)
    w.draw(w.camera)

    pygame.display.update()
    clock.tick(60)