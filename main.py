import pygame
from queue import PriorityQueue
import numpy as np
from igraph import Graph
from movement_algorithms import Kinematic, KinematicSeek, KinematicFlee

# Define game window dimensions and grid properties
HEIGHT, WIDTH = 848, 848
ROWS, COLS = 16, 16
GRID_SIZE = WIDTH // COLS
SPEED_BASE = GRID_SIZE / 16

# Initialize Pygame and set up the game window
pygame.init()
pygame.font.init() 
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pokemon")

# Define button properties
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50
BUTTON_POS = (10, 10)  # Position of the button on the screen
BUTTON_COLOR = (0, 0, 255)  # Blue color for the button
BUTTON_HOVER_COLOR = (0, 100, 255)  # Darker blue when hovered
BUTTON_TEXT_COLOR = (255, 255, 255)  # White text color
BUTTON_FONT = pygame.font.SysFont('Arial', 20)  # Font for button text

# Font for other text elements
FONT = pygame.font.SysFont('Arial', 28)

# Load game images
PATH_IMG = pygame.image.load('path.jpg')
WALL_IMG = pygame.image.load('wall.png')
CHARACTER_IMG = pygame.image.load('pikachu.png')
ENEMY1_IMG = pygame.image.load('enemy1.png')
ENEMY2_IMG = pygame.image.load('enemy2.png')  # Enemy 2 Image
ENEMY3_IMG = pygame.image.load('enemy3.png')  # Enemy 3 Image
THUNDERSTONE_IMG = pygame.image.load('thunderstone.png')
RAICHU_IMG = pygame.image.load('raichu.png')

# Scale images to fit the grid size
THUNDERSTONE_IMG = pygame.transform.scale(THUNDERSTONE_IMG, (GRID_SIZE, GRID_SIZE))
PATH_IMG = pygame.transform.scale(PATH_IMG, (GRID_SIZE, GRID_SIZE))
WALL_IMG = pygame.transform.scale(WALL_IMG, (GRID_SIZE, GRID_SIZE))
CHARACTER_IMG = pygame.transform.scale(CHARACTER_IMG, (GRID_SIZE, GRID_SIZE))
ENEMY1_IMG = pygame.transform.scale(ENEMY1_IMG, (GRID_SIZE, GRID_SIZE))
ENEMY2_IMG = pygame.transform.scale(ENEMY2_IMG, (GRID_SIZE, GRID_SIZE))  # Scale Enemy 2 Image
ENEMY3_IMG = pygame.transform.scale(ENEMY3_IMG, (GRID_SIZE, GRID_SIZE))  # Scale Enemy 3 Image
RAICHU_IMG = pygame.transform.scale(RAICHU_IMG, (GRID_SIZE, GRID_SIZE))  # Scale Raichu image

# Load music files for characters
PIKACHU_MUSIC = 'pikachu.mp3'
RAICHU_MUSIC = 'raichu.mp3'

# Start playing Pikachu's music on loop
pygame.mixer.music.load(PIKACHU_MUSIC)
pygame.mixer.music.play(-1)

class Button:
    """Class to create and manage the button to show the connections"""    
    def __init__(self, x, y, width, height, text, 
                 color, hover_color, text_color, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.hovered = False

    def draw(self, win):
        """Draws the button on the game window."""
        # Change color if hovered
        current_color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(win, current_color, self.rect)

        # Render and center the button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        win.blit(text_surface, text_rect)
    
    def is_hovered(self, mouse_pos):
        """Checks if the mouse is hovering over the button."""
        return self.rect.collidepoint(mouse_pos)

class Character(Kinematic):
    """Class representing the player's character."""
    def __init__(self, position):
        super().__init__(
            position=position,
            orientation=0.0,
            velocity=np.array([0.0, 0.0]),
            rotation=0.0
        )
        
        self.target_position = None  # Position the character is moving towards
        self.hp = 1000
        self.is_transformed = False  # Transformation state (Pikachu/Raichu)
        self.transform_start_time = None  # Timestamp when transformation began
        self.original_image = CHARACTER_IMG  # Pikachu's image
        self.transformed_image = RAICHU_IMG  # Raichu's image
        self.current_image = self.original_image  # Current image displayed

    def draw(self, win):
        """Draws the character and its HP on the game window."""
        win.blit(self.current_image, (int(self.position[0]), int(self.position[1])))
        
        # Render HP text above the character
        hp_text = FONT.render(f"HP: {self.hp}", True, (255, 255, 255))
        text_rect = hp_text.get_rect(center=(self.position[0] + GRID_SIZE // 2, self.position[1]))
        win.blit(hp_text, (text_rect.x, text_rect.y - 20))  # Offset above the character        

    def set_target_position(self, target_node):
        """Sets the target position for the character based on a grid node."""
        self.target_position = np.array([target_node.col * GRID_SIZE, target_node.row * GRID_SIZE])

    def update(self, time):
        """Updates the character's state and movement."""
        # Handle transformation duration
        if self.is_transformed:
            current_time = pygame.time.get_ticks()
            if current_time - self.transform_start_time >= 10000:  # 10 seconds
                self.revert_transformation()
        
        if self.target_position is not None:
            # Calculate direction and distance to target
            direction = self.target_position - self.position
            distance = np.linalg.norm(direction)

            # If close enough, snap to target
            if distance < SPEED_BASE * time:
                self.position = self.target_position
                self.target_position = None
            else:
                # Move towards target
                direction = direction / distance
                self.position += direction * SPEED_BASE * time

    def transform(self):
        """Transform Pikachu into Raichu."""
        if not self.is_transformed:
            self.is_transformed = True
            self.transform_start_time = pygame.time.get_ticks()
            self.current_image = self.transformed_image
            self.play_raichu_music()

    def revert_transformation(self):
        """Revert Raichu back to Pikachu."""
        self.is_transformed = False
        self.current_image = self.original_image
        self.play_pikachu_music()

    def is_out_of_bounds(self, position, grid):
        """Checks if the given position is out of the grid or into a barrier."""
        row = int(position[1] // GRID_SIZE)
        col = int(position[0] // GRID_SIZE)
        if row < 0 or row >= ROWS or col < 0 or col >= COLS:
            return True
        if grid[row][col].is_barrier:
            return True
        return False

    def play_pikachu_music(self):
        """Plays Pikachu's music."""
        pygame.mixer.music.stop()
        pygame.mixer.music.load(PIKACHU_MUSIC)
        pygame.mixer.music.play(-1)  # Loop indefinitely

    def play_raichu_music(self):
        """Plays Raichu's music."""
        pygame.mixer.music.stop()
        pygame.mixer.music.load(RAICHU_MUSIC)
        pygame.mixer.music.play(-1)  # Loop indefinitely

class Enemy(Kinematic):
    """Class representing enemy characters."""
    def __init__(self, position, target, image):
        super().__init__(position=position, orientation=0.0, velocity=np.array([0.0, 0.0]), rotation=0.0)
        self.seek_behavior = KinematicSeek(self, target, maxSpeed=0.2)
        self.flee_behavior = KinematicFlee(self, target, maxSpeed=0.2)
        self.current_behavior = self.seek_behavior
        self.hp = 50 
        self.max_hp = self.hp
        self.image = image  # Enemy's image
    
    def draw(self, win):
        """Draws the enemy and its HP on the game window."""
        win.blit(self.image, (int(self.position[0]), int(self.position[1])))        
        # Render HP text above the enemy
        hp_text = FONT.render(f"HP: {self.hp}", True, (255, 0, 0))  # Red color for visibility
        text_rect = hp_text.get_rect(center=(self.position[0] + GRID_SIZE // 2, self.position[1]))
        win.blit(hp_text, (text_rect.x, text_rect.y - 20))  # Offset above the enemy           

    def update(self, time, enemies, grid):
        """Updates the enemy's behavior and movement."""
        # Determine if Pikachu is active based on transformation
        isPikachu = not self.seek_behavior.target.is_transformed
        self.decision_tree(isPikachu)

        steering = self.current_behavior.getSteering()

        # Prevent enemies from moving through walls
        wall_steering = self.compute_wall_collision(steering, time, grid)
        if wall_steering is not None:
            steering.linear += wall_steering

        # Apply separation to avoid crowding
        separation = self.compute_separation(enemies)
        if separation is not None:
            steering.linear += separation

        super().update(steering, time)

    def get_random_position(self, grid):
        """Finds a random position on the grid that's not a barrier."""
        while True:
            row = np.random.randint(0, ROWS)
            col = np.random.randint(0, COLS)
            node = grid[row][col]
            if not node.is_barrier:
                return np.array([node.x, node.y], dtype=float)

    def respawn(self, grid):
        """Respawns the enemy at a new random position and resets HP."""
        self.position = self.get_random_position(grid)
        self.hp = self.max_hp

    def compute_separation(self, enemies):
        """Calculates a steering force to separate from nearby enemies."""
        desired_separation = (3 * GRID_SIZE) / 2 
        steer = np.array([0.0, 0.0])
        count = 0
        for other in enemies:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < desired_separation and distance > 0:
                    diff = self.position - other.position
                    diff /= distance
                    steer += diff
                    count += 1
        if count > 0:
            steer /= count
            max_force = SPEED_BASE * 0.05
            if np.linalg.norm(steer) > max_force:
                steer = (steer / np.linalg.norm(steer)) * max_force
            return steer
        return None

    def compute_wall_collision(self, steering, time, grid):
        """Calculates a steering force to avoid walls."""
        desired_distance = GRID_SIZE * 3  # Threshold to start avoiding walls
        steer = np.array([0.0, 0.0])
        count = 0

        # Check proximity to each wall and adjust steering accordingly
        if self.position[1] < desired_distance:
            steer[1] += 1  # Move down
            count += 1
        if self.position[1] > HEIGHT - desired_distance:
            steer[1] -= 1  # Move up
            count += 1
        if self.position[0] < desired_distance:
            steer[0] += 1  # Move right
            count += 1
        if self.position[0] > WIDTH - desired_distance:
            steer[0] -= 1  # Move left
            count += 1

        if count > 0:
            steer = steer / count
            steer = steer * (SPEED_BASE * 0.1)  # Scale the force
            return steer

        return None

    def decision_tree(self, isPikachu):
        """Determines enemy behavior based on character's transformation."""
        if isPikachu:
            self.current_behavior = self.seek_behavior
            # Calculate distance to Pikachu
            distance = np.linalg.norm(self.position - self.seek_behavior.target.position)
            if distance < GRID_SIZE:
                # Inflict damage to Pikachu
                if self.seek_behavior.target.hp <= 1:
                    game_over_screen(WIN)
                else:
                    self.seek_behavior.target.hp -= 1
        else:
            self.current_behavior = self.flee_behavior
            distance = np.linalg.norm(self.position - self.flee_behavior.target.position)
            if distance < GRID_SIZE:
                self.hp -= 5  # Enemy takes damage when too close

class Thunderstone:
    """Class representing the Thunderstone item in the game."""
    def __init__(self, grid):
        self.grid = grid
        self.image = THUNDERSTONE_IMG
        self.position = self.get_random_position()
    
    def get_random_position(self):
        """Finds a random position on the grid that's not a barrier."""
        while True:
            row = np.random.randint(0, ROWS)
            col = np.random.randint(0, COLS)
            node = self.grid[row][col]
            if not node.is_barrier:
                return np.array([node.x, node.y])
    
    def draw(self, win):
        """Draws the Thunderstone on the game window."""
        win.blit(self.image, (int(self.position[0]), int(self.position[1])))
    
    def reposition(self):
        """Moves the Thunderstone to a new random position."""
        self.position = self.get_random_position()

class Node:
    """Class representing a single node (cell) in the grid."""
    def __init__(self, row, col, is_barrier=False):
        self.row = row
        self.col = col
        self.x = row * GRID_SIZE
        self.y = col * GRID_SIZE
        self.is_barrier = is_barrier

    def get_pos(self):
        """Returns the node's position as a tuple."""
        return self.row, self.col

    def draw(self, win):
        """Draws the node on the game window."""
        if self.is_barrier:
            win.blit(WALL_IMG, (self.x, self.y))
        else:
            win.blit(PATH_IMG, (self.x, self.y))

    def make_barrier(self):
        """Sets the node as a barrier."""
        self.is_barrier = True

    def update_neighbors(self, grid):
        """Updates the list of neighboring nodes that are not barriers."""
        self.neighbors = []
        if not self.is_barrier:
            # Check and add neighbors in all four directions
            if self.row < ROWS - 1 and not grid[self.row + 1][self.col].is_barrier:
                self.neighbors.append(grid[self.row + 1][self.col])
            if self.row > 0 and not grid[self.row - 1][self.col].is_barrier:
                self.neighbors.append(grid[self.row - 1][self.col])
            if self.col < COLS - 1 and not grid[self.row][self.col + 1].is_barrier:
                self.neighbors.append(grid[self.row][self.col + 1])
            if self.col > 0 and not grid[self.row][self.col - 1].is_barrier:
                self.neighbors.append(grid[self.row][self.col - 1])
            
    def __repr__(self):
        """Returns a string representation of the node."""
        return f"Node(row={self.row}, col={self.col}, is_barrier={self.is_barrier})"

def h(p1, p2):
    """Heuristic function for A* (Manhattan distance)."""
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def a_star(draw, grid, start, end):
    """Implements the A* pathfinding algorithm."""
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float('inf') for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            return reconstruct_path(came_from, end)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)

    return False

def game_over_screen(win):
    """Displays the game over screen and exits the game after a delay."""
    win.fill((0, 0, 0))  # Clear the screen with black
    game_over_text = FONT.render("Game Over", True, (255, 0, 0))  # Red text
    text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    win.blit(game_over_text, text_rect)
    pygame.display.update()
    # Wait for 3 seconds before closing
    pygame.time.wait(3000)
    pygame.quit()
    exit()

def reconstruct_path(came_from, current):
    """Reconstructs the path from A* search."""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

def make_grid(rows, cols):
    """Creates a grid of nodes."""
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            node = Node(i, j)
            grid[i].append(node)
    return grid

def add_walls(grid):
    """Adds walls around the borders of the grid."""
    for row in range(ROWS):
        for col in range(COLS):
            if row == 0 or row == ROWS - 1 or col == 0 or col == COLS - 1:
                grid[row][col].make_barrier()

def define_maze(grid):
    """Defines a complex maze structure by adding walls."""
    walls = [
        (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10), (13, 10),
        (10, 13), (10, 12), (10, 11), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 3), (10, 2), (10, 1),
        (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (11, 6), (12, 6), (13, 6),
    ]

    # Add border walls
    for row in range(ROWS):
        walls.append((row, 0))
        walls.append((row, COLS - 1))
    for col in range(COLS):
        walls.append((0, col))
        walls.append((ROWS - 1, col))

    # Add all defined walls to the grid
    for (row, col) in walls:
        grid[row][col].make_barrier()

def draw(win, grid, rows, cols, character, enemies, show_connections, graph, thunderstone, kill_counter, toggle_button):
    """Draws all game elements on the window."""
    
    # Draw the background by tiling the path image
    for i in range(0, WIDTH, GRID_SIZE):
        for j in range(0, HEIGHT, GRID_SIZE):
            win.blit(PATH_IMG, (i, j))
    
    # Draw all grid nodes
    for row in grid:
        for node in row:
            node.draw(win)
            
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(win, (255, 255, 255), (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
    
    # Draw the character, enemies, and Thunderstone
    character.draw(win)
    for enemy in enemies:
        enemy.draw(win)
    thunderstone.draw(win)
    
    # Display the kill counter on the top-right
    kill_text = FONT.render(f"Kills: {kill_counter}", True, (255, 255, 255))
    text_rect = kill_text.get_rect(topright=(WIDTH - 10, 10))
    win.blit(kill_text, text_rect)
    
    # Optionally draw connections between nodes
    if show_connections:
        for row in grid:
            for node in row:
                if node.is_barrier:
                    continue  # Skip walls
                for neighbor in node.neighbors:
                    if neighbor.is_barrier:
                        continue  # Skip wall neighbors
                    pygame.draw.line(
                        win,
                        (255, 255, 0),  # Yellow color for connections
                        (node.x + GRID_SIZE // 2, node.y + GRID_SIZE // 2),
                        (neighbor.x + GRID_SIZE // 2, neighbor.y + GRID_SIZE // 2),
                        1
                    )
    
    # Draw the toggle connections button
    toggle_button.draw(win)    
    
    # Refresh the display
    pygame.display.update()

def get_clicked_pos(pos):
    """Converts mouse position to grid coordinates."""
    y, x = pos
    row = y // GRID_SIZE
    col = x // GRID_SIZE
    return row, col

def main(win, width):
    """Main game loop."""
    clock = pygame.time.Clock()
    grid = make_grid(ROWS, COLS)
    define_maze(grid)
    graph = Graph()
    graph.add_vertices(ROWS * COLS)
    
    # Update neighbors for each node
    for row in grid:
        for node in row:
            node.update_neighbors(grid)
    
    # Add edges to the graph based on node neighbors, excluding walls
    for row in grid:
        for node in row:
            if node.is_barrier:
                continue  # Skip wall nodes
            node_index = node.row * COLS + node.col  # Unique index for each node
            for neighbor in node.neighbors:
                if neighbor.is_barrier:
                    continue  # Skip wall neighbors
                neighbor_index = neighbor.row * COLS + neighbor.col
                # Prevent duplicate edges in an undirected graph
                if node_index < neighbor_index:
                    graph.add_edge(node_index, neighbor_index)
    
    # Initialize the character in the center of the screen
    character = Character(position=np.array([width // 2, width // 2], dtype=float))
    
    # Initialize enemies at different positions
    enemies = [
        Enemy(position=np.array([width // 4, width // 4], dtype=float), target=character, image=ENEMY1_IMG),
        Enemy(position=np.array([3 * width // 4, 3 * width // 4], dtype=float), target=character, image=ENEMY2_IMG),
        Enemy(position=np.array([width // 2, 3 * width // 4], dtype=float), target=character, image=ENEMY3_IMG)
    ]
    
     # Initialize the Thunderstone item
    thunderstone = Thunderstone(grid) 

    start = None
    end = None
    show_connections = False
    path = []
    global kill_counter  # Global variable to track kills
    kill_counter = 0

    # Initialize the toggle connections button
    toggle_button = Button(
        x=BUTTON_POS[0],
        y=BUTTON_POS[1],
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT,
        text="Show Connections",
        color=BUTTON_COLOR,
        hover_color=BUTTON_HOVER_COLOR,
        text_color=BUTTON_TEXT_COLOR,
        font=BUTTON_FONT
    )

    run = True
    while run:
        mouse_pos = pygame.mouse.get_pos()
        toggle_button.hovered = toggle_button.is_hovered(mouse_pos)

        # Change cursor icon when hovering over the button
        if toggle_button.is_hovered(mouse_pos):
            pygame.mouse.set_cursor(*pygame.cursors.diamond)
        else:
            pygame.mouse.set_cursor(*pygame.cursors.arrow)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if toggle_button.is_hovered(mouse_pos):
                    show_connections = not show_connections
                    toggle_button.text = "Hide Connections" if show_connections else "Show Connections"
                else:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos)
                    start_row = int(character.position[0] // GRID_SIZE)
                    start_col = int(character.position[1] // GRID_SIZE)
                    start = grid[start_row][start_col]
                    end = grid[row][col]
                    path = a_star(None, grid, start, end)  # Start pathfinding

        clock.tick(30)  # Limit the game to 30 FPS

        # Move the character along the path
        if path:
            next_node = path.pop(0)
            character.position = np.array([next_node.row * GRID_SIZE, next_node.col * GRID_SIZE], dtype=float)

        character.update(5)  # Update character position with a fixed time step
        for enemy in enemies:
            enemy.update(20, enemies, grid)
        
        # Check collision with Thunderstone to trigger transformation
        distance = np.linalg.norm(character.position - thunderstone.position)
        if distance < GRID_SIZE:
            thunderstone.reposition()
            character.transform()  # Transform to Raichu

            # Set all enemies to flee
            for enemy in enemies:
                enemy.decision_tree(False)

         # Handle enemy behavior based on character's transformation status
        if character.is_transformed:
            for enemy in enemies:
                enemy.decision_tree(False)
                if enemy.hp <= 0:
                    kill_counter += 1
                    enemy.respawn(grid)
        else:
            for enemy in enemies:
                enemy.decision_tree(True)
                if enemy.hp <= 0:
                    kill_counter += 1
                    enemy.respawn(grid)                

        # Draw all game elements
        draw(win, grid, ROWS, COLS, character, enemies, show_connections, graph, thunderstone, kill_counter, toggle_button)

    pygame.quit()

# Start the game
main(WIN, WIDTH)