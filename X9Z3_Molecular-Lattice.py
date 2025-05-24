from vpython import *
#Web VPython 3.2
from random import randint
"""
Created by Maximillian DeMarr.
Future work ideas:
- Explore 'nodictionary' for faster simulations
- Devise structures for graphene nanotubules and Fullerenes
- Devise more dynamic implementation for directions input/interpretation
    - Consider allowing users to enter in their own string of directions
    - Implement Einstein tiles and more advanced structures
Last major update 2025.
"""

# ================================= Canvas =====================================

scene = display(width = 600, height = 580, align='left')
scene.bind('mousedown', move_node)
scene.bind('mouseup', release_mouse_1)


about_me = """
            In the scene to the left, a model simulates vibrations in a lattice. Choose from several  
  pre-designed lattice types using the dropdown labeled <b>'Graphene'</b>. You can adjust the number of  
  nodes (atoms) simulated, but higher values are more computationally expensive. Use <b>'Render: Static'</b>  
  to visualize thousands of atoms without real-time physics. After any change, press <b>'Reload'</b> and  
  <b>left-click anywhere</b> in the scene to restart the simulation.

        The lattice starts at rest. You can <b>move any atom by left-clicking and dragging</b> it.  
  If the vibrations grow too large, reduce energy conservation to cool the system. The default  
  layout uses a breadth-first search (BFS) to form clustered networks. You can switch to a  
  depth-first search (DFS) for long, tendril-like chains. A <b>randomness slider</b> adds variability  
  to direction order. Use <b>middle mouse to zoom</b>, <b>right-click + drag to rotate</b>, and  
  <b>Shift + left-click + drag to pan</b> the view.

        \\(F = -k x_{dist},\\)     \\(F = ma\\)   \\(\\Rightarrow\\)   \\(a = \\dfrac{-k x_{dist}}{m}\\)  
            \\(\\Delta v = a \\Delta t\\)   and   \\(\\Delta x = v \\Delta t\\)

        Vibrations are computed using Hooke’s Law. The force between two atoms is based on  
  the vector \\(x_{dist}\\), and integrated via Newton’s Second Law using a forward Euler method.  
  To mimic sheer forces tangential to a spring, I am storing the original distance vector between
  two nodes and comparing any changes in distance to this original value. This gives the network 
  stiffness and avoids whole-lattice rotation (and internal collapse into a chaotic ball). This
  simplification improves performance without major physical compromise.

        I'm open to implementing new lattice types—reach out if you're interested! The current  
  algorithm uses shared tileable vectors for all nodes, which complicates patterns like Einstein shapes.

        -<i>Created by Maximillian DeMarr</i>
"""

scene.caption = about_me
MathJax.Hub.Queue(["Typeset", MathJax.Hub, scene.caption])  # LaTeX formatting







# ================================ Methods =====================================

def lattice_directions(type):
    """
    Returns a list of directions for generating a lattice based on the specified type.

    This method generates a list of directional vectors used for constructing 
    specific types of lattices. The behavior depends on the `type` parameter, 
    which must be a recognized lattice type.

    Args:
        type (str): The type of lattice to generate directions for. Must be one 
            of the supported lattice types (e.g., 'Graphene', 'Diamond').

    Returns:
        list: A list of directional vectors specific to the requested lattice type. 
        Each element in the list represents a direction relevant to the lattice 
        structure.

    Raises:
        TypeError: If an unsupported or unimplemented lattice type is passed. 
        This is a safeguard and should not occur during normal operation, 
        provided the input strings are valid.
    """
    
    if type == 'Graphene':
        return [vec(sqrt(3)/2, 1/2, 0), vec(-sqrt(3)/2, 1/2, 0), vec(0, -1, 0)]
        
    elif type == 'Diamond':
        A = vec(0,1,0)
        diamond_directions = [A]
        A = rotate(A, axis=vec(1,0,0), angle=radians(109.5))
        diamond_directions.append(A)
        for i in range(3):
            A = rotate(A, axis=vec(0,1,0), angle=radians(120))
            diamond_directions.append(A)
        return diamond_directions
        
    elif type == 'Triangular 2D':
        A = vec(0,1,0)
        triangular_directions = [A]
        for i in range(6):
            A = rotate(A, axis=vec(0,0,1), angle=radians(360/6))
            triangular_directions.append(vec(round(A.x, 5), round(A.y, 5), round(A.z, 5)))
        return triangular_directions
        
    elif type == 'Triangular 3D':
        A = vec(1,0,0)
        B = vec(1/2, sqrt(3)/2, 0)
        C = vec(1/2, 1/(4*sqrt(3)), sqrt(3)/2)
        return [A, B, C, -A, -B, -C]
    
    elif type == 'Grid 2D':
        return [vec(1,0,0), vec(0,1,0), vec(-1,0,0), vec(0,-1,0)]
    
    elif type == 'Grid 3D':
        return [vec(1,0,0), vec(0,1,0), vec(-1,0,0), vec(0,-1,0), vec(0,0,1), vec(0,0,-1)]
        
    else:
        return "'Type' Error?"  # This should never happen, please tell me if it does :D


def shuffle_list(directions):
    """
    Shuffles a list in place using the Fisher-Yates Shuffle algorithm.
    
    Args:
        directions (list): The list of elements to shuffle. Modified in place.
        
    Returns:
        list: The list with the elements shuffled.
        
    Algorithm:
        1. Start from the last element of the list and iterate backward.
        2. For each element, generate a random index from 0 to the current index.
        3. Swap the current element with the randomly chosen element.
    """
    
    # Iterate from the last index down to the second element
    for i in range(len(directions) - 1, 0, -1):
        # Pick a random index between 0 and i (inclusive)
        j = randint(0, i)
        
        # Swap the elements at index i and j using temporaries because of VPython behavior
        element_i = directions[i]
        element_j = directions[j]
        directions[i] = element_j
        directions[j] = element_i
    return directions





# ============================= Widget Methods =================================

def move_node(evt):
    """
    Handles the movement of a node when the mouse is dragged.
    
    This method detects when a node is clicked and dragged. It continuously updates
    the node's position based on the mouse's position while ensuring the node's
    velocity is reset to zero. It also updates the positions and orientations of
    edges connected to the node to maintain the structural integrity of the lattice.
    
    Args:
        evt: The mouse event triggering the function.
    """
    global mouse_1_up
    mouse_1_up = False
    hit = scene.mouse.pick
    
    if isinstance(hit, simple_sphere):
        plane_point = hit.pos
        
        while True:
            rate(60)
            if mouse_1_up:
                break
            
            hit.pos = scene.mouse.project( normal=scene.forward, point=plane_point)
            hit.velocity = vec(0,0,0)
            
            # Stitch the node's attached arrows to the children nodes
            for neighbor, edge in hit.edges.items():
                if neighbor in hit.children:
                    edge.pos = hit.pos
                    edge.axis = neighbor.pos - hit.pos
                else:
                    edge.axis = hit.pos - neighbor.pos


def release_mouse_1(evt):
    """
    Handles the event when the mouse button is released.
    
    This method sets the global flag `mouse_1_up` to True, indicating that the 
    mouse button has been released, and stops the node movement process.
    
    Args:
        evt: The mouse event triggering the function.
    """
    global mouse_1_up
    mouse_1_up = True


def lattice_selection(evt):
    """
    Selects the lattice directions based on the event's selected value. Calls the
    lattice_directions method to convert the evt string to a list of directions.

    Args:
        evt: An event object containing the selected value which determines the lattice directions.
    """
    global lattice
    lattice.directions = lattice_directions(evt.selected)


def change_number_of_nodes(evt):
    """
    Changes the number of nodes based on the event's number and warns the user if the number exceeds 500.

    Args:
        evt: An event object containing the new number of nodes.
    
    Notes:
        Displays a warning if the number of nodes exceeds 500, which may slow performance.
    """
    global frozen, quantity_of_nodes
    if evt.number > 500 and not frozen:
        print("Warning:\nMore than 500 nodes results in horrible performance. Consider toggling 'Static' and rerendering.")
    quantity_of_nodes = evt.number


def static_render(evt):
    """
    Toggles between static and dynamic rendering of the lattice based on the event.

    Args:
        evt: An event object that toggles the state of rendering.
    
    Notes:
        Changes the text on the static render button and toggles between 'Static' and 'Dynamic' states.
    """
    global inert
    inert = not inert
    static_button.text = 'Render: Static' if inert else 'Render: Dynamic'


def about_me_toggle(evt):
    """
    Toggles the display of the 'about me' section and adjusts the scene appearance.

    Args:
        evt: An event object that triggers the toggle.
    
    Notes:
        Changes the button background color and scene width. If the background is light grey, it displays a caption 
        for the user and narrows the scene width. Otherwise, it shows the full scene and resets the background.
    """
    global about_me
    if evt.background.equals(vec(0.8,0.8,0.8)):
        evt.background = color.white
        scene.caption = ''
        scene.width=1300
    else:
        evt.background = vec(0.8,0.8,0.8)
        scene.width = 600
        scene.caption = about_me
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, scene.caption])  # LaTeX formatting


def spring_constant_slide(slide):
    """
    Updates the spring constant based on the slide value.
    
    Args:
        slide: A slide object containing the new spring constant.
        
    Notes:
        The spring constant text is updated to reflect the current spring constant. The value is stored in a global variable.
    """
    global spring_constant
    spring_constant_text.text = f'Spring Constant: {slide.value:5.f}'
    spring_constant = slide.value


def energy_conservation_slide(slide):
    """
    Updates the energy conservation percentage based on the slide value.

    Args:
        slide: A slide object containing the new energy conservation percentage.
    
    Notes:
        The energy conservation text is updated to reflect the percentage. The value is stored in a global variable.
    """
    global conserved_energy
    energy_conservation.text = f'Energy Conservation: {slide.value:3.2f}%'
    conserved_energy = slide.value/100
    

def traversal_type_slide(slide):
    """
    Adjusts the graph traversal method based on the slide value.

    Args:
        slide: A slide object containing the new value for traversal method selection (DFS and BFS percentage).
    
    Notes:
        The traversal method text is updated, and the graph traversal method modifier is adjusted.
    """
    global lattice
    traversal_method.text = f'Graph Traversal Method: DFS = {100-slide.value:3.0f}% | BFS = {slide.value:3.0f}%'
    lattice.generation_modifier = slide.value/100


def randomness_slide(slide):
    """
    Updates the randomness of the lattice generation based on the slide value.

    Args:
        slide: A slide object containing the new value for graph randomness.
    
    Notes:
        The randomness text is updated to reflect the percentage. The lattice's shuffle modifier is adjusted.
    """
    global lattice
    generation_randomness.text = f'Graph Randomness: {slide.value:3.0f}%'
    lattice.shuffle_modifier = slide.value


def reload_lattice(evt):
    """
    Reloads the lattice if not already loading. Prevents multiple reloads during a single operation.

    Args:
        evt: An event object that triggers the lattice reloading action.
    
    Notes:
        If the lattice is already loading or queued, it will print a message asking the user to wait.
    """
    global loading
    if not loading:
        loading = True
        load_it()
    else:
        print("Please wait for scene to reload (need to click anywhere in scene).")







# ================================ Classes =====================================

class Node:
    """
    Represents a single node in the lattice.

    This class defines the properties and behavior of nodes in the lattice. Each
    node is visualized as a sphere and can be connected to other nodes through edges.
    
    Attributes:
        obj (simple_sphere): The graphical representation of the node.
        obj.velocity (vec): The velocity vector of the node.
        obj.edges (dict): A dictionary of edges (key: node_neighbor, value: edge) 
            connected to the node.
        obj.children (dict): A dictionary of edges (key: child_node, value: edge) 
            in which the main node is the owner of.
    """
    def __init__(self, position):
        self.obj = simple_sphere(pos=position, color=color.red, radius=atom_radius)
        self.obj.velocity = vec(0,0,0)
        self.obj.edges = {}
        self.obj.children = {}


class Edge:
    """
    Represents an edge connecting two nodes in the lattice.

    This class defines the properties and behavior of edges, which are visualized
    as cylinders connecting two nodes. Edges maintain the structural integrity
    of the lattice, using the initial_vec parameter.
    
    Attributes:
        node_1 (simple_sphere): The first node connected by the edge.
        node_2 (simple_sphere): The second node connected by the edge, first's child.
        edge (cylinder): The graphical representation of the edge to be recycled
    """
    def __init__(self, node_1, node_2, edge=None):
        if edge:
            edge.pos = node_1.pos
            edge.axis = node_2.pos - node_1.pos
            edge.initial_vec = node_2.pos - node_1.pos
            edge.visible = True
        else:
            edge = cylinder(pos=node_1.pos, axis=node_2.pos-node_1.pos, color=color.white, radius=node_1.radius/4, initial_vec=node_2.pos-node_1.pos)
        
        node_1.edges[node_2] = edge
        node_1.children[node_2] = edge
        node_2.edges[node_1] = edge
        

class Queue:
    """
    A Queue class that supports both BFS (Breadth-First Search) and DFS (Depth-First Search) behaviors 
    based on a behavior modifier.
    
    Attributes:
        items (list): A list to store the elements of the queue.
        behavior_modifier (float): A modifier that determines whether the queue behaves like a 
                                   BFS (1) or DFS (0). Values closeer to 1 favor BFS.
    """
    
    def __init__(self, behavior_modifier=1):
        """
        Initializes the queue with an optional behavior modifier.
        
        Args:
            behavior_modifier (float): Controls the dequeue behavior. Default is 1, which behaves like BFS.
        """
        self.items = []
        self.behavior_modifier = behavior_modifier

    def enqueue(self, item):
        """
        Adds an item to the end of the queue.
        
        Args:
            item: The item to be added to the queue.
        """
        self.items.append(item)

    def dequeue(self):
        """
        Removes and returns an item from the queue based on the behavior modifier.
        
        Returns:
            The item removed from the queue.
        
        Raises:
            IndexError: If attempting to dequeue from an empty queue.
        
        Behavior:
            - For BFS, removes from the front of the queue.
            - For DFS, removes from the end of the queue.
            - For values between 0 and 1, the behavior is probabilistic, leaning towards BFS or DFS.
        """
        # Determine the index to pop based on the behavior_modifier
        index = -1 * round(random()**(self.behavior_modifier * 2)) if self.behavior_modifier < 1 else 0
        
        if (not self.is_empty()):
            return self.items.pop(index)
        raise IndexError("dequeue from empty queue")

    def is_empty(self):
        """
        Checks if the queue is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return len(self.items) == 0


class Lattice:
    """
    Represents a lattice structure composed of interconnected nodes.

    This class handles the generation and management of a lattice using various
    geometric configurations and traversal methods. It supports dynamic node 
    generation with options to shuffle the direction of node placement.

    Attributes:
        directions (list): A list of vectors representing the possible directions
            for node placement in the lattice.
        generation_modifier (float/int): A value influencing the traversal method,
            where 1 is purely BFS and 0 is purely DFS.
        shuffle_modifier (float/int): A probability that controls the randomization 
            of the directions list during node generation. Again 0 to 1.
        origin (Node): The starting node of the lattice, typically at the origin.
        dead_edges (list): A list of edges that have been deactivated and can be reused.
        web (list): The list of active nodes forming the lattice.
    """
    
    def __init__(self, directions=None, generation_modifier=1, shuffle_modifier=0):
        """
        Initializes the Lattice with default or specified directions and modifiers.

        Args:
            directions (list, optional): A list of direction vectors for lattice construction.
                Defaults to a 2D hexagonal configuration.
            generation_modifier (float, optional): The modifier for traversal type.
                Defaults to 1 (BFS-like behavior).
            shuffle_modifier (float, optional): The modifier for randomizing direction order.
                Defaults to 0 (no shuffling).
        """
        # Default directions for a hexagonal lattice if none provided
        if directions is None:
            directions = [vec(sqrt(3)/2, 1/2, 0), vec(-sqrt(3)/2, 1/2, 0), vec(0, -1, 0)]
        self.directions = [axis * spacing for axis in directions]
        self.generation_modifier = generation_modifier
        self.shuffle_modifier = shuffle_modifier
        self.origin = None
        self.dead_edges = []
        self.web = []

    def add_node(self, node):
        """
        Adds a node to the lattice.

        Args:
            node (Node): The node to be added to the lattice.
        """
        self.web.append(node)

    def connect_nodes(self, node_1, node_2):
        """
        Connects two nodes with an edge.

        Reuses an edge from `dead_edges` if available or creates a new one.

        Args:
            node_1 (Node): The first node.
            node_2 (Node): The second node to be connected with the first.
        """
        edge_to_recycle = self.dead_edges.pop() if len(self.dead_edges) > 0 else None
        Edge(node_1, node_2, edge_to_recycle)

    def generate_nodes(self, n):
        """
        Generates `n` nodes in the lattice using a BFS-like or DFS-like approach.
        If the web already exists then it gets spliced and recycled.

        Args:
            n (int): The number of nodes to generate in the lattice.
        """
        nodes_to_recycle = self.__splice_web(n) if len(self.web) > 0 else []
        queue = Queue(self.generation_modifier)
        visited = {}

        # Initialize the first node at the origin
        if nodes_to_recycle == []:
            node = Node(vec(0,0,0)).obj
        else:
            node = nodes_to_recycle.pop(0)
            self.__reset_node(node, vec(0,0,0))
            
        self.origin = node
        self.add_node(node)
        queue.enqueue((node, 1))  # (Node, flip Boolean) flip boolean is used to handle lattice asymmetry
        visited[str(node.pos)] = node
        
        while (not queue.is_empty()) and len(self.web) < n:
            node, flip = queue.dequeue()
            
            # Optionally shuffle the directions
            if self.shuffle_modifier and random() < self.shuffle_modifier:
                self.directions = shuffle_list(self.directions)
            
            for direction in self.directions:
                next_position = vec(node.pos.x + direction.x * flip, node.pos.y + direction.y * flip, node.pos.z + direction.z * flip)
                blurred_position = vec(round(next_position.x, 5), round(next_position.y, 5), round(next_position.z, 5))
                
                """Check if this blurred position exists as a key in the visited dict. If the next node has already been visited,
                then we check to add a bond between the nodes. """
                if str(blurred_position) not in visited:
                    if nodes_to_recycle == []:
                        next_node = Node(next_position).obj
                    else:
                        next_node = nodes_to_recycle.pop(0)
                        self.__reset_node(next_node, next_position)
                    
                    self.connect_nodes(node, next_node)
                    self.add_node(next_node)
                    visited[str(blurred_position)] = next_node
                    queue.enqueue((next_node, -flip))
                    if len(self.web) >= n:
                        break
                elif visited.get(str(blurred_position)) not in node.edges:
                    self.connect_nodes(node, visited.get(str(blurred_position)))
    
    def __splice_web(self, n):
        """
        Splices the current lattice into two parts, retaining the first `n` nodes.
        None retained part gets sent to node deletion.
        
        Args:
            n (int): The number of nodes to retain.
        
        Returns:
            list: The nodes to be recycled for the new lattice.
        """
        nodes_to_rewrite = self.web[:-(len(self.web) - n)] if len(self.web) > n else self.web[:]
        nodes_to_delete = self.web[len(nodes_to_rewrite):]
        self.__delete_nodes(nodes_to_delete)
        self.web = []
        return nodes_to_rewrite
    
    def __delete_nodes(self, nodes_to_delete):
        """
        Deletes nodes by resetting their position and hiding them, effectively 
        removing their computational impact in memory.

        Args:
            nodes_to_delete (list): The nodes to be deleted.
        """
        for node in nodes_to_delete:
            node.pos = vec(0,0,0)
            node.velocity = vec(0,0,0)
            node.visible = False
            self.__reset_edges(node)
    
    def __reset_node(self, node, position):
        """
        Resets a node to a given position and reinitializes its properties.

        Args:
            node (Node): The node to reset.
            position (vector): The new position of the node.
        """
        node.pos = position
        node.velocity = vec(0,0,0)
        node.visible = True
        self.__reset_edges(node)
    
    def __reset_edges(self, node):
        """
        Resets the edges of a node, making them invisible and storing them for reuse.

        Args:
            node (Node): The node whose edges are to be reset.
        """
        for child, edge in node.children.items():
            edge.visible = False
            self.dead_edges.append(edge)
        node.edges = {}
        node.children = {}
    
    def __str__(self):
        """
        Returns a string representation of the lattice.

        Returns:
            str: A string indicating the number of nodes in the lattice.
        """
        return f"HexGrid with {len(self.web)} nodes."


def load_it():
    """This is the method to reload the lattice and restart the main() method."""
    global lattice, engine_running, quantity_of_nodes, inert, loading
    
    reload_button.text = "Loading"
    engine_running = False
    sleep(0.05)
    
    lattice.generate_nodes(quantity_of_nodes)
    print_options(clear=True)
    reload_button.text = "Loaded "
    print("Waiting for user to click anywhere in scene...")
    scene.waitfor('click')
    
    print_options(clear=True)
    print("Running and clicked :)")
    engine_running = not inert
    sleep(0.01)
    loading = False
    reload_button.text = "Reload "
    main()







# ================================ Globals =====================================

mass = 1
spacing = 1
atom_radius = 0.25
conserved_energy = 1
spring_constant = 1000
restoration_constant = 0.3

lattice_type = "Graphene"
quantity_of_nodes = 40
time_step = 0.0005
update_rate = 1/time_step

bfs_like = 1
shuffling_percent = 0

mouse_1_up = True
engine_running = True
inert = False
loading = False

lattice = Lattice()
lattice.generate_nodes(quantity_of_nodes)







# =============================== Widgets/Text =================================

lattice_type_menu = menu(bind=lattice_selection, choices=['Graphene', 'Diamond', 'Triangular 2D', 'Triangular 3D', 'Grid 2D', 'Grid 3D'], selected=lattice_type, index=0, pos=scene.title_anchor)
scene.append_to_title('   ')

node_quantity = winput(prompt='Nodes:', bind=change_number_of_nodes, type='numeric', text=str(quantity_of_nodes), number=quantity_of_nodes, width=35, pos=scene.title_anchor)

scene.append_to_title(' ')
reload_button = button(bind=reload_lattice, text='Reload ', pos=scene.title_anchor)

scene.append_to_title(' ')
static_button = button(bind=static_render, text='Render: Dynamic', pos=scene.title_anchor)

scene.append_to_title(' ')
caption_clear_button = button(bind=about_me_toggle, text='Explanation', pos=scene.title_anchor, background=vec(0.8,0.8,0.8))

scene.append_to_title('    ')
spring_constant_text = wtext(text=f'Spring Constant: {spring_constant}', pos=scene.title_anchor)
spring_constant_slider = slider(bind=spring_constant_slide, min=100, max=10000, step=100, value=spring_constant, length=150, pos=scene.title_anchor)


scene.append_to_title('|\n')
energy_conservation = wtext(text='Energy Conservation: 100.00%', pos=scene.title_anchor)
energy_conservation_slider = slider(bind=energy_conservation_slide, min=90, max=100, step=0.05, value=100, length=150, pos=scene.title_anchor)

scene.append_to_title('|        ')
traversal_method = wtext(text='Graph Traversal Method: DFS = 0% | BFS = 100%', pos=scene.title_anchor)
traversal_type_slider = slider(bind=traversal_type_slide, min=0, max=100, step=5, value=100, length=150, pos=scene.title_anchor)

scene.append_to_title('|        ')
generation_randomness = wtext(text='Graph Randomness: 0%', pos=scene.title_anchor)
randomness_slider = slider(bind=randomness_slide,  min=0, max=100, step=1, value=0, length=150, pos=scene.title_anchor)
scene.append_to_title('|\n')







# ================================== Main ======================================

main()

def main():
    global lattice, time_step, engine_running, conserved_energy, spring_constant
    time = 0
    while engine_running:
        rate(update_rate)
        
        """For each node, we calculate it's net change in velocity, by using Hooke's Law. For derivations, we begin with Hooke's Law,
        F = -k * x = m * a = m * (Δv/Δt)
        Δv = F * Δt / m
        therefore,
        Δv = -k * x * Δt / m
        In our application, we absorb the negative sign into the x vector, flipping it.
        """
        for node in lattice.web:
            net_displacement = vec(0,0,0)
            
            for neighbor, edge in node.edges.items():
                displacement_vector = neighbor.pos - node.pos
                net_displacement += displacement_vector - norm(displacement_vector) * spacing
                
                """These next two lines serve as the components which prevent the lattice from collapsing in on itself. Without this
                the bonds have no idea of what orientation they should favor. The most realistic model would be checking the bond
                angle formed from on neighbor between the next and creating a counter force based off of this. But this requires
                iterating through all of the neighbors in an n! or n^2 manner to determine who is the closest or the net result of
                all of these non-initial bond angles. To save on performance I ignore this and implement and idealized solution."""
                # Give the neighbor a restoring force to its ideal relative position.
                idealized_neighbor_vector = edge.initial_vec if neighbor in node.children else -edge.initial_vec
                neighbor.velocity += restoration_constant * spring_constant * (idealized_neighbor_vector - displacement_vector) * time_step / mass
            
            node.velocity += spring_constant * net_displacement * time_step / mass
            node.velocity *= conserved_energy
            
        """Update node positions by integrating the new velocity. Also update the edges."""
        for node in lattice.web:
            node.pos += node.velocity * time_step
            
            # This update order isn't 100% correct but it is close enough and saves performance.
            for child, edge in node.children.items():
                    edge.pos = node.pos
                    edge.axis = child.pos - node.pos
                    
        time += time_step



# End of program