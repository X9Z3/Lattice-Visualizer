from vpython import *
#Web VPython 3.2
from random import randint
"""
Created by Maximillian DeMarr.
Future work ideas:
- Explore 'nodictionary' for faster simulations (unsure if this even works?)
- Devise structures for graphene nanotubules and Fullerenes
- Devise a more dynamic implementation for directions input/interpretation
    - Consider allowing users to enter in their own string of directions
    - Implement Einstein tiles and more advanced structures

Last major update: 7/2/2025.
"""

# ================================= Canvas =====================================

scene = display(width = 600, height = 580, align='left')
scene.bind('mousedown', move_node)
scene.bind('mouseup', release_mouse_1)


about_me = """
            In the scene to the left, a model simulates vibrations in a lattice.
  You can move any atom by <b>left-clicking and dragging</b> it. After any change,
  or if the lattice flies away, press <b>'Reload'</b> to restart the simulation.
  Choose from several pre-designed lattice types using the dropdown labeled 
  <b>'Graphene'</b>. You can adjust the number of atoms simulated, but higher values
  are computationally expensive. Click <b>'Render: Dynamic'</b> to switch to 
  static mode, allowing for smooth rendering of thousands of atoms without 
  real-time physics. A Chrome browser seems to run best and anything beyond
  500 atoms, I suggest using the static renderer.

            If the vibrations grow too large, reduce <b>energy conservation</b> to
  cool the system. The default generation method uses a breadth-first search (BFS)
  to form clustered networks. You can increase depth-first search (DFS) 
  behavior for long, tendril-like chains. A <b>randomness slider</b> adds variability  
  to direction order. <b>Zoom with scroll wheel</b>, <b>right-click + drag to rotate</b>, and  
  <b>Shift + left-click + drag to pan</b> the view.


  ┌── Physics Discussion ──┐

            Vibrations are computed using a standard Hooke's Law spring approximation.
  The attracting or repelling force between two atoms is converted into an acceleration
  using Newton's Second Law, which is then integrated via a forward Euler method to
  update positions. To mimic shear forces tangential to a spring, I store the original
  distance vector between two nodes and compare any changes in distance to this baseline.
  This adds stiffness to the network and prevents whole-lattice rotation or collapse into
  a chaotic ball. The simplification improves performance with minimal physical compromise. 

            I'm open to implementing new lattice types—reach out if you're interested!
  The current algorithm uses shared, tileable vectors for all nodes, which complicates 
  patterns like Einstein shapes. I would also like to implement ABX₃ (perovskite) 
  lattice structures, but that may require a complete overhaul of the physics engine.
  Doing it justice might involve simulating electric force interactions, which would 
  defeat the purpose of using spring approximations for performance gains.

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
        D = vec(B.x, -B.y, B.z)
        E = vec(-C.x, C.y, C.z)
        return [A, B, C, D, E, -A, -B, -C, -D, -E]
    
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


def load_it():
    """Reload the lattice and restart the main simulation loop."""
    global lattice, engine_running, quantity_of_nodes, inert

    reload_button.text = "Loading"
    engine_running = False
    sleep(0.05)

    # Generate the new lattice nodes
    lattice.generate_nodes(quantity_of_nodes)

    # Set scene range based on new lattice span
    scene.autoscale = False
    scene.range = mag(lattice.web[-1].pos - lattice.web[0].pos) + 1

    # Wait for lattice to finish loading
    while lattice.is_loading:
        rate(30)

    # Reset simulation settings
    print_options(clear=True)
    engine_running = not inert

    # Brief autoscale toggle to update visuals
    scene.autoscale = True
    sleep(0.01)
    scene.autoscale = False

    reload_button.text = "Reload"
    main()









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
    if evt.number > 300 and not frozen:
        print("For more nodes, consider toggling 'Static' and rerendering.")
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
    frozen_simulation_button.text = 'Render: Static' if inert else 'Render: Dynamic'


def simulation_pause_run(evt):
    """
    Toggles the physics simulation between paused and running states.
    
    Updates both the simulation state and the triggering button's appearance
    to provide clear visual feedback about the current mode.
    
    Args:
        evt: The button event object triggering the toggle
        
    Global Dependencies:
        paused (bool): Tracks simulation pause state
        
    Effects:
        - Toggles global paused state
        - Updates button text and color (red when paused)
    """
    global paused
    paused = not paused
    if paused:
        evt.text = "Paused"
        evt.textcolor = color.red
    else:
        evt.text = "Running"
        evt.textcolor = color.black


def about_me_toggle(evt):
    """
    Toggles the display of an informational 'About Me' section with dynamic layout adjustments.
    
    This function manages:
    - Toggling between full scene view and about me view
    - Adjusting scene width based on current display mode
    - Updating button visual feedback
    - Rendering LaTeX-formatted content when displayed
    
    Args:
        evt: Event object triggering the toggle
        
    Global Dependencies:
        about_me (str): Content to display in the about me section
        caption_enabled (bool): Tracks if about me content is currently displayed
        about_me_button (button): Reference to the UI button triggering this function
        graphs_enabled (bool): Tracks if graph visualization is active
        
    Effects:
        - Modifies scene width and caption content
        - Updates button background color as visual feedback
        - Triggers MathJax rendering for LaTeX content
    """
    global about_me, caption_enabled
    caption_enabled = not caption_enabled
    if not caption_enabled:
        evt.text = "Explain me!"
        scene.width = 1000
        scene.caption = ''
    else:
        evt.text = "Hide words "
        scene.width = 600
        scene.caption = about_me
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, scene.caption])


def change_spring_constant(evt):
    """
    Updates the spring constant based on the slide value.
    
    Args:
        evt: A slide object containing the new spring constant.
        
    Notes:
        The spring constant text is updated to reflect the current spring constant. The value is stored in a global variable.
    """
    global spring_constant
    if evt.obj_type == 'slider':
        evt.value_winput_obj.text = evt.value
        spring_constant = evt.value
    elif evt.obj_type == 'winput':
        spring_constant = evt.text


def change_energy_conservation(evt):
    """
    Updates the energy conservation percentage based on the slide value.

    Args:
        evt: A slide object containing the new energy conservation percentage.
    
    Notes:
        The energy conservation text is updated to reflect the percentage. The value is stored in a global variable.
    """
    global conserved_energy
    if evt.obj_type == 'slider':
        evt.value_winput_obj.text = evt.value
        conserved_energy = evt.value
    elif evt.obj_type == 'winput':
        conserved_energy = evt.text
    

def change_traversal_type(evt):
    """
    Adjusts the graph traversal method based on the slide value.

    Args:
        evt: A slide object containing the new value for traversal method selection (DFS and BFS percentage).
    
    Notes:
        The traversal method text is updated, and the graph traversal method modifier is adjusted.
    """
    global lattice
    evt.title_wtext_obj.text = f'<b>Generation Method: <font color="green">DFS = {100-evt.value}%</font> | <font color="blue">BFS = {evt.value}%</font></b>'
    lattice.generation_modifier = evt.value/100


def change_generation_randomness(evt):
    """
    Updates the randomness of the lattice generation based on the slide value.

    Args:
        evt: A slide object containing the new value for graph randomness.
    
    Notes:
        The randomness text is updated to reflect the percentage. The lattice's shuffle modifier is adjusted.
    """
    global lattice
    if evt.obj_type == 'slider':
        evt.value_winput_obj.text = evt.value
        lattice.shuffle_modifier = evt.value
    elif evt.obj_type == 'winput':
        lattice.shuffle_modifier = evt.text


def reload_lattice(evt):
    """
    Initiates lattice reload. Mostly deprecated.

    Args:
        evt: An event object that triggers the lattice reloading action.
    """
    load_it()






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
    def __init__(self, position=vec(0,0,0), node_color=color.red, node_radius=None):
        if node_radius == None:
            node_radius = atom_radius
        self.obj = simple_sphere(pos=position, color=node_color, radius=node_radius)
        self.obj.velocity = vec(0,0,0)
        self.obj.edges = {}
        self.obj.children = {}
        self.obj.neighbor_interactions = []


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
            edge.visible = True
        else:
            edge = cylinder(pos=node_1.pos, axis=node_2.pos-node_1.pos, color=color.white, radius=atom_radius/4)
        
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
        self.is_loading = False  # Used to check when the loading process is done
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
        self.is_loading = True
        nodes_to_recycle = self.__splice_web(n) if len(self.web) > 0 else []
        queue = Queue(self.generation_modifier)
        visited = {}

        # Initialize the first node at the origin
        if len(nodes_to_recycle) == 0:
            node = Node(position=vec(0,0,0)).obj
        else:
            node = nodes_to_recycle.pop(0)
            self.__reset_node(node, vec(0,0,0))
            
        self.origin = node
        self.add_node(node)
        queue.enqueue((node, 1))  # (Node, flip Boolean) flip boolean is used to handle lattice asymmetry
        visited[str(node.pos)] = node  # Using the position as the key to check for later
        
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
                        next_node = Node(position=next_position).obj
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

        self.__fill_angular_rigidity()
        self.is_loading = False
        
    
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
        node.neighbor_interactions = []
    
    def __fill_angular_rigidity(self):
        """
        For each node in the lattice, this method initializes 'neighbor_interactions'
        that represent second-degree (angular) connections between pairs of neighbors.

        These virtual "springs" (edges) are added between each pair of neighbors of the node,
        effectively introducing angular rigidity (resistance to bending) to the lattice.

        Each entry in node.neighbor_interactions is a list:
            [[neighbor1, neighbor2], rest_length]
        where rest_length is the initial distance between the two neighbor nodes.

        This is useful for maintaining lattice shape under deformation.
        """
        for node in self.web:
            if len(node.edges) < 2:
                # Skip nodes with fewer than two neighbors — no angular interaction possible
                continue

            # Get a list of this node's neighbors
            neighbors = list(node.edges.keys())
            num_neighbors = len(neighbors)

            # Initialize the list to store angular (second-degree) interactions
            node.neighbor_interactions = []

            for i in range(num_neighbors):
                # Form a "spring" between neighbor i and neighbor i+1 (with wrap-around)
                neighbor1 = neighbors[i]
                neighbor2 = neighbors[(i + 1) % num_neighbors]
                neighbor3 = neighbors[(i - 1) % num_neighbors]

                # Compute initial (rest) length of the virtual spring
                rest_length1 = mag(neighbor1.pos - neighbor2.pos)
                rest_length2 = mag(neighbor2.pos - neighbor3.pos)

                # Append the interaction
                node.neighbor_interactions.append([[neighbor1, neighbor2], rest_length1])
                node.neighbor_interactions.append([[neighbor2, neighbor3], rest_length2])

    
    def __str__(self):
        """
        Returns a string representation of the lattice.

        Returns:
            str: A string indicating the number of nodes in the lattice.
        """
        return f"HexGrid with {len(self.web)} nodes."









# ============================== Global Settings ===============================

# --- Physical Parameters ---
mass                   = 1
spacing                = 1
atom_radius            = 0.25
spring_constant        = 1000
restoration_constant   = 0.3        # A value used for second degree neighbors
conserved_energy       = 85

# --- Simulation Settings ---
lattice_type           = "Graphene"
quantity_of_nodes      = 40
time_step              = 0.0005
update_rate            = int(1 / time_step)

# --- Lattice Generation Controls ---
bfs_like               = 1          # 1 = BFS, 0 = DFS
shuffling_percent      = 0          # Directional randomness (0–100)

# --- Engine State Flags ---
mouse_1_up             = True
engine_running         = True
inert                  = False
loading                = False
caption_enabled        = True

# --- Main Lattice Object ---
lattice                = Lattice()




# =============================== Widgets/Text ================================

lattice_type_menu = menu(
    choices=['Graphene', 'Diamond', 'Triangular 2D', 'Triangular 3D', 'Grid 2D', 'Grid 3D'],
    title_wtext_obj=wtext(text='<b>Model:</b> ', pos=scene.title_anchor),
    bind=lattice_selection, selected=lattice_type, index=0,
    pos=scene.title_anchor, suffix_wtext_obj=None
)
lattice_type_menu.suffix_wtext_obj = wtext(text='  ', pos=scene.title_anchor)

node_quantity_winput = winput(
    prompt='<b>Nodes:</b>', bind=change_number_of_nodes, type='numeric',
    text=str(quantity_of_nodes), number=quantity_of_nodes, width=35,
    pos=scene.title_anchor
)
node_quantity_winput.suffix_wtext_obj = wtext(text=' ', pos=scene.title_anchor)

reload_button = button(
    text='Reload', pos=scene.title_anchor,
    bind=reload_lattice
)
reload_button.suffix_wtext_obj = wtext(text=' ', pos=scene.title_anchor)

run_pause_button = button(  # ▣/▶ Toggles simulation execution
    text="Running", pos=scene.title_anchor,
    bind=simulation_pause_run
)
run_pause_button.suffix_wtext_obj = wtext(text=' ', pos=scene.title_anchor)

frozen_simulation_button = button(
    text='Render: Dynamic', pos=scene.title_anchor,
    bind=static_render
)
frozen_simulation_button.suffix_wtext_obj = wtext(text=' ', pos=scene.title_anchor)

about_me_button = button(  # 🔍 Toggles descriptive text
    text='Hide words ', pos=scene.title_anchor,
    bind=about_me_toggle
)

spring_constant_slider = slider(
    min=100, max=10000, step=100, value=spring_constant,
    length=150, pos=scene.title_anchor,
    bind=change_spring_constant,
    title_wtext_obj=wtext(text=f'\n<b>Spring Constant:</b>', pos=scene.title_anchor),
    obj_type='slider'
)
spring_constant_slider.value_winput_obj = winput(
    type='numeric', width=40, height=20,
    text=spring_constant_slider.value,
    pos=scene.title_anchor,
    bind=change_spring_constant,
    obj_type='winput'
)
spring_constant_slider.suffix_wtext_obj = wtext(text=' ', pos=scene.title_anchor)

energy_conservation_slider = slider(
    min=0, max=100, step=0.5, value=conserved_energy,
    length=150, pos=scene.title_anchor,
    bind=change_energy_conservation,
    title_wtext_obj=wtext(text=f'<b>Energy Conservation:</b>', pos=scene.title_anchor),
    obj_type='slider'
)
energy_conservation_slider.value_winput_obj = winput(
    type='numeric', width=30, height=20,
    text=energy_conservation_slider.value,
    pos=scene.title_anchor,
    bind=change_energy_conservation,
    obj_type='winput'
)
energy_conservation_slider.suffix_wtext_obj = wtext(text='%\n', pos=scene.title_anchor)

traversal_type_slider = slider(
    min=0, max=100, step=5, value=100,
    length=150, pos=scene.title_anchor,
    bind=change_traversal_type,
    title_wtext_obj=wtext(
        text=(
            '<b>Generation Method: '
            '<font color="green">DFS = 0%</font> | '
            '<font color="blue">BFS = 100%</font></b>'
        ),
        pos=scene.title_anchor
    ),
    obj_type='slider'
)
traversal_type_slider.suffix_wtext_obj = wtext(text='     ', pos=scene.title_anchor)

randomness_slider = slider(
    min=0, max=100, step=1, value=0,
    length=150, pos=scene.title_anchor,
    bind=change_generation_randomness,
    title_wtext_obj=wtext(text=f'<b>Generation Randomness:</b>', pos=scene.title_anchor),
    obj_type='slider'
)
randomness_slider.value_winput_obj = winput(
    type='numeric', width=30, height=20,
    text=randomness_slider.value,
    pos=scene.title_anchor,
    bind=change_generation_randomness,
    obj_type='winput'
)
randomness_slider.suffix_wtext_obj = wtext(text='%\n', pos=scene.title_anchor)







# ================================== Main ======================================

def main():
    """
    Runs the core physics simulation loop using Hooke’s Law and explicit Euler integration.

    Forces are computed as:
        F = -k * (x - x₀)
    Where:
        - x is the current displacement vector between nodes,
        - x₀ is the original rest length vector (stored in edge.initial_vec),
        - k is the spring constant.

    Acceleration is then calculated using Newton’s second law:
        F = m * a  =>  a = F / m

    Velocity is updated via:
        v += a * Δt  =>  v += (F * Δt) / m

    Position is integrated using forward Euler:
        x += v * Δt

    A restoring force based on the original bond vector helps prevent collapse by adding angular stiffness,
    which would otherwise require complex, angle-based constraints.
    """
    global lattice, time_step, engine_running, conserved_energy, spring_constant
    global paused

    time = 0

    while engine_running:
        rate(update_rate)

        if paused:
            continue  # Skip physics update if simulation is paused

        # ------------------- Compute net forces for each node -------------------
        for node in lattice.web:
            net_displacement = vec(0, 0, 0)

            for neighbor in node.edges.keys():
                current_distance = neighbor.pos - node.pos
                net_displacement += current_distance - norm(current_distance) * spacing

            # Evaluate second-degree neighbor distances
            for nodes, equilibrium_distance in node.neighbor_interactions:
                node_1, node_2 = nodes[0], nodes[1]
                
                current_distance = node_1.pos - node_2.pos
                node_1.velocity -= restoration_constant * spring_constant * (current_distance - norm(current_distance) * equilibrium_distance) * time_step / mass
                node_2.velocity += restoration_constant * spring_constant * (current_distance - norm(current_distance) * equilibrium_distance) * time_step / mass

            # Apply total spring force to node
            node.velocity += spring_constant * net_displacement * time_step / mass

        # ------------------- Integrate velocity and update positions -------------------
        for node in lattice.web:
            # Apply artificial damping based on energy conservation
            damping = ((100 - conserved_energy) / 100) ** 4
            node.velocity *= (1 - damping)

            # Forward Euler integration
            node.pos += node.velocity * time_step

            # Update edge visuals (only from parent to child)
            for child, edge in node.children.items():
                edge.pos = node.pos
                edge.axis = child.pos - node.pos

        time += time_step


# Bootstraps the simulation on load
load_it()
main()

# ================================= End of Program ==============================
