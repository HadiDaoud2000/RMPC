# RMPC

this repository represent a course work in the subject RMPC

**Problem statement:**

this project's main goal is to achieve path planning, trajectory planning, and motion control for Franka Emika 7DoF robotic manipulator with static obstacles avoidance using different algorithms (PF, RRT_connect, RRT*, RL) for path planning, (joint space interpolation) for trajectory planning, and finally computed torque controller as a motion controller.

**Path Planning Algorithms:**
The path planning of robotic arms in complex and narrow environments with multiple obstacles poses challenges, and path-planning algorithms can generally be categorized into three types: graph-based search, deep learning-based, and sampling-based algorithms.

**Potential field (PF):**
![image](https://github.com/user-attachments/assets/c6c481b7-f0bb-4312-8e2a-7b61eaf9df4d)

The potential field method models the robot as a point moving through the configuration space, influenced by an artificial potential field U.
This field U is designed with two main effects:

    • It attracts the robot toward the goal position `Q_final` 
    
    • It repels the robot away from obstacles (represented by the boundaries​, the obstacle region).
    
Ideally, this potential field is shaped so that it has only one global minimum, exactly at the goal, meaning the robot will naturally follow the gradient downhill to the target without getting stuck.
However, in practice, designing such a perfect field is difficult. Most potential fields suffer from local minima, spots where the robot can get stuck without reaching the goal, making this method unreliable on its own in complex environments.
In general, the field U is an additive field consisting of one component that attracts the robot to  and a second component that repels the robot from the boundary. Given that, path planning can be treated as an optimization problem (find the global minimum in U) starting from initial configuration. One of the easiest algorithms to solve this problem is gradient descent. In this case, the negative gradient of U can be considered as a force acting on the robot.

**Rapidly exploring Random Tree(RRT):**

Path planning involves finding a way through a space called the configuration space, where each point represents a specific position and orientation of an object (or multiple objects) in a 2D or 3D environment. This space can be complex due to the shape and movement of the objects involved.
A distance metric is defined on this space to measure how far apart two configurations are.
The collision-free space, ​, contains all the configurations where the object(s) do not collide with any of the static obstacles in the environment. These obstacles are fully defined in the environment, but we don't have an explicit map of collision-free space​. Instead, we rely on collision detection algorithms to check whether a given configuration is safe.
The goal of a single-query path planning problem is to find a continuous path from a starting configuration to a goal configuration​, without any prior processing of the environment.
The basic RRT algorithm works by repeatedly extending a tree structure through the configuration space. In each iteration, the algorithm tries to grow the tree toward a randomly chosen configuration.
Here's how it works step-by-step:
    1. A random configuration q is sampled.
    2. The algorithm finds the nearest existing node in the tree to q.
    3. Using the EXTEND function, it attempts to move from the nearest node toward q, but only by a small fixed step
    4. This motion is handled by a function called NEW_CONFIG, which also checks whether the motion stays inside the collision-free space ​.
Depending on the result, one of three outcomes occurs:
    • Reached: If the tree already has a node very close to q, the algorithm considers it "reached" and adds no new node.
    • Advanced: If the motion toward q is valid (collision-free), a new node is added at the point reached.
    • Trapped: If the path toward q would result in a collision, the attempt is abandoned, and nothing is added to the tree.

**RRT_connect:**

RRT-Connect algorithm can simultaneously generate two trees at the initial and goal points. By employing a greedy strategy, the two trees grow towards each other, enabling faster pathfinding.
In this approach, two trees are grown simultaneously — one starting from the initial configuration and the other from the goal. These trees are maintained separately until they meet, which means a valid path has been found.
During each iteration:
    1. One of the trees is selected to grow toward a random sample.
    
    2. After adding a new node to that tree, the algorithm tries to connect the other tree to the new node by extending it toward the same point.
    
    3. The roles of the two trees are then swapped, so they take turns expanding.
    
This alternating strategy helps both trees explore the collision-free space while continuously trying to connect with each other, increasing the chance of finding a valid path efficiently.
 However, the bidirectional search still exhibits a certain degree of randomness and cannot guarantee an optimal path obtainment.
 ![image](https://github.com/user-attachments/assets/e9fd6fe7-e6b7-423e-9a04-c484ce99d18b)

 **RRT_star:**
 
RRT*, an extension of RRT, which incorporates the parental node reselection and rewiring mechanisms, achieving asymptotic optimality at the cost of increased computation time.
It incrementally builds a tree structure within the configuration space to find the optimal path from the start node `S_init`  to the goal node `S_goal` ​.
Here’s how it works in each iteration:

    1. A random point `S_rand` is sampled from the collision-free space.
    
    2. The algorithm identifies the nearest node `S_nearest`​ in the current tree based on Euclidean distance to `S_rand` .
    
    3. A new node ​`S_new` is generated by taking a step from `S_nearest` toward ​`S_rand`.
    
    4. The algorithm then looks for all existing nodes within a certain radius `r_near` of ​ `S_new`. This group is called the near set `S_near`
    
    5. From this set, it selects the best parent node — the one that results in the lowest total path cost from the start node to `S_new` ​. Then it connects ​`S_new` to that parent.
    
    6. Next, it checks all other nodes in `S_near`  (excluding the new parent). For each, it evaluates whether connecting through ​`S_new`​ would lower their total cost from `S_init` ​. If so, the tree is rewired to make `S_new`​ their new parent.
    
Through repeated iterations of this process — sampling, connecting, and rewiring — RRT* gradually improves the quality of the path. Over time, it converges toward the optimal solution, ensuring asymptotic optimality while still exploring complex environments efficiently.

![image](https://github.com/user-attachments/assets/f20bcca0-cfa2-474d-8ef4-d3c3272375a0)





