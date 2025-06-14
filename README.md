# RMPC

this repository represent a course work in the subject RMPC

**Problem statement:**

this project's main goal is to achieve path planning, trajectory planning, and motion control for Franka Emika 7DoF robotic manipulator with static obstacles avoidance using different algorithms (PF, RRT_connect, RRT*, RL) for path planning, (joint space interpolation) for trajectory planning, and finally computed torque controller as a motion controller.

**Required Packages:**

```
pip install panda-gym
pip install roboticstoolbox-python
pip install pybullet
pip install pybullet_planning
```


**Modeling The Robot:**

The first step in this problem is to find the forward kinematics of the robot. To obtain the forward kinematics, one can begin by defining the robot's kinematic structure, including joint variables, link lengths, joint angles, and their corresponding transformation matrices using the Denavit-Hartenberg (DH) parameters. Denavit-Hartenberg (DH) transformation matrix between two consecutive links $i$ and $i+1$ is given by the form:

|  aⱼ₋₁   |  ⍺ⱼ₋₁  | θⱼ  |  dⱼ   |   q⁻    |   q⁺   |
|---------|--------|-----|-------|---------|--------|
|    0.0  |   0.0° |  q1 | 0.333 | -166.0° | 166.0° |
|    0.0  | -90.0° |  q2 |   0.0 | -101.0° | 101.0° |
|    0.0  |  90.0° |  q3 | 0.316 | -166.0° | 166.0° |
| 0.0825  |  90.0° |  q4 |   0.0 | -176.0° |  -4.0° |
| -0.0825 | -90.0° |  q5 | 0.384 | -166.0° | 166.0° |
|    0.0  |  90.0° |  q6 |   0.0 |   -1.0° | 215.0° |
|  0.088  |  90.0° |  q7 | 0.107 | -166.0° | 166.0° |

Subsequently, the transformation matrices for each joint can be derived, and these can be multiplied to obtain the transformation matrix that maps the joint variables to the end-effector position and orientation. Finally, the position and orientation of the end-effector can be extracted from the resulting transformation matrix. This process allows for the formal derivation of the forward kinematics for a robotic system.



$$
T_{i,i+1}(\theta) = \begin{bmatrix}
\cos(\theta_i) & -\sin(\theta_i)\cos(\alpha_i) & \sin(\theta_i)\sin(\alpha_i) & a_i\cos(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i)\cos(\alpha_i) & -\cos(\theta_i)\sin(\alpha_i) & a_i\sin(\theta_i) \\
0 & \sin(\alpha_i) & \cos(\alpha_i) & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}
$$


$$
T_{0,n}(\theta) = \prod_{i=0}^{n} T_{i,i-1} = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & p_x \\
r_{21} & r_{22} & r_{23} & p_y \\
r_{31} & r_{32} & r_{33} & p_z \\
0 & 0 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
R_{0,n} & P_{0,n}  \\
0 & 1  
\end{bmatrix}
$$


**Inverse Kinematics problem**

The inverse kinematics (IK) problem for redundant manipulators focuses on computing the joint angles needed to reach a desired end-effector position and orientation while leveraging the system's redundancy. Due to the multiple possible joint-space configurations that can achieve the same end-effector pose, this problem presents significant complexity in robotics.
Common Methods For IK in redundant manipulators

##### 1. Jacobian-Based Methods

Jacobian-based IK methods compute joint-to-end-effector velocity mappings via the Jacobian matrix. Key approaches include the Pseudoinverse (Moore-Penrose) for minimal joint velocities, Damped Least Squares (DLS) for singularity robustness via fixed damping, and Selectively Damped Least Squares (SDLS) with adaptive damping per singular value. While efficient for real-time control, all three methods face challenges near kinematic singularities, requiring careful tuning to balance precision and stability.

##### 2. Optimization-Based Methods

Optimization-based IK frames the problem as constrained minimization of objectives (e.g., joint displacement/energy) subject to limits and obstacles. Key solvers include gradient descent (fast local solutions), SQP (handles nonlinear constraints), and genetic algorithms (global search). While highly flexible for multi-criteria tasks, computational demands can be substantial.
##### 3. Machine Learning Approaches

Recent advances in inverse kinematics (IK) leverage diverse computational approaches. Machine learning methods like neural networks and reinforcement learning enable data-driven IK solutions, offering fast inference for complex high-DOF systems but requiring extensive training datasets. Population-based methods like Particle Swarm Optimization (PSO) are commonly used to solve the Inverse Kinematics problem.

Population-based optimization techniques like improve solution precision for industrial robotic applications. Jacobian-based approaches (pseudoinverse, DLS, SDLS) maintain popularity for real-time control due to their computational efficiency. Gradient descent provides fast convergence but risks local minima, while genetic algorithms explore global solutions at the cost of higher computational overhead and potential convergence to near-optimal rather than exact solutions, Jacobian techniques struggle with singularities. Each method presents distinct trade-offs, the optimal choice depends on application-specific requirements for speed, accuracy, and implementation constraints.

##### Suggested solution for the Inverse Kinematics problem

For our problem we used the ikine_LM to calculate the inverse kinematics of our manipulator when needed, wich depends on Levenberg-Marquadt Numerical Inverse Kinematics Solver (LM)
The LM algorithm blends the concepts of the Gauss-Newton algorithm and gradient descent. It is particularly useful when the Jacobian matrix is ill-conditioned or when the initial guess is far from the solution.

The update rule for the joint angles $θ$ is given by:

$$
\Delta\theta = (J^\top J + \lambda I)^{-1} J^\top (x_d - f(\theta))
$$

Here,  $J$ is the Jacobian matrix of partial derivatives of the end-effector position with respect to the joint angles, $λ$ is a damping factor, and $I$ is the identity matrix.

The joint angles is updated iteratively as follows:

$$
\theta_{k+1} = \theta_k + \Delta\theta
$$

This process is repeated until the change in θ is below a certain threshold, indicating convergence.

**Path Planning Algorithms:**

The path planning of robotic arms in complex and narrow environments with multiple obstacles poses challenges, and path-planning algorithms can generally be categorized into three types: graph-based search, deep learning-based, and sampling-based algorithms.

**Potential field (PF):**


The potential field method models the robot as a point moving through the configuration space, influenced by an artificial potential field U.
This field U is designed with two main effects:

- It attracts the robot toward the goal position Q_final
    
- It repels the robot away from obstacles (represented by the boundaries​, the obstacle region).
    
Ideally, this potential field is shaped so that it has only one global minimum, exactly at the goal, meaning the robot will naturally follow the gradient downhill to the target without getting stuck.
However, in practice, designing such a perfect field is difficult. Most potential fields suffer from local minima, spots where the robot can get stuck without reaching the goal, making this method unreliable on its own in complex environments.
In general, the field U is an additive field consisting of one component that attracts the robot to  and a second component that repels the robot from the boundary. Given that, path planning can be treated as an optimization problem (find the global minimum in U) starting from initial configuration. One of the easiest algorithms to solve this problem is gradient descent. In this case, the negative gradient of U can be considered as a force acting on the robot.



https://github.com/user-attachments/assets/5239612d-5b9d-41f3-86e9-634c4e966fbb



**Rapidly exploring Random Tree(RRT):**

Path planning involves finding a way through a space called the configuration space, where each point represents a specific position and orientation of an object (or multiple objects) in a 2D or 3D environment. This space can be complex due to the shape and movement of the objects involved.
A distance metric is defined on this space to measure how far apart two configurations are.
The collision-free space, contains all the configurations where the object(s) do not collide with any of the static obstacles in the environment. These obstacles are fully defined in the environment, but we don't have an explicit map of collision-free space​. Instead, we rely on collision detection algorithms to check whether a given configuration is safe.
The goal of a single-query path planning problem is to find a continuous path from a starting configuration to a goal configuration​, without any prior processing of the environment.
The basic RRT algorithm works by repeatedly extending a tree structure through the configuration space. In each iteration, the algorithm tries to grow the tree toward a randomly chosen configuration.
Here's how it works step-by-step:

1- A random configuration q is sampled.
    
2- The algorithm finds the nearest existing node in the tree to q.
    
3- Using the EXTEND function, it attempts to move from the nearest node toward q, but only by a small fixed step.
    
4- This motion is handled by a function called NEW_CONFIG, which also checks whether the motion stays inside the collision-free space.
    
Depending on the result, one of three outcomes occurs:

- Reached: If the tree already has a node very close to q, the algorithm considers it "reached" and adds no new node.
    
- Advanced: If the motion toward q is valid (collision-free), a new node is added at the point reached.
    
- Trapped: If the path toward q would result in a collision, the attempt is abandoned, and nothing is added to the tree.

  ![image](https://github.com/user-attachments/assets/c6c481b7-f0bb-4312-8e2a-7b61eaf9df4d)


**RRT_connect:**

RRT-Connect algorithm can simultaneously generate two trees at the initial and goal points. By employing a greedy strategy, the two trees grow towards each other, enabling faster pathfinding.
In this approach, two trees are grown simultaneously — one starting from the initial configuration and the other from the goal. These trees are maintained separately until they meet, which means a valid path has been found.
During each iteration:

1- One of the trees is selected to grow toward a random sample.
    
2- After adding a new node to that tree, the algorithm tries to connect the other tree to the new node by extending it toward the same point.
    
3- The roles of the two trees are then swapped, so they take turns expanding.
    
This alternating strategy helps both trees explore the collision-free space while continuously trying to connect with each other, increasing the chance of finding a valid path efficiently.
 However, the bidirectional search still exhibits a certain degree of randomness and cannot guarantee an optimal path obtainment.

 
![RRT](https://github.com/user-attachments/assets/ba574325-38a7-430f-8b5f-2c2a96ffcd40)




https://github.com/user-attachments/assets/39902602-0215-47af-87f6-134cb930c35f




https://github.com/user-attachments/assets/2d8e182e-fe3e-4932-8f25-d197fb85ad96




 **RRT_star:**
 
RRT*, an extension of RRT, which incorporates the parental node reselection and rewiring mechanisms, achieving asymptotic optimality at the cost of increased computation time.
It incrementally builds a tree structure within the configuration space to find the optimal path from the start node S_init  to the goal node S_goal ​.
Here’s how it works in each iteration:

1- A random point `S_rand` is sampled from the collision-free space.
    
2- The algorithm identifies the nearest node S_nearest in the current tree based on Euclidean distance to S_rand .
    
3- A new node ​S_new is generated by taking a step from S_nearest toward ​S_rand.
    
4- The algorithm then looks for all existing nodes within a certain radius r_near of ​ S_new. This group is called the near set S_near
    
5- From this set, it selects the best parent node — the one that results in the lowest total path cost from the start node to S_new ​. Then it connects ​S_new to that parent.
    
6- Next, it checks all other nodes in S_near (excluding the new parent). For each, it evaluates whether connecting through ​S_new​ would lower their total cost from S_init ​. If so, the tree is rewired to make S_new​ their new parent.
    
Through repeated iterations of this process sampling, connecting, and rewiring  RRT* gradually improves the quality of the path. Over time, it converges toward the optimal solution, ensuring asymptotic optimality while still exploring complex environments efficiently.

![RRTTT](https://github.com/user-attachments/assets/99232967-9cda-46da-a72a-c8e2e05af877)


https://github.com/user-attachments/assets/44808be3-5100-45d5-a987-2a85e116634e



https://github.com/user-attachments/assets/9317ee44-bee5-43f1-8457-a76e2ab44a46



**Reinforcement Learning:**

Reinforcement learning (RL) is a branch of machine learning depended on
the interactions between the agent and the environment to train the agent, it is in-
spired by human’s behavior where the human learns from the consequences of his
actions, so the main idea of RL is to give a reward for the agent, this reward is
predicted using a value function and the Agent will try to maximize the long-term
reward. However applying RL algorithms in continuous action space is a chal-
lenge, one of the most popular methods to use in continuous action space is Deep
Deterministic Policy Gradient(DDPG). DDPG is an off-policy algorithm which
can reuse past experiences stored in a replay buffer to improve learning, DDPG
depends on Actor-Critic architecture to develop the optimal policy. in this report
we will show the results of implementing DDPG algorithm on PandaReach-v3 task, using Panda Gym framework (based on pybullet physics engine) due to its ease of usage in RL applications.
due to the lack of time, RL was implemented only for environments without obstacles.

**Task description in RL framework:**

Reach problem in Panda-gym environment: A target position must be
reached with the gripper.This target position is randomly generated in a volume of
30 cm × 30 cm × 30 cm, with an observation of position and speed of the gripper
(6 coordinates), and three actions including the end-effector movement on each of
movement axis (X, Y, and Z) (3 coordinates). if the reward set as sparse (default
option), it means a reward of 0 is obtained if the entity to move is at the desired
position (with a tolerance of 5 cm), and −1 otherwise, and if the reward is set as
dense, it means that the reward is the opposite of the distance between the entity
to move and the desired position2.

![panda_agent Policy](https://github.com/user-attachments/assets/a4d8fd37-2a61-4d7c-a167-84f6eaa2d8c6)


**Trajectory Planning:**

After obtaining a collision-free path from an obstacle avoidance algorithm, the next step is joint-space trajectory planning to ensure smooth and dynamically feasible motion between consecutive waypoints. For this purpose, we employ the multiple trajectory planning methods, such as quantic jtraj (joint trajectory) function, a widely used method in robotics for generating interpolated trajectories in joint space.

$$
\theta(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5
$$

By leveraging jtraj for joint-space interpolation, we achieve precise, smooth, and dynamically feasible motion for the Franka Emika Panda, while seamlessly integrating obstacle-free paths from sampling-based planners.


**Computed torque controller:**

we can describe the dynamics of a manipulator in this equation :

$$
M(q)\ddot{q} + C(q, \dot{q})\dot{q} + B\dot{q} + G(q) = u
$$

Where:

- $q$: Joint positions (n×1 vector)
- $M(q)$: Inertia matrix (n×n, symmetric positive definite)
- $C(q,\dot{q})$: Coriolis/centrifugal matrix (n×n)
- $B$: Damping/friction matrix (n×n)
- $G(q)$: Gravity vector (n×1)
- $u$: Control input torque (n×1)

now let $q_d$ be the desired trajectory, we introduce $a_q$ :

$$
a_q = \ddot{q_d}(t) + K_d(\dot{q_d}(t) - \dot{q}) +K_p(q_d(t) -q) = \ddot{q}
$$

By substituting $a_q$ in dynamics equation we get:

$$
M(q)a_q + C(q, \dot{q})\dot{q} + B\dot{q} + G(q) = u
$$

$$
u = M(q)( \ddot{q_d}(t) + K_d(\dot{q_d}(t) - \dot{q}) +K_p(q_d(t) -q)) + C(q, \dot{q})\dot{q} + B\dot{q} + G(q)
$$

$$
u = M(q)( \ddot{q_d}(t) + K_d\dot{e} +K_pe) + C(q, \dot{q})\dot{q} + B\dot{q} + G(q)
$$


**Conclusion:**

In this project we implemented various path planning algorithms (PF, RRT_connect, RRT*, RL) to plan a path for Franka Emika 7 DoF manipulator to avoid obstacles, then we planned the trajectory using joint space interpolation, and we controlled the motion by implementing computed torque controller. all simulations performed in pybullet simulator, Panda Gym framework was used in RL training.
As a result for our experements, we concluded that using RRT* is the best for the reviewd method, RL also achieved good results but without obstacle avoidance. due to the lack of time, the collision check while training coudln't be solved because of some interceptions between RL training and collision checker.
PF is not a good method for manipulators because it is only planning the path for the end effector without considering the collision between the manipulator's body and obstacles.


