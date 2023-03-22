
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import dccp  


def find_max_radius_in_sector(center:np.ndarray, radius:float, start_angle:float, end_angle:float, n_circle:int=20, seed:int=134):
    """
    在扇形内找到最大数量的不重叠圆的最大半径
    Args:
        center (np.ndarray): 圆心
        radius (float): 扇形半径
        start_angle (float): 起始角度，弧度制
        end_angle (float): 终止角度，弧度制
        n_circle (int, optional): 放置圆的数量. Defaults to 20.
        seed (int, optional): 随机种子，用于随机化算法. Defaults to 134.
    Returns:
        半径：最大半径
        所有圆的圆心坐标
    """
    np.random.seed(seed)
    n = n_circle
    r = cp.Variable()
    c = cp.Variable((n, 2))
    theta_i = np.linspace(start_angle,end_angle,n)
    constr = []
    for i in range(n - 1):
        constr.append(cp.norm(cp.reshape(c[i, :], (1, 2)) - c[i + 1: n, :], 2, axis=1) >= 2 * r)  
    constraint = []
    for i in range(n):
        constr.append(cp.norm(c[i, :]-center)<=radius)
        constr.append(cp.norm(c[i,:] - center, 2) * cp.pos(np.sin(theta_i - start_angle)) >= r * cp.pos(np.sin(theta_i - start_angle)))
        constr.append(cp.norm(c[i,:] - center, 2) * cp.neg(np.sin(theta_i - end_angle)) <= -r * cp.neg(np.sin(theta_i - end_angle)))
    prob = cp.Problem(cp.Maximize(r), constr)
    prob.solve(method="dccp", solver="ECOS", ep=1e-6, max_slack=1e-2)
    return r.value, [(c[i, 0].value, c[i,1].value) for i in range(n)]


# 逆时针摆放三个点描述任意三角形
center = np.array([0, 0])
radius = 10
start_angle = 0
end_angle = np.pi / 6

# 圆的个数
n_circle = 1

max_r, circle_centers = find_max_radius_in_sector(center=center, radius=radius, start_angle=start_angle, end_angle=end_angle, n_circle=n_circle, seed=156)
print("最大圆的半径为：", max_r)
print("所有圆的圆心坐标为：", circle_centers)


# plot

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.set_tight_layout(True)
for i in range(n_circle):
     circ = plt.Circle((circle_centers[i][0], circle_centers[i][1]), max_r, color='b', alpha=0.3)
     ax.add_artist(circ)



wedge = patches.Wedge(center, radius, start_angle * 180 / np.pi, end_angle * 180 / np.pi, fill=False, ec='r')
ax.add_patch(wedge)
ax.set_aspect("equal")  

ax.set_aspect("equal")  
ax.set_xlim([0, radius + 1])  
ax.set_ylim([0, radius + 1])
plt.show()
