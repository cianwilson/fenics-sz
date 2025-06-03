import time
time_start = time.time()
from poisson_2d import solve_poisson_2d
solve_poisson_2d(100, p=2, petsc_options={"ksp_type": "cg", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
time_end = time.time()
elapsedTime = time_end - time_start
print(f'The elapsed time is {elapsedTime} seconds')

