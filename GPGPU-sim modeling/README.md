GPGPU-sim Architecture modeling projects:

**1. Warp Divergence modeling in SIMT stack**

We had to make significant changes across the following four files:
//update for task3 - comment wherever changes are made in the following files

shader.h -
We create new counters and initialize them to zero here in the header file
(m_n_conditional_divergent_warps and m_n_conditional_branch_warps).

shader.cc -
Inside issue_warp() for total warps that see conditional branch we check if the next_inst->op is
BRANCH_OP or not, since this is just a category of instructions we still need to distinguish
between conditional branch and other types of branch such as uniform or indirect. To do this we
check if the reconv_pc is equal to -1 or not (branch conditional).
Inside issue_warp() for warps that have divergence in their branches we modify the
updateSIMTStack() to return a boolean value (warp_divergence) to see if there is divergence in
warp. Using this bool we check if the opcode is BRANCH_OP and warp_divergence == true. If
both conditions are met we increment the counter.

Inside shader_core_stats::print() - updated to print the counter stats at the end of run.

abstract_hardware.h -
Change simt_stack::update() return type to bool
Change updateSIMTStack() return type to bool

abstract_hardware.cc -
Modify simt_stack::update() to return warp_divergence (which is a bool).
Modify updateSIMTStack() to return bool based on warp_divergence.​

**2. Profile based cache bypassing:**

For each global memory access, I take the address, convert it to an L1 cache block address,
and update a counter using the format of SM id, kernel launch uid, and block address. At the end of the first run, these counters are dumped into a profile file so they can be reused later.

In the second run, I load the profile file before simulation starts and use those saved counters to guide cache bypassing. The bypass decision is made in the load/store path. For each global
memory request, I check the profiled reference count of its corresponding L1 data block for that SM and kernel. If the reference count is less than 3, I bypass the L1 data cache and send the request directly through the bypass path.​
​

I modified shader.cc to add the actual profiling and profile-based bypass logic in the load/store
path, and to make sure the same bypass decision is handled correctly again on the return/fill
path. I also modified shader.h to add the new config options for profiling and profile-based bypass so the simulator can recognize the new flags and threshold cleanly. In gpu-sim.cc and
gpu-sim.h I made changes to store the per-SM, per-kernel block reference profiles, dump them to a file after run 1, and load them back before run 2.