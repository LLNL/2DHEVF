# project.py
import flow
from flow import FlowProject


@FlowProject.label
def check_100_iterations(job):
    return job.isfile("control_iterations_f_10.vtu") and \
            job.isfile("final_output.txt")# Check the job at least has > 1500 iterations.


@FlowProject.operation
@flow.cmd
@FlowProject.post(check_100_iterations)
def launch_opti(job):
    import os
    output = job.ws + "/output.txt"
    simulation = "source /g/g92/miguel/workspace/firedrake_setup.sh && \
            srun --output={3} python3 he_volume_frac.py \
            --mu {0:.5f} \
            --enthalpy_scale {1:.5f} \
            --alphabar {4:.8f} \
            --filter {5:.8f} \
            --output_dir {2}".format(job.sp.mu, job.sp.enthalpy_scale, job.ws, output, job.sp.alphabar, job.sp.filter)
    return simulation


@FlowProject.label
def check_design(job):
    return job.isfile(job.id + ".png")


@FlowProject.operation
@flow.cmd
@FlowProject.pre(check_100_iterations)
@FlowProject.post(check_design)
def post_process_design(job):
    parameters = "".join([key + " " + f"{job.sp[key]}" + "\n" for key in job.sp.keys()])
    import os
    post_process = "srun pvpython screenshot_design.py \
            --parameters '{0}' \
            --filename {1} \
            --results_dir {2}".format(parameters, job.id, job.ws)
    return post_process

if __name__ == '__main__':
    FlowProject().main()
