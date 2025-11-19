import os
import sys
import af # pyright: ignore[reportMissingImports]
import argparse

def submit_job(
        name: str,
        command: str,
        working_dir: str | None = os.environ.get("PIPELINE_TOOLS")
) -> None:
    """Submit training task to AF Renderfarm"""
    job = af.Job(name)

    block = af.Block('train', 'torch')   # or custom service name you use
    block.setWorkingDirectory(working_dir)
    block.setParser('generic')

    task = af.Task('training model')
    task.setCommand(command)
    block.tasks.append(task)

    job.blocks.append(block)
    
    # Print job JSON to verify
    job.output()
    job.send()

if __name__ == "__main__":
    submit_job(f"torch_{sys.argv[1]}_job",
        f"tools\\wrappers\\af_wrapper.bat python {" ".join(sys.argv[2:])}")
