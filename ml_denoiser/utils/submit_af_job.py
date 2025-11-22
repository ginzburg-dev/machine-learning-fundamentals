import os
import sys
import af
import argparse
from typing import List

from dataclasses import dataclass, field
from pathlib import Path

from ml_denoiser.config import WORKING_DIR, AF_WRAPPER_PATH, TRAINER_APP_PATH

@dataclass
class Command:
    title: str
    command: str

@dataclass
class CommandBlock:
    title: str
    commands: List[Command]
    service: str = 'torch'

@dataclass
class JobConfig:
    name: str = "torch-train_job"
    command_blocks: list[CommandBlock] = field(default_factory=list)

def submit_job(job_config: JobConfig, parser: str = "generic") -> None:
    """Submit task to AF Renderfarm"""
    job = af.Job(job_config.name)

    for cmd_block in job_config.command_blocks:
        block = af.Block(cmd_block.title, cmd_block.service)   # or custom service name you use
        block.setWorkingDirectory(working_directory=str(WORKING_DIR))
        block.setParser(parser=parser)
        for cmd in cmd_block.commands:
            task = af.Task(cmd.title)
            task.setCommand(cmd.command)
            block.tasks.append(task)

        job.blocks.append(block)
    
    # Print job JSON to verify
    job.output()
    job.send()

if __name__ == "__main__":
    submit_job(JobConfig(
        name=f"torch_{sys.argv[1]}_job",
        command_blocks=[
            CommandBlock(
                title="Training Block",
                commands=[
                    Command(
                        title="Training model",
                        command=f"{AF_WRAPPER_PATH} python {TRAINER_APP_PATH} {' '.join(sys.argv[2:])}"
                    )
                ]
            )
        ]
    ))